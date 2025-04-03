# Copyright (c) OpenMMLab. All rights reserved.
import copy
import re
import warnings
from typing import Dict, Optional, Tuple, Union
from mmdet.models.utils import (filter_gt_instances, rename_loss_dict,
                                reweight_loss_dict)
from mmdet.structures.bbox import bbox_project
import random
import torch
import torch.nn as nn
from mmengine.runner.amp import autocast
from torch import Tensor
from collections import Counter
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType
from ..layers import SinePositionalEncoding
from ..layers.transformer.grounding_dino_layers import (
    GroundingDinoTransformerDecoder, GroundingDinoTransformerEncoder, MetaGroundingDinoTransformerEncoder)
from .semi_dino import SemiDINO
from .glip import (create_positive_map, create_positive_map_label_to_token,
                   run_ner)
import torch.distributed as dist
import torch.nn.functional as F
from torch.autograd import Variable

class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, rank, world_size):
        output = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank: ctx.batch_size * (ctx.rank + 1)],
            None,
            None
        )

allgather = AllGather.apply

def clean_label_name(name: str) -> str:
    name = re.sub(r'\(.*\)', '', name)
    name = re.sub(r'_', ' ', name)
    name = re.sub(r'  ', ' ', name)
    return name


def chunks(lst: list, n: int) -> list:
    """Yield successive n-sized chunks from lst."""
    all_ = []
    for i in range(0, len(lst), n):
        data_index = lst[i:i + n]
        all_.append(data_index)
    counter = 0
    for i in all_:
        counter += len(i)
    assert (counter == len(lst))

    return all_


@MODELS.register_module()
class SemiGroundingDINO(SemiDINO):
    """Implementation of `Grounding DINO: Marrying DINO with Grounded Pre-
    Training for Open-Set Object Detection.

    <https://arxiv.org/abs/2303.05499>`_

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/GroundingDINO>`_.
    """

    def __init__(self,
                 language_model,
                 *args,
                 use_autocast=False,
                 **kwargs) -> None:

        self.language_model_cfg = language_model
        self._special_tokens = '. '
        self.use_autocast = use_autocast
        
        self.img_emb_list = []
        self.txt_emb_list = []
        
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        
        # TODO: 创建一个Multi-Modal Meta-Transfer
        '''
        1.首先经过Meta-Transfer[self.pre_encoder]:使用image_feat但是不能使用text_feat （以GroundingDinoTransformerEncoder为主干修改）训练阶段使用
        2.然后经过Transformer Encoder[self.encoder]:需要求该输入特征为（图像+文本或者是图像+图像转文本）
        3.构造视觉转换成前后变换的损失
        4.考虑预训练阶段随机不使用一些文本参加训练
        '''
        self.pre_encoder = MetaGroundingDinoTransformerEncoder(**self.encoder)
        self.encoder = GroundingDinoTransformerEncoder(**self.encoder) 
        self.decoder = GroundingDinoTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

        # text modules
        self.language_model = MODELS.build(self.language_model_cfg)
        self.text_feat_map = nn.Linear(
            self.language_model.language_backbone.body.language_dim,
            self.embed_dims,
            bias=True)

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super().init_weights()
        nn.init.constant_(self.text_feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.text_feat_map.weight.data)
    ############################
    """
    增强文本描述，给文本提示加入到原始的caption里面
    •	original_caption 是一个包含原始描述的词列表。
	•	enhanced_text_prompts 是一个字典，包含需要增强的词及其对应的增强信息（前缀、名称和后缀）。
	•	self._special_tokens 是一个特殊标记，在每个词或增强部分之后添加，用于分隔词语。
    """
    ############################
    def to_enhance_text_prompts(self, original_caption, enhanced_text_prompts):
        caption_string = ''
        tokens_positive = []
        for idx, word in enumerate(original_caption):
            if word in enhanced_text_prompts:
                enhanced_text_dict = enhanced_text_prompts[word]
                if 'prefix' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['prefix']
                start_i = len(caption_string)
                if 'name' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['name']
                else:
                    caption_string += word
                end_i = len(caption_string)
                tokens_positive.append([[start_i, end_i]])

                if 'suffix' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['suffix']
            else:
                tokens_positive.append(
                    [[len(caption_string),
                      len(caption_string) + len(word)]])
                caption_string += word
            caption_string += self._special_tokens
        return caption_string, tokens_positive
    ############################
    """
    这个方法主要用于将原始文本描述转换为带有特殊标记的纯文本描述，同时保留每个词的位置信息，方便后续处理或分析。具体过程如下：
	1.	对于 original_caption 中的每个词：
	•	计算并记录其在 caption_string 中的起始和结束位置。
	•	将词添加到 caption_string 中。
	•	在词后面添加特殊标记。
	2.	返回构建好的 caption_string 和 tokens_positive 列表。
    """
    ############################
    def to_plain_text_prompts(self, original_caption):
        caption_string = ''
        tokens_positive = []
        for idx, word in enumerate(original_caption):
            tokens_positive.append(
                [[len(caption_string),
                  len(caption_string) + len(word)]])
            caption_string += word
            caption_string += self._special_tokens
        return caption_string, tokens_positive

    def get_tokens_and_prompts(
        self,
        original_caption: Union[str, list, tuple],
        custom_entities: bool = False,
        enhanced_text_prompts: Optional[ConfigType] = None
    ) -> Tuple[dict, str, list]:
        """Get the tokens positive and prompts for the caption."""
        if isinstance(original_caption, (list, tuple)) or custom_entities:
            if custom_entities and isinstance(original_caption, str):
                original_caption = original_caption.strip(self._special_tokens)
                original_caption = original_caption.split(self._special_tokens)
                original_caption = list(
                    filter(lambda x: len(x) > 0, original_caption))

            original_caption = [clean_label_name(i) for i in original_caption]

            if custom_entities and enhanced_text_prompts is not None:
                caption_string, tokens_positive = self.to_enhance_text_prompts(
                    original_caption, enhanced_text_prompts)
            else:
                caption_string, tokens_positive = self.to_plain_text_prompts(
                    original_caption)

            # NOTE: Tokenizer in Grounding DINO is different from
            # that in GLIP. The tokenizer in GLIP will pad the
            # caption_string to max_length, while the tokenizer
            # in Grounding DINO will not.
            tokenized = self.language_model.tokenizer(
                [caption_string],
                padding='max_length'
                if self.language_model.pad_to_max else 'longest',
                return_tensors='pt')
            entities = original_caption
        else:
            if not original_caption.endswith('.'):
                original_caption = original_caption + self._special_tokens
            # NOTE: Tokenizer in Grounding DINO is different from
            # that in GLIP. The tokenizer in GLIP will pad the
            # caption_string to max_length, while the tokenizer
            # in Grounding DINO will not.
            tokenized = self.language_model.tokenizer(
                [original_caption],
                padding='max_length'
                if self.language_model.pad_to_max else 'longest',
                return_tensors='pt')
            tokens_positive, noun_phrases = run_ner(original_caption)
            entities = noun_phrases
            caption_string = original_caption

        return tokenized, caption_string, tokens_positive, entities

    def get_positive_map(self, tokenized, tokens_positive):
        positive_map = create_positive_map(
            tokenized,
            tokens_positive,
            max_num_entities=self.bbox_head.cls_branches[
                self.decoder.num_layers].max_text_len)
        positive_map_label_to_token = create_positive_map_label_to_token(
            positive_map, plus=1)
        return positive_map_label_to_token, positive_map

    def get_tokens_positive_and_prompts(
        self,
        original_caption: Union[str, list, tuple],
        custom_entities: bool = False,
        enhanced_text_prompt: Optional[ConfigType] = None,
        tokens_positive: Optional[list] = None,
    ) -> Tuple[dict, str, Tensor, list]:
        """Get the tokens positive and prompts for the caption.

        Args:
            original_caption (str): The original caption, e.g. 'bench . car .'
            custom_entities (bool, optional): Whether to use custom entities.
                If ``True``, the ``original_caption`` should be a list of
                strings, each of which is a word. Defaults to False.

        Returns:
            Tuple[dict, str, dict, str]: The dict is a mapping from each entity
            id, which is numbered from 1, to its positive token id.
            The str represents the prompts.
        """
        if tokens_positive is not None:
            if tokens_positive == -1:
                if not original_caption.endswith('.'):
                    original_caption = original_caption + self._special_tokens
                return None, original_caption, None, original_caption
            else:
                if not original_caption.endswith('.'):
                    original_caption = original_caption + self._special_tokens
                tokenized = self.language_model.tokenizer(
                    [original_caption],
                    padding='max_length'
                    if self.language_model.pad_to_max else 'longest',
                    return_tensors='pt')
                positive_map_label_to_token, positive_map = \
                    self.get_positive_map(tokenized, tokens_positive)

                entities = []
                for token_positive in tokens_positive:
                    instance_entities = []
                    for t in token_positive:
                        instance_entities.append(original_caption[t[0]:t[1]])
                    entities.append(' / '.join(instance_entities))
                return positive_map_label_to_token, original_caption, \
                    positive_map, entities

        chunked_size = self.test_cfg.get('chunked_size', -1)
        if not self.training and chunked_size > 0:
            assert isinstance(original_caption,
                              (list, tuple)) or custom_entities is True
            all_output = self.get_tokens_positive_and_prompts_chunked(
                original_caption, enhanced_text_prompt)
            positive_map_label_to_token, \
                caption_string, \
                positive_map, \
                entities = all_output
        else:
            tokenized, caption_string, tokens_positive, entities = \
                self.get_tokens_and_prompts(
                    original_caption, custom_entities, enhanced_text_prompt)
            positive_map_label_to_token, positive_map = self.get_positive_map(
                tokenized, tokens_positive)
        return positive_map_label_to_token, caption_string, \
            positive_map, entities

    def get_tokens_positive_and_prompts_chunked(
            self,
            original_caption: Union[list, tuple],
            enhanced_text_prompts: Optional[ConfigType] = None):
        chunked_size = self.test_cfg.get('chunked_size', -1)
        original_caption = [clean_label_name(i) for i in original_caption]

        original_caption_chunked = chunks(original_caption, chunked_size)
        ids_chunked = chunks(
            list(range(1,
                       len(original_caption) + 1)), chunked_size)

        positive_map_label_to_token_chunked = []
        caption_string_chunked = []
        positive_map_chunked = []
        entities_chunked = []

        for i in range(len(ids_chunked)):
            if enhanced_text_prompts is not None:
                caption_string, tokens_positive = self.to_enhance_text_prompts(
                    original_caption_chunked[i], enhanced_text_prompts)
            else:
                caption_string, tokens_positive = self.to_plain_text_prompts(
                    original_caption_chunked[i])
            tokenized = self.language_model.tokenizer([caption_string],
                                                      return_tensors='pt')
            if tokenized.input_ids.shape[1] > self.language_model.max_tokens:
                warnings.warn('Inputting a text that is too long will result '
                              'in poor prediction performance. '
                              'Please reduce the --chunked-size.')
            positive_map_label_to_token, positive_map = self.get_positive_map(
                tokenized, tokens_positive)

            caption_string_chunked.append(caption_string)
            positive_map_label_to_token_chunked.append(
                positive_map_label_to_token)
            positive_map_chunked.append(positive_map)
            entities_chunked.append(original_caption_chunked[i])

        return positive_map_label_to_token_chunked, \
            caption_string_chunked, \
            positive_map_chunked, \
            entities_chunked
    
    """
    forward_transformer()调用下面三个 forward_encoder，pre_decoder 和 forward_decoder 函数
    """
    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        text_dict: Dict,
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)
        
        ## Meta-Transfer
        memory_text_by_vis, text_attention_mask_by_vis = self.forward_pre_encoder(
            feat=encoder_inputs_dict['feat'],
            feat_mask=encoder_inputs_dict['feat_mask'],
            feat_pos=encoder_inputs_dict['feat_pos'],
            spatial_shapes=encoder_inputs_dict['spatial_shapes'],
            level_start_index=encoder_inputs_dict['level_start_index'],
            valid_ratios=encoder_inputs_dict['valid_ratios'],
            text_dict=text_dict,img_feats=img_feats) # text_dict不发挥作用,目前保持一致性没有修改，但内部函数不使用文本分支
        
        # ## Transformer Encoder
        # encoder_outputs_dict = self.forward_encoder(
        #     feat=encoder_inputs_dict['feat'],
        #     feat_mask=encoder_inputs_dict['feat_mask'],
        #     feat_pos=encoder_inputs_dict['feat_pos'],
        #     spatial_shapes=encoder_inputs_dict['spatial_shapes'],
        #     level_start_index=encoder_inputs_dict['level_start_index'],
        #     valid_ratios=encoder_inputs_dict['valid_ratios'],
        #     # 文本参数，如果输入文本则存在这些参数，如果没有文本则为使用memory_text_by_vis
        #     text_dict=text_dict)
        if self.training:
            loss_meta_ = self.get_contr_loss(memory_text_by_vis.mean(dim=1, keepdim=True), text_dict['class_embedded'].mean(dim=1, keepdim=True))
        else:
            loss_meta_ = 0.0
            
        # ############save embs##################
        # self.img_emb_list.append(memory_text_by_vis.mean(dim=1, keepdim=False).data.cpu().numpy())
        # # self.img_emb_list.append(text_dict['class_embedded'].mean(dim=1, keepdim=False).data.cpu().numpy())
        # ########################################
        # loss_meta_ = 0.0 
        # bs, num_mv, _ = memory_text_by_vis.shape
        new_text_dict = text_dict
        text_dict['embedded'] = memory_text_by_vis + text_dict['embedded']
        # text_dict['text_token_mask'] = text_attention_mask_by_vis

        # bs, num_mv, _ = memory_text_by_vis.shape
        # text_dict['embedded'] = torch.cat([memory_text_by_vis, text_dict['embedded']], dim=1)
        # text_dict['hidden'] = torch.cat([text_dict['hidden'][:,0:num_mv,:], text_dict['hidden']], dim=1)
        # text_dict['text_token_mask'] = torch.cat([text_attention_mask_by_vis, text_dict['text_token_mask']], dim=1)
        # text_dict['masks'] = torch.cat([text_dict['masks'][:,:,0:num_mv], text_dict['masks']], dim=-1)
        # text_dict['masks'] = torch.cat([text_dict['masks'][:,0:num_mv,:], text_dict['masks']], dim=1)
        # text_dict['position_ids'] = torch.cat([text_dict['position_ids'][:,0:num_mv], text_dict['position_ids']], dim=1)

        encoder_outputs_dict = self.forward_encoder(
            feat=encoder_inputs_dict['feat'],
            feat_mask=encoder_inputs_dict['feat_mask'],
            feat_pos=encoder_inputs_dict['feat_pos'],
            spatial_shapes=encoder_inputs_dict['spatial_shapes'],
            level_start_index=encoder_inputs_dict['level_start_index'],
            valid_ratios=encoder_inputs_dict['valid_ratios'],
            # 文本参数，如果输入文本则存在这些参数，如果没有文本则为使用memory_text_by_vis
            text_dict=new_text_dict)
        
        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)

        text_dict = new_text_dict
        return head_inputs_dict, loss_meta_
        # return head_inputs_dict

    def forward_pre_encoder(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor, spatial_shapes: Tensor,
                        level_start_index: Tensor, valid_ratios: Tensor,
                        text_dict: Dict, img_feats: Tensor) -> Dict:
        memory_text_, text_attention_mask_ = self.pre_encoder(
            query=feat,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,  # for self_attn
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            # for text encoder
            memory_text=text_dict['embedded'],
            text_attention_mask=~text_dict['text_token_mask'],
            position_ids=text_dict['position_ids'],
            text_self_attention_masks=text_dict['masks'],img_feats=img_feats)
        # encoder_outputs_dict = dict(
        #     memory=memory,
        #     memory_mask=feat_mask,
        #     spatial_shapes=spatial_shapes,
        #     memory_text=memory_text,
        #     text_token_mask=text_token_mask)
        # return encoder_outputs_dict
        return memory_text_, text_attention_mask_

    def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor, spatial_shapes: Tensor,
                        level_start_index: Tensor, valid_ratios: Tensor,
                        text_dict: Dict) -> Dict:
        text_token_mask = text_dict['text_token_mask']
        memory, memory_text = self.encoder(
            query=feat,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,  # for self_attn
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            # for text encoder
            memory_text=text_dict['embedded'],
            text_attention_mask=~text_token_mask,
            position_ids=text_dict['position_ids'],
            text_self_attention_masks=text_dict['masks'])
        encoder_outputs_dict = dict(
            memory=memory,
            memory_mask=feat_mask,
            spatial_shapes=spatial_shapes,
            memory_text=memory_text,
            text_token_mask=text_token_mask)
        # encoder_outputs_dict = dict(
        #     memory=memory,
        #     memory_mask=feat_mask,
        #     spatial_shapes=spatial_shapes,
        #     memory_text=memory_text[:,20:,:], # 不把场景信息加入到最终的decoder阶段
        #     text_token_mask=text_token_mask[:,20:]) # 不把场景信息加入到最终的decoder阶段
        return encoder_outputs_dict
    
    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        memory_text: Tensor,
        text_token_mask: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        bs, _, c = memory.shape 

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)

        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](output_memory, memory_text,
                                     text_token_mask)
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].max_text_len
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals

        # NOTE The DINO selects top-k proposals according to scores of
        # multi-class classification, while DeformDETR, where the input
        # is `enc_outputs_class[..., 0]` selects according to scores of
        # binary classification.
        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]

        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        query = self.query_embedding.weight[:, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)
        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask,
            memory_text=memory_text,
            text_attention_mask=~text_token_mask,
        )
        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            dn_meta=dn_meta) if self.training else dict()
        # append text_feats to head_inputs_dict
        head_inputs_dict['memory_text'] = memory_text
        head_inputs_dict['text_token_mask'] = text_token_mask
        return decoder_inputs_dict, head_inputs_dict
    
    # 计算损失函数的地方
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        # text_prompts = [
        #     data_samples.text for data_samples in batch_data_samples
        # ]

        # gt_labels = [
        #     data_samples.gt_instances.labels
        #     for data_samples in batch_data_samples
        # ] # 这里是按照batchsize中每个图片中包含的类别 0 = tensor([12,  7], device='cuda:0') 1= tensor([16, 16], device='cuda:0')...

        text_prompts = []
        gt_labels = []
        # TODO: 根据gt_labels生成一个new_text_prompts
        for data_samples in batch_data_samples:
            text_tuple = data_samples.text
            gt_label = data_samples.gt_instances.labels
            
            if len(text_tuple) > 60:
                gt_lable_list = list(gt_label.cpu().numpy())
                element_counts = Counter(gt_lable_list)
                uni_label = list(set(gt_lable_list)) # 一个图中有多少个类别
                postive_text_tuple = tuple()
                depostive_text_tuple = tuple()
                for i in range(len(text_tuple)):
                    if i in uni_label:
                        postive_text_tuple = postive_text_tuple + (text_tuple[i],)
                    else:
                        depostive_text_tuple = depostive_text_tuple + (text_tuple[i],)
                new_text_tuple =  postive_text_tuple + tuple(random.sample(depostive_text_tuple, 60 - len(postive_text_tuple)))
                new_gt_label = []
                for id in range(len(uni_label)):
                    new_gt_label = new_gt_label + [id] * element_counts[uni_label[id]]
            else:
                new_text_tuple = text_tuple
                new_gt_label = gt_label

            gt_labels.append(torch.tensor(new_gt_label).to(gt_label.device))
            text_prompts.append(new_text_tuple)

        if 'tokens_positive' in batch_data_samples[0]:
            tokens_positive = [
                data_samples.tokens_positive
                for data_samples in batch_data_samples
            ]
            positive_maps = []
            for token_positive, text_prompt, gt_label in zip(
                    tokens_positive, text_prompts, gt_labels):
                tokenized = self.language_model.tokenizer(
                    [text_prompt],
                    padding='max_length'
                    if self.language_model.pad_to_max else 'longest',
                    return_tensors='pt')
                new_tokens_positive = [
                    token_positive[label.item()] for label in gt_label
                ]
                _, positive_map = self.get_positive_map(
                    tokenized, new_tokens_positive)
                positive_maps.append(positive_map)
            new_text_prompts = text_prompts
        else:
            new_text_prompts = []
            positive_maps = []
            if len(set(text_prompts)) == 1:
                # All the text prompts are the same,
                # so there is no need to calculate them multiple times.
                tokenized, caption_string, tokens_positive, _ = \
                    self.get_tokens_and_prompts(
                        text_prompts[0], True)
                new_text_prompts = [caption_string] * len(batch_inputs)
                for gt_label in gt_labels:
                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive)
                    positive_maps.append(positive_map)
            else:
                for text_prompt, gt_label in zip(text_prompts, gt_labels):
                    tokenized, caption_string, tokens_positive, _ = \
                        self.get_tokens_and_prompts(
                            text_prompt, True)
                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive)
                    positive_maps.append(positive_map)
                    new_text_prompts.append(caption_string)

        text_dict = self.language_model(new_text_prompts)
        if self.text_feat_map is not None:
            text_dict['embedded'] = self.text_feat_map(text_dict['embedded'])
        class_embeddeds_li_batch = []
        for i, data_samples in enumerate(batch_data_samples):
            positive_map = positive_maps[i].to(
                batch_inputs.device).bool().float()
            text_token_mask = text_dict['text_token_mask'][i]
            data_samples.gt_instances.positive_maps = positive_map # torch.Size([X, 256]) X为单张图片中不同类个数
            data_samples.gt_instances.text_token_mask = \
                text_token_mask.unsqueeze(0).repeat(
                    len(positive_map), 1) # torch.Size([1, 256])
            ########## 加入get_class_embedded_from_positive_map ###########获取每个batch中所有的class_embedded存入text_dict['class_embedded']
            text_embedded = text_dict['embedded'][i] # torch.Size([57, 256])
            class_positive_map = positive_map[:,:len(text_token_mask)] #torch.Size([X, 57]) X为单张图片中不同类个数
            if class_positive_map.shape[0] > 0:
                class_embeddeds = text_embedded.unsqueeze(0) * class_positive_map.unsqueeze(-1) # torch.Size([57, 256]) * torch.Size([X, 57, 1]) = torch.Size([X, 57, 256])
            else:
                class_positive_map_zeros = torch.zeros(1, len(text_token_mask)).to(batch_inputs.device)
                class_embeddeds = text_embedded.unsqueeze(0) * class_positive_map_zeros.unsqueeze(-1)
            # class_embeddeds_li = []
            # for i in range(class_positive_map.shape[0]): # 这个步骤太慢了，一张图有多少个实例就得走多少遍
            #     indices = torch.where(class_positive_map[i] == 1)[0]
            #     class_embedded = text_embedded[indices]
            #     class_embeddeds_li.append(class_embedded)
            # if len(class_embeddeds_li) > 0:
            #     class_embeddeds = torch.concat(class_embeddeds_li, dim=0).sum(dim=0) / len(class_embeddeds_li)
            # else:
            #     class_embeddeds = torch.zeros(positive_map.shape[1]).to(batch_inputs.device)
            class_embeddeds_li_batch.append((class_embeddeds.sum(dim=1)).mean(dim=0, keepdim=True).unsqueeze(0))

        text_dict['class_embedded'] = torch.concat(class_embeddeds_li_batch, dim=0)

        if self.use_autocast:
            with autocast(enabled=True):
                visual_features = self.extract_feat(batch_inputs)
        else:
            visual_features = self.extract_feat(batch_inputs)

        # head_inputs_dict = self.forward_transformer(visual_features, text_dict,
        #                                     batch_data_samples)
        head_inputs_dict, loss_meta_ = self.forward_transformer(visual_features, text_dict,
                                                    batch_data_samples)


        losses_all = dict()
        losses_sup_ = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)
        losses_sup = rename_loss_dict('sup_', reweight_loss_dict(losses_sup_, 1)) # 设置域值
        losses_all.update(**losses_sup)

        loss_meta = rename_loss_dict('meta_', reweight_loss_dict(loss_meta_, 10)) # 设置域值
        # loss_meta = rename_loss_dict('meta_', reweight_loss_dict(loss_meta_, 0)) # 设置域值
        losses_all.update(**loss_meta)
        return losses_all
    
    # 计算半监督微调时损失函数的地方
    def loss_STCA(self, multi_batch_inputs: Dict[str, Tensor],
             multi_batch_data_samples: Dict[str, SampleList]) -> Union[dict, list]:
        batch_inputs = multi_batch_inputs['sup']
        batch_data_samples = multi_batch_data_samples['sup']
        text_prompts = [
            data_samples.text for data_samples in batch_data_samples
        ]

        gt_labels = [
            data_samples.gt_instances.labels
            for data_samples in batch_data_samples
        ] # 这里是按照batchsize中每个图片中包含的类别 0 = tensor([12,  7], device='cuda:0') 1= tensor([16, 16], device='cuda:0')...

        if 'tokens_positive' in batch_data_samples[0]:
            tokens_positive = [
                data_samples.tokens_positive
                for data_samples in batch_data_samples
            ]
            positive_maps = []
            for token_positive, text_prompt, gt_label in zip(
                    tokens_positive, text_prompts, gt_labels):
                tokenized = self.language_model.tokenizer(
                    [text_prompt],
                    padding='max_length'
                    if self.language_model.pad_to_max else 'longest',
                    return_tensors='pt')
                new_tokens_positive = [
                    token_positive[label.item()] for label in gt_label
                ]
                _, positive_map = self.get_positive_map(
                    tokenized, new_tokens_positive)
                positive_maps.append(positive_map)
            new_text_prompts = text_prompts
        else:
            new_text_prompts = []
            positive_maps = []
            if len(set(text_prompts)) == 1:
                # All the text prompts are the same,
                # so there is no need to calculate them multiple times.
                tokenized, caption_string, tokens_positive, _ = \
                    self.get_tokens_and_prompts(
                        text_prompts[0], True)
                new_text_prompts = [caption_string] * len(batch_inputs)
                for gt_label in gt_labels:
                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive)
                    positive_maps.append(positive_map)
            else:
                for text_prompt, gt_label in zip(text_prompts, gt_labels):
                    tokenized, caption_string, tokens_positive, _ = \
                        self.get_tokens_and_prompts(
                            text_prompt, True)
                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive)
                    positive_maps.append(positive_map)
                    new_text_prompts.append(caption_string)

        text_dict = self.language_model(new_text_prompts)
        if self.text_feat_map is not None:
            text_dict['embedded'] = self.text_feat_map(text_dict['embedded'])
        class_embeddeds_li_batch = []
        for i, data_samples in enumerate(batch_data_samples):
            positive_map = positive_maps[i].to(
                batch_inputs.device).bool().float()
            text_token_mask = text_dict['text_token_mask'][i]
            data_samples.gt_instances.positive_maps = positive_map # torch.Size([X, 256]) X为单张图片中不同类个数
            data_samples.gt_instances.text_token_mask = \
                text_token_mask.unsqueeze(0).repeat(
                    len(positive_map), 1) # torch.Size([1, 256])
            ########## 加入get_class_embedded_from_positive_map ###########获取每个batch中所有的class_embedded存入text_dict['class_embedded']
            text_embedded = text_dict['embedded'][i] # torch.Size([57, 256])
            class_positive_map = positive_map[:,:len(text_token_mask)] #torch.Size([X, 57]) X为单张图片中不同类个数
            if class_positive_map.shape[0] > 0:
                class_embeddeds = text_embedded.unsqueeze(0) * class_positive_map.unsqueeze(-1) # torch.Size([57, 256]) * torch.Size([X, 57, 1]) = torch.Size([X, 57, 256])
            else:
                class_positive_map_zeros = torch.zeros(1, len(text_token_mask)).to(batch_inputs.device)
                class_embeddeds = text_embedded.unsqueeze(0) * class_positive_map_zeros.unsqueeze(-1)
            # class_embeddeds_li = []
            # for i in range(class_positive_map.shape[0]): # 这个步骤太慢了，一张图有多少个实例就得走多少遍
            #     indices = torch.where(class_positive_map[i] == 1)[0]
            #     class_embedded = text_embedded[indices]
            #     class_embeddeds_li.append(class_embedded)
            # if len(class_embeddeds_li) > 0:
            #     class_embeddeds = torch.concat(class_embeddeds_li, dim=0).sum(dim=0) / len(class_embeddeds_li)
            # else:
            #     class_embeddeds = torch.zeros(positive_map.shape[1]).to(batch_inputs.device)
            class_embeddeds_li_batch.append((class_embeddeds.sum(dim=1)).mean(dim=0, keepdim=True).unsqueeze(0))

        text_dict['class_embedded'] = torch.concat(class_embeddeds_li_batch, dim=0)

        if self.use_autocast:
            with autocast(enabled=True):
                visual_features = self.extract_feat(batch_inputs)
        else:
            visual_features = self.extract_feat(batch_inputs)

        # head_inputs_dict = self.forward_transformer(visual_features, text_dict,
        #                                     batch_data_samples)
        head_inputs_dict, loss_meta_ = self.forward_transformer(visual_features, text_dict,
                                                    batch_data_samples)


        losses_all = dict()
        losses_sup_ = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)
        losses_sup = rename_loss_dict('sup_', reweight_loss_dict(losses_sup_, 1)) # 设置域值
        losses_all.update(**losses_sup)

        loss_meta = rename_loss_dict('meta_', reweight_loss_dict(loss_meta_, 10)) # 设置域值
        losses_all.update(**loss_meta)

        ## TODO: 增加教师模型的损失
        '''
        0.导入数据处理方式(监督和无监督的数据，现在是按照softtecher的结构，无监督的数据同时使用两种处理方式)
        1.引入新的Student头[self.bbox_head]和Teacher头[self.bbox_head_teacher](这些功能可以都写在一个头里面SemiGroundingDINOHead，创建两个实例头即可)
        2.构建EMA更新参数方式（Student头学的参数动量更新给Teacher头）
        3.构建get_pseudo_instances_from_teacher_head函数，从无监督的数据中获得原始的数据和结果列表
        4.构建loss_by_psedo_instances函数，从伪样本中学习获取gt，然后给到student head 计算无监督的loss
        '''

        '''
        无监督训练阶段
        '''
        # origin_pseudo_data_samples, batch_info = self.get_pseudo_instances_from_teacher_head(
        #     multi_batch_inputs['unsup_teacher'],
        #     multi_batch_data_samples['unsup_teacher'])
        
        # losses_unsup_ = self.loss_by_psedo_instances(
        #     multi_batch_inputs['unsup_student'],
        #     multi_batch_data_samples['unsup_student'], origin_pseudo_data_samples)
        
        # losses_unsup = rename_loss_dict('unsup_', reweight_loss_dict(losses_unsup_, 1)) # 设置域值
        # losses_all.update(losses_unsup)
        
        return losses_all
    
    def loss_by_psedo_instances(self, batch_inputs: Tensor, batch_data_samples: SampleList, origin_pseudo_data_samples) -> Union[dict, list]:
        """无监督部分计算无监督损失"""
        for data_samples, pseudo_samples in zip(batch_data_samples, origin_pseudo_data_samples):
            data_samples.gt_instances.bboxes = pseudo_samples.gt_instances.bboxes
            data_samples.gt_instances.labels = pseudo_samples.gt_instances.labels
        
        text_prompts = [ # 这个是没有的需要自己重新生成和制造的，可以替换成一个模型来变换成text_特征
            data_samples.text for data_samples in batch_data_samples
        ]

        gt_labels = [
            data_samples.gt_instances.labels
            for data_samples in batch_data_samples
        ]

        if 'tokens_positive' in batch_data_samples[0]:
            tokens_positive = [
                data_samples.tokens_positive
                for data_samples in batch_data_samples
            ]
            positive_maps = []
            for token_positive, text_prompt, gt_label in zip(
                    tokens_positive, text_prompts, gt_labels):
                tokenized = self.language_model.tokenizer(
                    [text_prompt],
                    padding='max_length'
                    if self.language_model.pad_to_max else 'longest',
                    return_tensors='pt')
                new_tokens_positive = [
                    token_positive[label.item()] for label in gt_label
                ]
                _, positive_map = self.get_positive_map(
                    tokenized, new_tokens_positive)
                positive_maps.append(positive_map)
            new_text_prompts = text_prompts
        else:
            new_text_prompts = []
            positive_maps = []
            if len(set(text_prompts)) == 1:
                # All the text prompts are the same,
                # so there is no need to calculate them multiple times.
                tokenized, caption_string, tokens_positive, _ = \
                    self.get_tokens_and_prompts(
                        text_prompts[0], True)
                new_text_prompts = [caption_string] * len(batch_inputs)
                for gt_label in gt_labels:
                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive)
                    positive_maps.append(positive_map)
            else:
                for text_prompt, gt_label in zip(text_prompts, gt_labels):
                    tokenized, caption_string, tokens_positive, _ = \
                        self.get_tokens_and_prompts(
                            text_prompt, True)
                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive)
                    positive_maps.append(positive_map)
                    new_text_prompts.append(caption_string)

        text_dict = self.language_model(new_text_prompts)
        if self.text_feat_map is not None:
            text_dict['embedded'] = self.text_feat_map(text_dict['embedded'])

        for i, data_samples in enumerate(batch_data_samples):
            positive_map = positive_maps[i].to(
                batch_inputs.device).bool().float()
            text_token_mask = text_dict['text_token_mask'][i]
            data_samples.gt_instances.positive_maps = positive_map
            data_samples.gt_instances.text_token_mask = \
                text_token_mask.unsqueeze(0).repeat(
                    len(positive_map), 1)
        if self.use_autocast:
            with autocast(enabled=True):
                visual_features = self.extract_feat(batch_inputs)
        else:
            visual_features = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(visual_features, text_dict,
                                                    batch_data_samples)

        losses_unsup_ = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples) ## 注意这里也应该是有pesdo label的
        
        return losses_unsup_

    @torch.no_grad()
    def get_pseudo_instances_from_teacher_head(self, batch_inputs: Tensor, batch_data_samples: SampleList, rescale: bool = True):
        """Get pseudo instances from teacher head."""
        text_prompts = []  
        enhanced_text_prompts = []
        tokens_positives = []
        for data_samples in batch_data_samples:
            text_prompts.append(data_samples.text) # TODO: text_dict 应该是不存在的，因为unlabel的数据没有(需要重新构建一个可以生成或者代替文本Image2Text)
            if 'caption_prompt' in data_samples:
                enhanced_text_prompts.append(data_samples.caption_prompt)
            else:
                enhanced_text_prompts.append(None)
            tokens_positives.append(data_samples.get('tokens_positive', None))

        if 'custom_entities' in batch_data_samples[0]:
            # Assuming that the `custom_entities` flag
            # inside a batch is always the same. For single image inference
            custom_entities = batch_data_samples[0].custom_entities
        else:
            custom_entities = False
        if len(text_prompts) == 1:
            # All the text prompts are the same,
            # so there is no need to calculate them multiple times.
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(
                    text_prompts[0], custom_entities, enhanced_text_prompts[0],
                    tokens_positives[0])
            ] * len(batch_inputs)
        else:
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(text_prompt,
                                                     custom_entities,
                                                     enhanced_text_prompt,
                                                     tokens_positive)
                for text_prompt, enhanced_text_prompt, tokens_positive in zip(
                    text_prompts, enhanced_text_prompts, tokens_positives)
            ]
        token_positive_maps, text_prompts, _, entities = zip(
            *_positive_maps_and_prompts)

        # image feature extraction
        visual_feats = self.extract_feat(batch_inputs)

        if isinstance(text_prompts[0], list):
            # chunked text prompts, only bs=1 is supported
            assert len(batch_inputs) == 1
            count = 0
            results_list = []

            entities = [[item for lst in entities[0] for item in lst]]

            for b in range(len(text_prompts[0])):
                text_prompts_once = [text_prompts[0][b]]
                token_positive_maps_once = token_positive_maps[0][b]
                text_dict = self.language_model(text_prompts_once)
                # text feature map layer
                if self.text_feat_map is not None:
                    text_dict['embedded'] = self.text_feat_map(
                        text_dict['embedded'])

                batch_data_samples[
                    0].token_positive_map = token_positive_maps_once

                head_inputs_dict = self.forward_transformer(
                    copy.deepcopy(visual_feats), text_dict, batch_data_samples)
                pred_instances = self.bbox_head_teacher.predict(
                    **head_inputs_dict,
                    rescale=rescale,
                    batch_data_samples=batch_data_samples)[0]

                if len(pred_instances) > 0:
                    pred_instances.labels += count
                count += len(token_positive_maps_once)
                results_list.append(pred_instances)
            results_list = [results_list[0].cat(results_list)]
            is_rec_tasks = [False] * len(results_list)
        else:
            # extract text feats
            text_dict = self.language_model(list(text_prompts))
            # text feature map layer
            if self.text_feat_map is not None:
                text_dict['embedded'] = self.text_feat_map(
                    text_dict['embedded'])

            is_rec_tasks = []
            for i, data_samples in enumerate(batch_data_samples):
                if token_positive_maps[i] is not None:
                    is_rec_tasks.append(False)
                else:
                    is_rec_tasks.append(True)
                data_samples.token_positive_map = token_positive_maps[i]
            # TODO: text_dict 应该是不存在的，因为unlabel的数据没有(需要重新构建一个可以生成或者代替文本Image2Text)
            head_inputs_dict = self.forward_transformer(
                visual_feats, text_dict, batch_data_samples)
            results_list = self.bbox_head_teacher.predict(
                hidden_states = head_inputs_dict['hidden_states'],
                references = head_inputs_dict['references'],
                memory_text = head_inputs_dict['memory_text'],
                text_token_mask = head_inputs_dict['text_token_mask'],
                # **head_inputs_dict,
                rescale=rescale,
                batch_data_samples=batch_data_samples)
            
        for data_samples, results in zip(batch_data_samples, results_list):
            data_samples.gt_instances = results

        for data_sample, pred_instances, entity, is_rec_task in zip(
                batch_data_samples, results_list, entities, is_rec_tasks):
            if len(pred_instances) > 0:
                label_names = []
                for labels in pred_instances.labels:
                    if is_rec_task:
                        label_names.append(entity)
                        continue
                    if labels >= len(entity):
                        warnings.warn(
                            'The unexpected output indicates an issue with '
                            'named entity recognition. You can try '
                            'setting custom_entities=True and running '
                            'again to see if it helps.')
                        label_names.append('unobject')
                    else:
                        label_names.append(entity[labels])
                # for visualization
                pred_instances.label_names = label_names
            data_sample.pred_instances = pred_instances

        '''
        filter_gt_instances && get batch_info
        '''
        batch_data_samples = filter_gt_instances(
            batch_data_samples,
            score_thr=0.2) # TODO: 需要设置域值


        for data_samples in batch_data_samples:
            data_samples.gt_instances.bboxes = bbox_project(
                data_samples.gt_instances.bboxes,
                torch.from_numpy(data_samples.homography_matrix).inverse().to(
                    self.data_preprocessor.device), data_samples.ori_shape)
            
        batch_info = {}
        # batch_info = {
        #     'feat': head_inputs_dict,
        #     'img_shape': [],
        #     'homography_matrix': [],
        #     'metainfo': []
        # }
        # for data_samples in batch_data_samples:
        #     batch_info['img_shape'].append(data_samples.img_shape)
        #     batch_info['homography_matrix'].append(
        #         torch.from_numpy(data_samples.homography_matrix).to(
        #             self.data_preprocessor.device))
        #     batch_info['metainfo'].append(data_samples.metainfo)

        return batch_data_samples, batch_info

    ### 推理时候用的函数
    def predict(self, batch_inputs, batch_data_samples, rescale: bool = True):
        text_prompts = []
        enhanced_text_prompts = []
        tokens_positives = []
        for data_samples in batch_data_samples:
            text_prompts.append(data_samples.text)
            if 'caption_prompt' in data_samples:
                enhanced_text_prompts.append(data_samples.caption_prompt)
            else:
                enhanced_text_prompts.append(None)
            tokens_positives.append(data_samples.get('tokens_positive', None))

        if 'custom_entities' in batch_data_samples[0]:
            # Assuming that the `custom_entities` flag
            # inside a batch is always the same. For single image inference
            custom_entities = batch_data_samples[0].custom_entities
        else:
            custom_entities = False
        if len(text_prompts) == 1:
            # All the text prompts are the same,
            # so there is no need to calculate them multiple times.
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(
                    text_prompts[0], custom_entities, enhanced_text_prompts[0],
                    tokens_positives[0])
            ] * len(batch_inputs)
        else:
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(text_prompt,
                                                     custom_entities,
                                                     enhanced_text_prompt,
                                                     tokens_positive)
                for text_prompt, enhanced_text_prompt, tokens_positive in zip(
                    text_prompts, enhanced_text_prompts, tokens_positives)
            ]
        token_positive_maps, text_prompts, _, entities = zip(
            *_positive_maps_and_prompts)

        # image feature extraction
        visual_feats = self.extract_feat(batch_inputs)

        if isinstance(text_prompts[0], list):
            # chunked text prompts, only bs=1 is supported
            assert len(batch_inputs) == 1
            count = 0
            results_list = []

            entities = [[item for lst in entities[0] for item in lst]]

            for b in range(len(text_prompts[0])):
                text_prompts_once = [text_prompts[0][b]]
                token_positive_maps_once = token_positive_maps[0][b]
                text_dict = self.language_model(text_prompts_once)
                # text feature map layer
                if self.text_feat_map is not None:
                    text_dict['embedded'] = self.text_feat_map(
                        text_dict['embedded'])

                batch_data_samples[
                    0].token_positive_map = token_positive_maps_once

                head_inputs_dict = self.forward_transformer(
                    copy.deepcopy(visual_feats), text_dict, batch_data_samples)
                pred_instances = self.bbox_head.predict(
                    **head_inputs_dict,
                    rescale=rescale,
                    batch_data_samples=batch_data_samples)[0]

                if len(pred_instances) > 0:
                    pred_instances.labels += count
                count += len(token_positive_maps_once)
                results_list.append(pred_instances)
            results_list = [results_list[0].cat(results_list)]
            is_rec_tasks = [False] * len(results_list)
        else:
            # extract text feats
            text_dict = self.language_model(list(text_prompts))
            # text feature map layer
            if self.text_feat_map is not None:
                text_dict['embedded'] = self.text_feat_map(
                    text_dict['embedded'])

            is_rec_tasks = []
            for i, data_samples in enumerate(batch_data_samples):
                if token_positive_maps[i] is not None:
                    is_rec_tasks.append(False)
                else:
                    is_rec_tasks.append(True)
                data_samples.token_positive_map = token_positive_maps[i]

            # head_inputs_dict = self.forward_transformer(
            #     visual_feats, text_dict, batch_data_samples)
            
            head_inputs_dict, _ = self.forward_transformer(
                visual_feats, text_dict, batch_data_samples)
            
            # #############save embs #####################
            # import numpy as np
            # img_emb = np.concatenate(self.img_emb_list)
            # # txt_emb = np.concatenate(self.txt_emb_list)
            # np.save('img_emb.npy', img_emb, allow_pickle=True)
            # # np.save('txt_emb.npy', img_emb, allow_pickle=True)
            # ##################################
            
            results_list = self.bbox_head.predict(
                **head_inputs_dict,
                rescale=rescale,
                batch_data_samples=batch_data_samples)

        for data_sample, pred_instances, entity, is_rec_task in zip(
                batch_data_samples, results_list, entities, is_rec_tasks):
            if len(pred_instances) > 0:
                label_names = []
                for labels in pred_instances.labels:
                    if is_rec_task:
                        label_names.append(entity)
                        continue
                    if labels >= len(entity):
                        warnings.warn(
                            'The unexpected output indicates an issue with '
                            'named entity recognition. You can try '
                            'setting custom_entities=True and running '
                            'again to see if it helps.')
                        label_names.append('unobject')
                    else:
                        label_names.append(entity[labels])
                # for visualization
                pred_instances.label_names = label_names
            data_sample.pred_instances = pred_instances
        return batch_data_samples

    def get_contr_loss(self, image_feat, text_feat, idx=None, label=None, config=None):
        """
        Args:
            image_feat, text_feat: normalized
        Returns: contrastive loss
        """
        # assert image_feat.size(-1) == self.embed_dim
        # assert text_feat.size(-1) == self.embed_dim

        # image_feat_all = allgather(image_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        # text_feat_all = allgather(text_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        image_feat_all = F.normalize(image_feat[:,0,:])
        text_feat_all = F.normalize(text_feat[:,0,:])
        logits = image_feat_all @ text_feat_all.t() / 0.07
        # print(logits)
        bsz = image_feat_all.shape[0]

        if idx is None:
            labels = torch.arange(bsz, device=image_feat.device)  # 注意这里batchsize一定要大于1，不然loss为0
            loss_i2t = F.cross_entropy(logits, labels)
            loss_t2i = F.cross_entropy(logits.t(), labels)
        else:
            idx = idx.view(-1, 1)
            # assert idx.size(0) == image_feat.size(0)

            ## 生成对角阵
            # idx_all = allgather(idx, torch.distributed.get_rank(), torch.distributed.get_world_size())
            idx_all = idx
            pos_idx = torch.eq(idx_all, idx_all.t()).float()
            labels = pos_idx / pos_idx.sum(dim=1, keepdim=True)

            loss_i2t = -torch.sum(F.log_softmax(logits, dim=1) * labels, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(logits.t(), dim=1) * labels, dim=1).mean()

        loss = (loss_i2t + loss_t2i) / 2
        loss_dict = {"contr_loss": loss}
        return loss_dict

    def get_triplet_loss(self, image_feat, text_feat, margin=0.2, max_violation=False):

        # assert image_feat.size(-1) == self.embed_dim
        # assert text_feat.size(-1) == self.embed_dim

        # image_feat_all = allgather(image_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        # text_feat_all = allgather(text_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        image_feat_all = F.normalize(image_feat[:,0,:])
        text_feat_all = F.normalize(text_feat[:,0,:])
        scores = image_feat_all @ text_feat_all.t()

        # print(logits)
        bsz = image_feat_all.shape[0]


        diagonal = scores.diag().view(bsz, 1)

        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (margin + scores - d2).clamp(min=0)

        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda(device=image_feat.device)
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        if max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        sum_cost_s = cost_s.sum()
        sum_cost_im = cost_im.sum()
    
        loss_dict = {"contr_loss": sum_cost_s + sum_cost_im}
        return loss_dict