# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Tuple, Optional

import torch
from torch import Tensor, nn

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from ..layers import RTDETRHybridEncoder, RTDETRTransformerDecoder, Rtdetr_CdnQueryGenerator, Vision_aug_Text
from .deformable_detr import DeformableDETR, MultiScaleDeformableAttention
from .dino import DINO
from mmdet.structures import OptSampleList, SampleList
from typing import Dict, List, Tuple, Union
from mmdet.utils import OptConfigType
from open_clip import tokenizer
import clip
import open_clip
import logging


@MODELS.register_module()
class RTDETR(DINO):
    r"""Implementation of `DETRs Beat YOLOs on Real-time Object Detection
    <https://arxiv.org/abs/2304.08069>`_

    Code is modified from the `official github repo
    <https://github.com/lyuwenyu/RT-DETR>`_.
    """
    def __init__(self, *args, load_clip_name: str='RN50', **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.load_clip_name = load_clip_name
        # if rtdetr_dn_cfg is not None:
        #     assert 'num_classes' not in rtdetr_dn_cfg and \
        #            'num_queries' not in rtdetr_dn_cfg and \
        #            'hidden_dim' not in rtdetr_dn_cfg, \
        #         'The three keyword args `num_classes`, `embed_dims`, and ' \
        #         '`num_matching_queries` are set in `detector.__init__()`, ' \
        #         'users should not set them in `dn_cfg` config.'
        #     rtdetr_dn_cfg['num_classes'] = self.bbox_head.num_classes
        #     rtdetr_dn_cfg['embed_dims'] = self.embed_dims
        #     rtdetr_dn_cfg['num_matching_queries'] = self.num_queries
        #     rtdetr_dn_cfg['txt_dims'] = self.bbox_head.txt_dims
        # self.dn_query_generator = Rtdetr_CdnQueryGenerator(**rtdetr_dn_cfg)
        
        
    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.encoder = RTDETRHybridEncoder(**self.encoder)
        self.decoder = RTDETRTransformerDecoder(**self.decoder)
        self.embed_dims = self.decoder.embed_dims
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)
        
        self.txt_dims = self.bbox_head.txt_dims
        self.encoder_Vision_aug_Text = Vision_aug_Text(text_channels=self.txt_dims, embed_channels=self.embed_dims)

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super(DeformableDETR, self).init_weights()
        for p in self.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        nn.init.xavier_uniform_(self.memory_trans_fc.weight)

    def pre_transformer(
            self,
            mlvl_feats: Tuple[Tensor],
            txt_feats: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[Dict, Dict]:
        """Process image features before feeding them to the transformer.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            mlvl_feats (tuple[Tensor]): Multi-level features that may have
                different resolutions, output from neck. Each feature has
                shape (bs, dim, h_lvl, w_lvl), where 'lvl' means 'layer'.
            batch_data_samples (list[:obj:`DetDataSample`], optional): The
                batch data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[dict]: The first dict contains the inputs of encoder and the
            second dict contains the inputs of decoder.

            - encoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_encoder()`, which includes 'mlvl_feats' and
              'spatial_shapes'.
            - decoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_decoder()`, which includes 'spatial_shapes' and
              `level_start_index`.
        """
        spatial_shapes = []
        for feat in mlvl_feats:
            spatial_shape = torch._shape_as_tensor(feat)[2:].to(feat.device)
            spatial_shapes.append(spatial_shape)

        # (num_level, 2)
        spatial_shapes = torch.cat(spatial_shapes).view(-1, 2)
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1, )),  # (num_level)
            spatial_shapes.prod(1).cumsum(0)[:-1]))

        encoder_inputs_dict = dict(
            mlvl_feats=mlvl_feats, txt_feats=txt_feats, spatial_shapes=spatial_shapes)
        decoder_inputs_dict = dict(
            memory_mask=None,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=None)
        return encoder_inputs_dict, decoder_inputs_dict

    def forward_encoder(self, mlvl_feats: Tuple[Tensor],
                        txt_feats: Tensor,
                        spatial_shapes: Tensor) -> Dict:
        """Forward with Transformer encoder.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            mlvl_feats (tuple[Tensor]): Multi-level features that may have
                different resolutions, output from neck. Each feature has
                shape (bs, dim, h_lvl, w_lvl), where 'lvl' means 'layer'.
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).

        Returns:
            dict: The output of the Transformer encoder, which includes
            `memory` and `spatial_shapes`.
        """
        mlvl_feats = self.encoder(mlvl_feats)

        feat_flatten = []
        multi_text_feats = []
        for feat in mlvl_feats:
            batch_size, c, h, w = feat.shape
            # [bs, c, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl, c]
            aug_text = self.encoder_Vision_aug_Text(feat, txt_feats)
            multi_text_feats.append(aug_text)
            feat = feat.view(batch_size, c, -1).permute(0, 2, 1)
            feat_flatten.append(feat)

        # (bs, num_feat_points, dim)
        memory = torch.cat(feat_flatten, 1)

        encoder_outputs_dict = dict(
            memory=memory, multi_text_feats=multi_text_feats, memory_mask=None, spatial_shapes=spatial_shapes)
        return encoder_outputs_dict

    def pre_decoder(
        self,
        memory: Tensor,
        multi_text_feats,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        txt_feats: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        """Prepare intermediate variables before entering Transformer decoder,
        such as `query`, `query_pos`, and `reference_points`.

        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points). Will only be used when
                `as_two_stage` is `True`.
            spatial_shapes (Tensor): Spatial shapes of features in all levels.
                With shape (num_levels, 2), last dimension represents (h, w).
                Will only be used when `as_two_stage` is `True`.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[dict]: The decoder_inputs_dict and head_inputs_dict.

            - decoder_inputs_dict (dict): The keyword dictionary args of
              `self.forward_decoder()`, which includes 'query', 'memory',
              `reference_points`, and `dn_mask`. The reference points of
              decoder input here are 4D boxes, although it has `points`
              in its name.
            - head_inputs_dict (dict): The keyword dictionary args of the
              bbox_head functions, which includes `topk_score`, `topk_coords`,
              and `dn_meta` when `self.training` is `True`, else is empty.
        """
        bs, _, c = memory.shape
        # cls_out_features = self.bbox_head.cls_branches[
        #     self.decoder.num_layers].out_features

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)
        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](
                output_memory, txt_feats)
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
            topk_indices.unsqueeze(-1).repeat(1, 1, enc_outputs_class.shape[-1]))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        query = torch.gather(output_memory, 1,
                             topk_indices.unsqueeze(-1).repeat(1, 1, c))
        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = query.detach()  # detach() is not used in DINO
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
            txt_feats=txt_feats,
            multi_text_feats=multi_text_feats,
            dn_mask=dn_mask)
        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            dn_meta=dn_meta) if self.training else dict()
        return decoder_inputs_dict, head_inputs_dict
    
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                      batch_data_samples)
        head_inputs_dict = self.forward_transformer(img_feats,
                                                    txt_feats,
                                                    batch_data_samples)
        losses = self.bbox_head.loss(
            **head_inputs_dict, txt_feats=txt_feats, batch_data_samples=batch_data_samples)

        return losses
    
    def extract_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor, has shape (bs, dim, H, W).

        Returns:
            tuple[Tensor]: Tuple of feature maps from neck. Each feature map
            has shape (bs, dim, H, W).
        """
        txt_feats = None
        if batch_data_samples is None:
            texts = self.texts
            txt_feats = self.text_feats
        elif isinstance(batch_data_samples, dict) and 'texts' in batch_data_samples:
            texts = batch_data_samples['texts']
        elif isinstance(batch_data_samples, list) and hasattr(batch_data_samples[0], 'texts'):
            texts = [data_sample.texts for data_sample in batch_data_samples]
        elif hasattr(self, 'text_feats'):
            texts = self.texts
            txt_feats = self.text_feats
        else:
            raise TypeError('batch_data_samples should be dict or list.')
        
        
        if txt_feats is not None:
            img_feats = self.backbone.forward_image(batch_inputs)
        else:
            img_feats, txt_feats = self.backbone(batch_inputs, texts)
            
        
        if self.with_neck:
            img_feats = self.neck(img_feats, txt_feats)

        return img_feats, txt_feats

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs, has shape (bs, dim, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the input images.
            Each DetDataSample usually contain 'pred_instances'. And the
            `pred_instances` usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                  batch_data_samples)
        head_inputs_dict = self.forward_transformer(img_feats,
                                                    txt_feats,
                                                    batch_data_samples)
        results_list = self.bbox_head.predict(
            **head_inputs_dict,
            txt_feats=txt_feats,
            batch_data_samples=batch_data_samples,
            rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples
    
    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs, has shape (bs, dim, H, W).
            batch_data_samples (List[:obj:`DetDataSample`], optional): The
                batch data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[Tensor]: A tuple of features from ``bbox_head`` forward.
        """
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        head_inputs_dict = self.forward_transformer(img_feats,
                                                    txt_feats,
                                                    batch_data_samples)
        results = self.bbox_head.forward(**head_inputs_dict, txt_feats=txt_feats)
        return results
    
    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        txt_feats: Tuple[Tensor],
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        """Forward process of Transformer.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.
        The difference is that the ground truth in `batch_data_samples` is
        required for the `pre_decoder` to prepare the query of DINO.
        Additionally, DINO inherits the `pre_transformer` method and the
        `forward_encoder` method of DeformableDETR. More details about the
        two methods can be found in `mmdet/detector/deformable_detr.py`.

        Args:
            img_feats (tuple[Tensor]): Tuple of feature maps from neck. Each
                feature map has shape (bs, dim, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            dict: The dictionary of bbox_head function inputs, which always
            includes the `hidden_states` of the decoder output and may contain
            `references` including the initial and intermediate references.
        """
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, txt_feats=txt_feats, batch_data_samples=batch_data_samples)

        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, txt_feats=txt_feats, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict

    def forward_decoder(self,
                        query: Tensor,
                        memory: Tensor,
                        memory_mask: Tensor,
                        reference_points: Tensor,
                        spatial_shapes: Tensor,
                        level_start_index: Tensor,
                        valid_ratios: Tensor,
                        txt_feats: Tensor,
                        dn_mask: Optional[Tensor] = None,
                        **kwargs) -> Dict:
        
        inter_states, references = self.decoder(
            query=query,
            value=memory,
            key_padding_mask=memory_mask,
            self_attn_mask=dn_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=self.bbox_head.reg_branches,
            txt_feats=txt_feats,
            **kwargs)

        if len(query) == self.num_queries:
            # NOTE: This is to make sure label_embeding can be involved to
            # produce loss even if there is no denoising query (no ground truth
            # target in this GPU), otherwise, this will raise runtime error in
            # distributed training.
            inter_states[0] += \
                self.dn_query_generator.label_embedding.weight[0, 0] * 0.0

        decoder_outputs_dict = dict(
            hidden_states=inter_states, references=list(references))
        return decoder_outputs_dict

    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:
        
        load_flag=True
        for key in state_dict:
            if key.startswith('backbone.text_model'):
                load_flag=False
                break

        if load_flag:
            backbone_keys = [key for key in state_dict.keys() if key.startswith('backbone.')]
            for key in backbone_keys:
                del state_dict[key]
                
            try:
                # Load a pre-trained CLIP model
                clip_model, _, _ = open_clip.create_model_and_transforms(self.load_clip_name, pretrained='openai')
                openai_clip_state_dict = clip_model.state_dict()
            except Exception as e:
                error_msgs.append(f"Failed to load pre-trained CLIP model: {e}")
                logging.error(f"Failed to load pre-trained CLIP model: {e}")
                return
            
            
            for key in openai_clip_state_dict:
                if key.startswith('visual.'):
                    new_key = 'backbone.image_model.' + key[len('visual.'):]
                else:
                    new_key = 'backbone.text_model.' + key
                    
                if new_key in self.state_dict():
                    state_dict[new_key] = openai_clip_state_dict[key]
                else:
                    missing_keys.append(new_key)
                    logging.warning(f"Key {new_key} not found in the model's state_dict.")

        
        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )