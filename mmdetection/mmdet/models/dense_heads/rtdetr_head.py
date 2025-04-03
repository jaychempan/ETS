# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_overlaps
from mmdet.utils import OptConfigType, InstanceList, reduce_mean, OptMultiConfig
from ..losses import RTDETRVarifocalLoss
from .dino_head import DINOHead
import copy
from typing import Dict, List, Tuple
from mmcv.cnn import Linear
from mmengine.model import BaseModule, ModuleList
from ..layers import inverse_sigmoid
from mmdet.structures import SampleList
import json, os
from mmengine.structures import InstanceData

class MLP_WO_ACT(BaseModule):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = ModuleList(
            Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x) if i < self.num_layers - 1 else layer(x)
        return x


@MODELS.register_module()
class ContrastiveHead(BaseModule):
    """Contrastive Head for YOLO-World
    compute the region-text scores according to the
    similarity between image and text features
    Args:
        embed_dims (int): embed dim of text and image features
    """

    def __init__(self,
                 txt_dims: int,
                 embed_dims: int,) -> None:

        super().__init__()

        self.bias = nn.Parameter(torch.zeros([]))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        # CLIP权重
        # self.logit_scale = nn.Parameter(torch.tensor(4.6052))
        # self.logit_scale.requires_grad = False

        # self.img_proj = Linear(embed_dims, txt_dims)
        self.img_proj = MLP_WO_ACT(embed_dims, embed_dims*2, txt_dims, 2)
        # self.txt_proj = Linear(txt_dims, embed_dims)

    def forward(self, visual_feat: Tensor, text_feat: Tensor) -> Tensor:
        """Forward function of contrastive learning."""
        visual_feat = self.img_proj(visual_feat)
        
        visual_feat = F.normalize(visual_feat, dim=-1, p=2)
        text_feat = F.normalize(text_feat, dim=-1, p=2)
                
        logits_similarity = torch.matmul(visual_feat, text_feat.transpose(1, 2))
        logits_similarity = logits_similarity * self.logit_scale.exp() + self.bias
        return logits_similarity



@MODELS.register_module()
class RTDETRHead(DINOHead):
    r"""Head of the DETRs Beat YOLOs on Real-time Object Detection

    Code is modified from the `official github repo
    <https://github.com/lyuwenyu/RT-DETR>`_.

    More details can be found in the `paper
    <https://arxiv.org/abs/2304.08069>`_ .
    """

    def __init__(self,
                 txt_dims: int = 1024,
                 out_predict_flag: bool = False,
                 embed_dims: int = 256,
                 share_pred_layer: bool = False,
                 **kwargs):
        self.txt_dims = txt_dims
        self.embed_dims = embed_dims
        self.share_pred_layer = share_pred_layer
        self.out_predict_flag = out_predict_flag
        super().__init__(**kwargs)
    
    def _init_layers(self) -> None:
        """Initialize classification branch and regression branch of head."""
        fc_cls = ContrastiveHead(self.txt_dims, self.embed_dims)
        # fc_cls = Linear(self.txt_dims, self.embed_dims)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, 4))
        reg_branch = nn.Sequential(*reg_branch)

        if self.share_pred_layer:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(self.num_pred_layer)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(self.num_pred_layer)])  # num_pre_layer:7 last_is_encoder_cls
        else:
            self.cls_branches = nn.ModuleList(
                [copy.deepcopy(fc_cls) for _ in range(self.num_pred_layer)])
            self.reg_branches = nn.ModuleList([
                copy.deepcopy(reg_branch) for _ in range(self.num_pred_layer)
            ])
    
    def loss(self, hidden_states: Tensor, references: List[Tensor],
             enc_outputs_class: Tensor, enc_outputs_coord: Tensor,
             txt_feats: Tensor,
             batch_data_samples: SampleList, dn_meta: Dict[str, int]) -> dict:
        
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        outs = self(hidden_states, references, txt_feats)
        loss_inputs = outs + (enc_outputs_class, enc_outputs_coord,
                              batch_gt_instances, batch_img_metas, dn_meta)
        losses = self.loss_by_feat(*loss_inputs)
        return losses
    
    def _loss_dn_single(self, dn_cls_scores: Tensor, dn_bbox_preds: Tensor,
                        batch_gt_instances: InstanceList,
                        batch_img_metas: List[dict],
                        dn_meta: Dict[str, int]) -> Tuple[Tensor]:
        """Denoising loss for outputs from a single decoder layer.

        Args:
            dn_cls_scores (Tensor): Classification scores of a single decoder
                layer in denoising part, has shape (bs, num_denoising_queries,
                cls_out_channels).
            dn_bbox_preds (Tensor): Regression outputs of a single decoder
                layer in denoising part. Each is a 4D-tensor with normalized
                coordinate format (cx, cy, w, h) and has shape
                (bs, num_denoising_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls`, `loss_box` and
            `loss_iou`.
        """
        cls_reg_targets = self.get_dn_targets(batch_gt_instances,
                                              batch_img_metas, dn_meta)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = dn_cls_scores.reshape(-1, dn_cls_scores.shape[-1])
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = \
            num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if len(cls_scores) > 0:
            if isinstance(self.loss_cls, RTDETRVarifocalLoss):
                # bg_class_ind = self.num_classes
                bg_class_ind = cls_scores.shape[-1]
                pos_inds = ((labels >= 0)
                            & (labels < bg_class_ind)).nonzero().squeeze(1)
                cls_iou_targets = label_weights.new_zeros(cls_scores.shape)
                pos_bbox_targets = bbox_targets[pos_inds]
                pos_decode_bbox_targets = bbox_cxcywh_to_xyxy(pos_bbox_targets)
                pos_bbox_pred = dn_bbox_preds.reshape(-1, 4)[pos_inds]
                pos_decode_bbox_pred = bbox_cxcywh_to_xyxy(pos_bbox_pred)
                pos_labels = labels[pos_inds]
                cls_iou_targets[pos_inds, pos_labels] = bbox_overlaps(
                    pos_decode_bbox_pred.detach(),
                    pos_decode_bbox_targets,
                    is_aligned=True)
                loss_cls = self.loss_cls(
                    cls_scores, cls_iou_targets, avg_factor=cls_avg_factor)
            else:
                loss_cls = self.loss_cls(
                    cls_scores,
                    labels,
                    label_weights,
                    avg_factor=cls_avg_factor)
        else:
            loss_cls = torch.zeros(
                1, dtype=cls_scores.dtype, device=cls_scores.device)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, dn_bbox_preds):
            img_h, img_w = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = dn_bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        return loss_cls, loss_bbox, loss_iou

    def loss_by_feat_single(self, cls_scores: Tensor, bbox_preds: Tensor,
                            batch_gt_instances: InstanceList,
                            batch_img_metas: List[dict]) -> Tuple[Tensor]:
        """Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images, has shape (bs, num_queries, cls_out_channels).
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape (bs, num_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls`, `loss_box` and
            `loss_iou`.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           batch_gt_instances, batch_img_metas)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, cls_scores.shape[-1])
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if isinstance(self.loss_cls, RTDETRVarifocalLoss):
            # bg_class_ind = self.num_classes
            bg_class_ind = cls_scores.shape[-1]
            pos_inds = ((labels >= 0)
                        & (labels < bg_class_ind)).nonzero().squeeze(1)
            cls_iou_targets = label_weights.new_zeros(cls_scores.shape)
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_decode_bbox_targets = bbox_cxcywh_to_xyxy(pos_bbox_targets)
            pos_bbox_pred = bbox_preds.reshape(-1, 4)[pos_inds]
            pos_decode_bbox_pred = bbox_cxcywh_to_xyxy(pos_bbox_pred)
            pos_labels = labels[pos_inds]
            cls_iou_targets[pos_inds, pos_labels] = bbox_overlaps(
                pos_decode_bbox_pred.detach(),
                pos_decode_bbox_targets,
                is_aligned=True)
            loss_cls = self.loss_cls(
                cls_scores, cls_iou_targets, avg_factor=cls_avg_factor)
        else:
            loss_cls = self.loss_cls(
                cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, bbox_preds):
            img_h, img_w, = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        return loss_cls, loss_bbox, loss_iou

    def forward(self, hidden_states: Tensor,
                references: List[Tensor], txt_feats: Tensor) -> Tuple[Tensor, Tensor]:
        all_layers_outputs_classes = []
        all_layers_outputs_coords = []

        for layer_id in range(hidden_states.shape[0]):
            reference = inverse_sigmoid(references[layer_id])
            # NOTE The last reference will not be used.
            hidden_state = hidden_states[layer_id]
            outputs_class = self.cls_branches[layer_id](hidden_state, txt_feats)
            tmp_reg_preds = self.reg_branches[layer_id](hidden_state)
            if reference.shape[-1] == 4:
                # When `layer` is 0 and `as_two_stage` of the detector
                # is `True`, or when `layer` is greater than 0 and
                # `with_box_refine` of the detector is `True`.
                tmp_reg_preds += reference
            else:
                # When `layer` is 0 and `as_two_stage` of the detector
                # is `False`, or when `layer` is greater than 0 and
                # `with_box_refine` of the detector is `False`.
                assert reference.shape[-1] == 2
                tmp_reg_preds[..., :2] += reference
            outputs_coord = tmp_reg_preds.sigmoid()
            all_layers_outputs_classes.append(outputs_class)
            all_layers_outputs_coords.append(outputs_coord)

        all_layers_outputs_classes = torch.stack(all_layers_outputs_classes)
        all_layers_outputs_coords = torch.stack(all_layers_outputs_coords)

        return all_layers_outputs_classes, all_layers_outputs_coords

    def predict(self,
                hidden_states: Tensor,
                references: List[Tensor],
                txt_feats: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> InstanceList:
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        outs = self(hidden_states, references, txt_feats)

        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, rescale=rescale)
        return predictions

    def predict_by_feat(self,
                        all_layers_cls_scores: Tensor,
                        all_layers_bbox_preds: Tensor,
                        batch_img_metas: List[Dict],
                        rescale: bool = False) -> InstanceList:
        cls_scores = all_layers_cls_scores[-1]
        bbox_preds = all_layers_bbox_preds[-1]

        result_list = []
        for img_id in range(len(batch_img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_meta = batch_img_metas[img_id]
            results = self._predict_by_feat_single(cls_score, bbox_pred,
                                                   img_meta, rescale)
            if self.out_predict_flag:
                metadata_dict = {
                    'img_meta': img_meta,
                    'scores': results.scores.tolist(),
                    'labels': results.labels.tolist(),
                    'bboxes': results.bboxes.tolist()
                }
                filename = img_meta['img_path'].split('/')[-1].split('.')[0]
                dataset_name = img_meta['img_path'].split('/')[-3]
                save_path = f'save_predict_result/{dataset_name}/{filename}.json'
                
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'w') as file:
                    json.dump(metadata_dict, file)
                
            result_list.append(results)
            
        return result_list
    
    def _predict_by_feat_single(self,
                                cls_score: Tensor,
                                bbox_pred: Tensor,
                                img_meta: dict,
                                rescale: bool = True) -> InstanceData:
        """Transform outputs from the last decoder layer into bbox predictions
        for each image.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_queries, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from the last decoder layer
                for each image, with coordinate format (cx, cy, w, h) and
                shape [num_queries, 4].
            img_meta (dict): Image meta info.
            rescale (bool): If True, return boxes in original image
                space. Default True.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(cls_score) == len(bbox_pred)  # num_queries
        num_bbox, num_cls = cls_score.shape
        max_per_img = self.test_cfg.get('max_per_img', len(cls_score))
        img_shape = img_meta['img_shape']
        # exclude background
        if self.loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            det_labels = indexes % num_cls
            bbox_index = indexes // num_cls
            bbox_pred = bbox_pred[bbox_index]
        else:
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
            bbox_pred = bbox_pred[bbox_index]
            det_labels = det_labels[bbox_index]

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            assert img_meta.get('scale_factor') is not None
            det_bboxes /= det_bboxes.new_tensor(
                img_meta['scale_factor']).repeat((1, 2))

        results = InstanceData()
        results.bboxes = det_bboxes
        results.scores = scores
        results.labels = det_labels
        return results