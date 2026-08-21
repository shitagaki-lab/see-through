from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from torch import Tensor

from mmdet.models.utils import multi_apply
from mmdet.registry import MODELS
from mmdet.structures.bbox import distance2bbox
from mmdet.utils import InstanceList, OptInstanceList, reduce_mean
from mmdet.models.dense_heads.rtmdet_ins_head import RTMDetInsHead, RTMDetInsSepBNHead

from mmdet.utils import AvoidCUDAOOM



def sthgoeswrong(logits):
    return torch.any(torch.isnan(logits)) or torch.any(torch.isinf(logits))

from time import time

@MODELS.register_module(force=True)
class RTMDetInsHeadCustom(RTMDetInsHead):

    def loss_by_feat(self,
                     cls_scores: List[Tensor],
                     bbox_preds: List[Tensor],
                     kernel_preds: List[Tensor],
                     mask_feat: Tensor,
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict],
                     batch_gt_instances_ignore: OptInstanceList = None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Decoded box for each scale
                level with shape (N, num_anchors * 4, H, W) in
                [tl_x, tl_y, br_x, br_y] format.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_imgs = len(batch_img_metas)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)
        flatten_cls_scores = torch.cat([
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.cls_out_channels)
            for cls_score in cls_scores
        ], 1)
        flatten_kernels = torch.cat([
            kernel_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                    self.num_gen_params)
            for kernel_pred in kernel_preds
        ], 1)
        decoded_bboxes = []
        for anchor, bbox_pred in zip(anchor_list[0], bbox_preds):
            anchor = anchor.reshape(-1, 4)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            bbox_pred = distance2bbox(anchor, bbox_pred)
            decoded_bboxes.append(bbox_pred)

        flatten_bboxes = torch.cat(decoded_bboxes, 1)
        for gt_instances in batch_gt_instances:
            gt_instances.masks = gt_instances.masks.to_tensor(
                dtype=torch.bool, device=device)

        cls_reg_targets = self.get_targets(
            flatten_cls_scores,
            flatten_bboxes,
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)
        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         assign_metrics_list, sampling_results_list) = cls_reg_targets

        losses_cls, losses_bbox,\
            cls_avg_factors, bbox_avg_factors = multi_apply(
                self.loss_by_feat_single,
                cls_scores,
                decoded_bboxes,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                assign_metrics_list,
                self.prior_generator.strides)

        cls_avg_factor = reduce_mean(sum(cls_avg_factors)).clamp_(min=1).item()
        losses_cls = list(map(lambda x: x / cls_avg_factor, losses_cls))

        bbox_avg_factor = reduce_mean(
            sum(bbox_avg_factors)).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))

        loss_mask = self.loss_mask_by_feat(mask_feat, flatten_kernels,
                                           sampling_results_list,
                                           batch_gt_instances)
        loss = dict(
            loss_cls=losses_cls, loss_bbox=losses_bbox, loss_mask=loss_mask)

        return loss


    def _mask_predict_by_feat_single(self, mask_feat: Tensor, kernels: Tensor,
                                     priors: Tensor) -> Tensor:

        ori_maskfeat = mask_feat

        num_inst = priors.shape[0]
        h, w = mask_feat.size()[-2:]
        if num_inst < 1:
            return torch.empty(
                size=(num_inst, h, w),
                dtype=mask_feat.dtype,
                device=mask_feat.device)
        if len(mask_feat.shape) < 4:
            mask_feat.unsqueeze(0)

        coord = self.prior_generator.single_level_grid_priors(
            (h, w), level_idx=0, device=mask_feat.device).reshape(1, -1, 2)
        num_inst = priors.shape[0]
        points = priors[:, :2].reshape(-1, 1, 2)
        strides = priors[:, 2:].reshape(-1, 1, 2)
        relative_coord = (points - coord).permute(0, 2, 1) / (
            strides[..., 0].reshape(-1, 1, 1) * 8)
        relative_coord = relative_coord.reshape(num_inst, 2, h, w)

        mask_feat = torch.cat(
            [relative_coord,
             mask_feat.repeat(num_inst, 1, 1, 1)], dim=1)
        weights, biases = self.parse_dynamic_params(kernels)

        fp16_used = weights[0].dtype == torch.float16

        n_layers = len(weights)
        x = mask_feat.reshape(1, -1, h, w)
        for i, (weight, bias) in enumerate(zip(weights, biases)):
            with torch.amp.autocast("cuda", enabled=False):
                if fp16_used:
                    weight = weight.to(torch.float32)
                    bias = bias.to(torch.float32)
                x = F.conv2d(
                    x, weight, bias=bias, stride=1, padding=0, groups=num_inst)
                if i < n_layers - 1:
                    x = F.relu(x)

        if fp16_used:
            x = torch.clip(x, -8192, 8192)
        if sthgoeswrong(x):
            torch.save({'mask_feat': ori_maskfeat, 'kernels': kernels, 'priors': priors}, 'maskhead_nan_input.pt')
            raise Exception('Mask Head NaN')

        x = x.reshape(num_inst, h, w)
        return x

    def loss_mask_by_feat(self, mask_feats: Tensor, flatten_kernels: Tensor,
                          sampling_results_list: list,
                          batch_gt_instances: InstanceList) -> Tensor:
        batch_pos_mask_logits = []
        pos_gt_masks = []
        ignore_masks = []
        for idx, (mask_feat, kernels, sampling_results,
                  gt_instances) in enumerate(
                      zip(mask_feats, flatten_kernels, sampling_results_list,
                          batch_gt_instances)):
            pos_priors = sampling_results.pos_priors
            pos_inds = sampling_results.pos_inds
            pos_kernels = kernels[pos_inds]  # n_pos, num_gen_params
            pos_mask_logits = self._mask_predict_by_feat_single(
                mask_feat, pos_kernels, pos_priors)
            if gt_instances.masks.numel() == 0:
                gt_masks = torch.empty_like(gt_instances.masks)
                if gt_masks.shape[0] > 0:
                    ignore = torch.zeros(gt_masks.shape[0], dtype=torch.bool).to(device=gt_masks.device)
                    ignore_masks.append(ignore)
            else:
                gt_masks = gt_instances.masks[
                    sampling_results.pos_assigned_gt_inds, :]
                ignore_masks.append(gt_instances.ignore_mask[sampling_results.pos_assigned_gt_inds])
            batch_pos_mask_logits.append(pos_mask_logits)
            pos_gt_masks.append(gt_masks)

        pos_gt_masks = torch.cat(pos_gt_masks, 0)
        batch_pos_mask_logits = torch.cat(batch_pos_mask_logits, 0)
        ignore_masks = torch.logical_not(torch.cat(ignore_masks, 0))

        pos_gt_masks = pos_gt_masks[ignore_masks]
        batch_pos_mask_logits = batch_pos_mask_logits[ignore_masks]


        # avg_factor
        num_pos = batch_pos_mask_logits.shape[0]
        num_pos = reduce_mean(mask_feats.new_tensor([num_pos
                                                     ])).clamp_(min=1).item()

        if batch_pos_mask_logits.shape[0] == 0:
            return mask_feats.sum() * 0

        scale = self.prior_generator.strides[0][0] // self.mask_loss_stride
        # upsample pred masks
        batch_pos_mask_logits = F.interpolate(
            batch_pos_mask_logits.unsqueeze(0),
            scale_factor=scale,
            mode='bilinear',
            align_corners=False).squeeze(0)
        # downsample gt masks
        pos_gt_masks = pos_gt_masks[:, self.mask_loss_stride //
                                    2::self.mask_loss_stride,
                                    self.mask_loss_stride //
                                    2::self.mask_loss_stride]

        loss_mask = self.loss_mask(
            batch_pos_mask_logits,
            pos_gt_masks,
            weight=None,
            avg_factor=num_pos)

        return loss_mask


@MODELS.register_module(force=True)
class RTMDetInsSepBNHeadCustom(RTMDetInsSepBNHead):
    def _mask_predict_by_feat_single(self, mask_feat: Tensor, kernels: Tensor,
                                     priors: Tensor) -> Tensor:

        ori_maskfeat = mask_feat

        num_inst = priors.shape[0]
        h, w = mask_feat.size()[-2:]
        if num_inst < 1:
            return torch.empty(
                size=(num_inst, h, w),
                dtype=mask_feat.dtype,
                device=mask_feat.device)
        if len(mask_feat.shape) < 4:
            mask_feat.unsqueeze(0)

        coord = self.prior_generator.single_level_grid_priors(
            (h, w), level_idx=0, device=mask_feat.device).reshape(1, -1, 2)
        num_inst = priors.shape[0]
        points = priors[:, :2].reshape(-1, 1, 2)
        strides = priors[:, 2:].reshape(-1, 1, 2)
        relative_coord = (points - coord).permute(0, 2, 1) / (
            strides[..., 0].reshape(-1, 1, 1) * 8)
        relative_coord = relative_coord.reshape(num_inst, 2, h, w)

        mask_feat = torch.cat(
            [relative_coord,
             mask_feat.repeat(num_inst, 1, 1, 1)], dim=1)
        weights, biases = self.parse_dynamic_params(kernels)

        fp16_used = weights[0].dtype == torch.float16

        n_layers = len(weights)
        x = mask_feat.reshape(1, -1, h, w)
        for i, (weight, bias) in enumerate(zip(weights, biases)):
            with torch.amp.autocast("cuda", enabled=False):
                if fp16_used:
                    weight = weight.to(torch.float32)
                    bias = bias.to(torch.float32)
                x = F.conv2d(
                    x, weight, bias=bias, stride=1, padding=0, groups=num_inst)
                if i < n_layers - 1:
                    x = F.relu(x)

        if fp16_used:
            x = torch.clip(x, -8192, 8192)
        if sthgoeswrong(x):
            torch.save({'mask_feat': ori_maskfeat, 'kernels': kernels, 'priors': priors}, 'maskhead_nan_input.pt')
            raise Exception('Mask Head NaN')

        x = x.reshape(num_inst, h, w)
        return x

    @AvoidCUDAOOM.retry_if_cuda_oom
    def loss_mask_by_feat(self, mask_feats: Tensor, flatten_kernels: Tensor,
                          sampling_results_list: list,
                          batch_gt_instances: InstanceList) -> Tensor:
        batch_pos_mask_logits = []
        pos_gt_masks = []
        ignore_masks = []
        for idx, (mask_feat, kernels, sampling_results,
                  gt_instances) in enumerate(
                      zip(mask_feats, flatten_kernels, sampling_results_list,
                          batch_gt_instances)):
            pos_priors = sampling_results.pos_priors
            pos_inds = sampling_results.pos_inds
            pos_kernels = kernels[pos_inds]  # n_pos, num_gen_params
            pos_mask_logits = self._mask_predict_by_feat_single(
                mask_feat, pos_kernels, pos_priors)
            if gt_instances.masks.numel() == 0:
                gt_masks = torch.empty_like(gt_instances.masks)
                # if gt_masks.shape[0] > 0:
                    # ignore = torch.zeros(gt_masks.shape[0], dtype=torch.bool).to(device=gt_masks.device)
                    # ignore_masks.append(ignore)
            else:
                msk = torch.logical_not(gt_instances.ignore_mask[sampling_results.pos_assigned_gt_inds])
                gt_masks = gt_instances.masks[
                    sampling_results.pos_assigned_gt_inds, :][msk]
                pos_mask_logits = pos_mask_logits[msk]
                # ignore_masks.append(gt_instances.ignore_mask[sampling_results.pos_assigned_gt_inds])
            batch_pos_mask_logits.append(pos_mask_logits)
            pos_gt_masks.append(gt_masks)

        pos_gt_masks = torch.cat(pos_gt_masks, 0)
        batch_pos_mask_logits = torch.cat(batch_pos_mask_logits, 0)
        # ignore_masks = torch.logical_not(torch.cat(ignore_masks, 0))

        # pos_gt_masks = pos_gt_masks[ignore_masks]
        # batch_pos_mask_logits = batch_pos_mask_logits[ignore_masks]


        # avg_factor
        num_pos = batch_pos_mask_logits.shape[0]
        num_pos = reduce_mean(mask_feats.new_tensor([num_pos
                                                     ])).clamp_(min=1).item()

        if batch_pos_mask_logits.shape[0] == 0:
            return mask_feats.sum() * 0

        scale = self.prior_generator.strides[0][0] // self.mask_loss_stride
        # upsample pred masks
        batch_pos_mask_logits = F.interpolate(
            batch_pos_mask_logits.unsqueeze(0),
            scale_factor=scale,
            mode='bilinear',
            align_corners=False).squeeze(0)
        # downsample gt masks
        pos_gt_masks = pos_gt_masks[:, self.mask_loss_stride //
                                    2::self.mask_loss_stride,
                                    self.mask_loss_stride //
                                    2::self.mask_loss_stride]

        loss_mask = self.loss_mask(
            batch_pos_mask_logits,
            pos_gt_masks,
            weight=None,
            avg_factor=num_pos)

        return loss_mask
