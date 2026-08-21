# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import torch
import torch.nn as nn
import torch.nn.functional as F
# torch.cuda.amp is deprecated since torch 2.4; the import below targets
# torch.amp (same API, no warning). Kept only for the commented-out block
# below; if that block is removed, this import should go with it.
import torch.amp as amp
import numpy as np


KEY_OUTPUT = 'metric_depth'


def extract_key(prediction, key):
    if isinstance(prediction, dict):
        return prediction[key]
    return prediction


def compute_errors(gt, pred):
    """Compute metrics for 'pred' compared to 'gt'

    Args:
        gt (numpy.ndarray): Ground truth values
        pred (numpy.ndarray): Predicted values

        gt.shape should be equal to pred.shape

    Returns:
        dict: Dictionary containing the following metrics:
            'a1': Delta1 accuracy: Fraction of pixels that are within a scale factor of 1.25
            'a2': Delta2 accuracy: Fraction of pixels that are within a scale factor of 1.25^2
            'a3': Delta3 accuracy: Fraction of pixels that are within a scale factor of 1.25^3
            'abs_rel': Absolute relative error
            'rmse': Root mean squared error
            'log_10': Absolute log10 error
            'sq_rel': Squared relative error
            'rmse_log': Root mean squared error on the log scale
            'silog': Scale invariant log error
    """
    thresh = torch.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)
    sq_rel = torch.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    err = torch.log(pred) - torch.log(gt)
    silog = torch.sqrt(torch.mean(err ** 2) - torch.mean(err) ** 2) * 100

    log_10 = (torch.abs(torch.log10(gt) - torch.log10(pred))).mean()
    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log,
                silog=silog, sq_rel=sq_rel)


def compute_metrics(gt, pred, interpolate=True, mask=None, garg_crop=False, eigen_crop=False, dataset='nyu', min_depth_eval=0.01, max_depth_eval=1000, **kwargs):
    """Compute metrics of predicted depth maps. Applies cropping and masking as necessary or specified via arguments. Refer to compute_errors for more details on metrics.
    """
    if 'config' in kwargs:
        config = kwargs['config']
        garg_crop = config.garg_crop
        eigen_crop = config.eigen_crop
        min_depth_eval = config.min_depth_eval
        max_depth_eval = config.max_depth_eval

    if gt.shape[-2:] != pred.shape[-2:] and interpolate:
        pred = nn.functional.interpolate(
            pred, gt.shape[-2:], mode='bilinear', align_corners=True)

    pred = pred.squeeze().to(dtype=torch.float32)
    pred[pred < min_depth_eval] = min_depth_eval
    pred[pred > max_depth_eval] = max_depth_eval
    pred[torch.isinf(pred)] = max_depth_eval
    pred[torch.isnan(pred)] = min_depth_eval

    gt_depth = gt.squeeze().to(dtype=torch.float32)
    valid_mask = torch.logical_and(
        gt_depth > min_depth_eval, gt_depth < max_depth_eval)
    
    if mask is not None:
        mask = mask.squeeze().to(dtype=torch.bool)
        valid_mask = torch.logical_and(mask, valid_mask)

    eval_mask = None
    if garg_crop or eigen_crop:
        gt_height, gt_width = gt_depth.shape
        eval_mask = torch.zeros(valid_mask.shape)

        if garg_crop:
            eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                      int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

        elif eigen_crop:
            # print("-"*10, " EIGEN CROP ", "-"*10)
            if dataset == 'kitti':
                eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                          int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
            else:
                # assert gt_depth.shape == (480, 640), "Error: Eigen crop is currently only valid for (480, 640) images"
                eval_mask[45:471, 41:601] = 1
        else:
            eval_mask = torch.ones(valid_mask.shape)

    if eval_mask is not None:
        valid_mask = torch.logical_and(valid_mask, eval_mask)

    return compute_errors(gt_depth[valid_mask], pred[valid_mask])



# # Main loss function used for ZoeDepth. Copy/paste from AdaBins repo (https://github.com/shariqfarooq123/AdaBins/blob/0952d91e9e762be310bb4cd055cbfe2448c0ce20/loss.py#L7)
# class SILogLoss(nn.Module):
#     """SILog loss (pixel-wise)"""
#     def __init__(self, beta=0.15):
#         super(SILogLoss, self).__init__()
#         self.name = 'SILog'
#         self.beta = beta

#     def forward(self, input, target, mask=None, interpolate=True, return_interpolated=False):
#         input = extract_key(input, KEY_OUTPUT)
#         if input.shape[-1] != target.shape[-1] and interpolate:
#             input = nn.functional.interpolate(
#                 input, target.shape[-2:], mode='bilinear', align_corners=True)
#             intr_input = input
#         else:
#             intr_input = input

#         if target.ndim == 3:
#             target = target.unsqueeze(1)

#         if mask is not None:
#             if mask.ndim == 3:
#                 mask = mask.unsqueeze(1)

#             input = input[mask]
#             target = target[mask]

#         with amp.autocast("cuda", enabled=False):  # amp causes NaNs in this loss function
#             alpha = 1e-7
#             g = torch.log(input + alpha) - torch.log(target + alpha)

#             # n, c, h, w = g.shape
#             # norm = 1/(h*w)
#             # Dg = norm * torch.sum(g**2) - (0.85/(norm**2)) * (torch.sum(g))**2

#             Dg = torch.var(g) + self.beta * torch.pow(torch.mean(g), 2)

#             loss = 10 * torch.sqrt(Dg)

#         if torch.isnan(loss):
#             print("Nan SILog loss")
#             print("input:", input.shape)
#             print("target:", target.shape)
#             print("G", torch.sum(torch.isnan(g)))
#             print("Input min max", torch.min(input), torch.max(input))
#             print("Target min max", torch.min(target), torch.max(target))
#             print("Dg", torch.isnan(Dg))
#             print("loss", torch.isnan(loss))

#         if not return_interpolated:
#             return loss

#         return loss, intr_input


def grad(x):
    # x.shape : n, c, h, w
    diff_x = x[..., 1:, 1:] - x[..., 1:, :-1]
    diff_y = x[..., 1:, 1:] - x[..., :-1, 1:]
    mag = diff_x**2 + diff_y**2
    # angle_ratio
    angle = torch.atan(diff_y / (diff_x + 1e-10))
    return mag, angle


def grad_mask(mask):
    return mask[..., 1:, 1:] & mask[..., 1:, :-1] & mask[..., :-1, 1:]


class GradL1Loss(nn.Module):
    """Gradient loss"""
    def __init__(self):
        super(GradL1Loss, self).__init__()
        self.name = 'GradL1'

    def forward(self, input, target, mask=None, interpolate=True, return_interpolated=False):
        input = extract_key(input, KEY_OUTPUT)
        if input.shape[-1] != target.shape[-1] and interpolate:
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = input
        else:
            intr_input = input

        grad_gt = grad(target)
        grad_pred = grad(input)
        mask_g = grad_mask(mask)

        loss = nn.functional.l1_loss(grad_pred[0][mask_g], grad_gt[0][mask_g])
        loss = loss + \
            nn.functional.l1_loss(grad_pred[1][mask_g], grad_gt[1][mask_g])
        if not return_interpolated:
            return loss
        return loss, intr_input


def grad_match_loss(input, target, mask=None):

    grad_gt = grad(target)
    grad_pred = grad(input)
    mask_g = grad_mask(mask)

    loss = nn.functional.l1_loss(grad_pred[0][mask_g], grad_gt[0][mask_g])
    loss = loss + \
        nn.functional.l1_loss(grad_pred[1][mask_g], grad_gt[1][mask_g])

    return loss



def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "SSILoss"

    def forward(self, prediction, target, mask, interpolate=True, return_interpolated=False):
        
        if prediction.shape[-1] != target.shape[-1] and interpolate:
            prediction = nn.functional.interpolate(prediction, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = prediction
        else:
            intr_input = prediction


        prediction, target, mask = prediction.squeeze(), target.squeeze(), mask.squeeze()
        assert prediction.shape == target.shape, f"Shape mismatch: Expected same shape but got {prediction.shape} and {target.shape}."

        scale, shift = compute_scale_and_shift(prediction, target, mask)

        scaled_prediction = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        loss = nn.functional.l1_loss(scaled_prediction[mask], target[mask])
        if not return_interpolated:
            return loss
        return loss, intr_input


from einops import reduce
# class AffineInvariantLoss

def affine_invariant_loss(prediction, gt, masks):
    # depth anything v1 section 3.1
    sums_per_batch = reduce(masks, 'b c h w -> b 1 1 1', 'sum')
    masks_bool = masks > 0

    def _align(x):
        with torch.no_grad():
            medians = []
            for m, xx in zip(masks_bool, x):
                medians.append(torch.median(xx[m]))
            medians = torch.tensor(medians, dtype=x.dtype, device=x.device)[..., None, None, None]
            s = reduce(torch.abs(x - medians) * masks, 'b c h w -> b 1 1 1', 'sum') / sums_per_batch
        return (x - medians) / (s + 1e-6)
    
    prediction = _align(prediction)
    gt = _align(gt)

    loss = torch.nn.functional.l1_loss(prediction, gt, reduction='none') * masks
    loss = reduce(loss, 'b c h w -> b 1 1 1', 'sum') / sums_per_batch
    return loss.mean()


class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target, valid_mask):
        valid_mask = valid_mask.detach()
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        # print(torch.log(pred[valid_mask]).mean())
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                          self.lambd * torch.pow(diff_log.mean(), 2))

        return loss

if __name__ == '__main__':
    # Tests for DiscreteNLLLoss
    celoss = DiscreteNLLLoss()
    print(celoss(torch.rand(4, 64, 26, 32)*10, torch.rand(4, 1, 26, 32)*10, ))

    d = torch.Tensor([6.59, 3.8, 10.0])
    print(celoss.dequantize_depth(celoss.quantize_depth(d)))
