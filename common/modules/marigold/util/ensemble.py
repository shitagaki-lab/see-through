# Copyright 2023-2025 Marigold Team, ETH Zürich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# More information about Marigold:
#   https://marigoldmonodepth.github.io
#   https://marigoldcomputervision.github.io
# Efficient inference pipelines are now part of diffusers:
#   https://huggingface.co/docs/diffusers/using-diffusers/marigold_usage
#   https://huggingface.co/docs/diffusers/api/pipelines/marigold
# Examples of trained models and live demos:
#   https://huggingface.co/prs-eth
# Related projects:
#   https://rollingdepth.github.io/
#   https://marigolddepthcompletion.github.io/
# Citation (BibTeX):
#   https://github.com/prs-eth/Marigold#-citation
# If you find Marigold useful, we kindly ask you to cite our papers.
# --------------------------------------------------------------------------

import numpy as np
import torch
from functools import partial
from typing import Optional, Tuple

from .image_util import get_tv_resample_method, resize_max_res


def ensemble_depth(
    depth: torch.Tensor,
    scale_invariant: bool = True,
    shift_invariant: bool = True,
    output_uncertainty: bool = False,
    reduction: str = "median",
    regularizer_strength: float = 0.02,
    max_iter: int = 50,
    tol: float = 1e-6,
    max_res: int = 1024,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Ensembles depth maps represented by the `depth` tensor with expected shape `(B, 1, H, W)`, where B is the
    number of ensemble members for a given prediction of size `(H x W)`. Even though the function is designed for
    depth maps, it can also be used with disparity maps as long as the input tensor values are non-negative. The
    alignment happens when the predictions have one or more degrees of freedom, that is when they are either
    affine-invariant (`scale_invariant=True` and `shift_invariant=True`), or just scale-invariant (only
    `scale_invariant=True`). For absolute predictions (`scale_invariant=False` and `shift_invariant=False`)
    alignment is skipped and only ensembling is performed.

    Args:
        depth (`torch.Tensor`):
            Input ensemble depth maps.
        scale_invariant (`bool`, *optional*, defaults to `True`):
            Whether to treat predictions as scale-invariant.
        shift_invariant (`bool`, *optional*, defaults to `True`):
            Whether to treat predictions as shift-invariant.
        output_uncertainty (`bool`, *optional*, defaults to `False`):
            Whether to output uncertainty map.
        reduction (`str`, *optional*, defaults to `"median"`):
            Reduction method used to ensemble aligned predictions. The accepted values are: `"mean"` and
            `"median"`.
        regularizer_strength (`float`, *optional*, defaults to `0.02`):
            Strength of the regularizer that pulls the aligned predictions to the unit range from 0 to 1.
        max_iter (`int`, *optional*, defaults to `2`):
            Maximum number of the alignment solver steps. Refer to `scipy.optimize.minimize` function, `options`
            argument.
        tol (`float`, *optional*, defaults to `1e-3`):
            Alignment solver tolerance. The solver stops when the tolerance is reached.
        max_res (`int`, *optional*, defaults to `1024`):
            Resolution at which the alignment is performed; `None` matches the `processing_resolution`.
    Returns:
        A tensor of aligned and ensembled depth maps and optionally a tensor of uncertainties of the same shape:
        `(1, 1, H, W)`.
    """
    if depth.dim() != 4 or depth.shape[1] != 1:
        raise ValueError(f"Expecting 4D tensor of shape [B,1,H,W]; got {depth.shape}.")
    if reduction not in ("mean", "median"):
        raise ValueError(f"Unrecognized reduction method: {reduction}.")
    if not scale_invariant and shift_invariant:
        raise ValueError("Pure shift-invariant ensembling is not supported.")

    def init_param(depth: torch.Tensor):
        init_min = depth.reshape(ensemble_size, -1).min(dim=1).values
        init_max = depth.reshape(ensemble_size, -1).max(dim=1).values

        if scale_invariant and shift_invariant:
            init_s = 1.0 / (init_max - init_min).clamp(min=1e-6)
            init_t = -init_s * init_min
            param = torch.cat((init_s, init_t)).to(device='cpu', dtype=torch.float32).numpy()
        elif scale_invariant:
            init_s = 1.0 / init_max.clamp(min=1e-6)
            param = init_s.to(device='cpu', dtype=torch.float32).numpy()
        else:
            raise ValueError("Unrecognized alignment.")

        return param.astype(np.float64)

    def align(depth: torch.Tensor, param: np.ndarray) -> torch.Tensor:
        if scale_invariant and shift_invariant:
            s, t = np.split(param, 2)
            s = torch.from_numpy(s).to(depth).view(ensemble_size, 1, 1, 1)
            t = torch.from_numpy(t).to(depth).view(ensemble_size, 1, 1, 1)
            out = depth * s + t
        elif scale_invariant:
            s = torch.from_numpy(param).to(depth).view(ensemble_size, 1, 1, 1)
            out = depth * s
        else:
            raise ValueError("Unrecognized alignment.")
        return out

    def ensemble(
        depth_aligned: torch.Tensor, return_uncertainty: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        uncertainty = None
        if reduction == "mean":
            prediction = torch.mean(depth_aligned, dim=0, keepdim=True)
            if return_uncertainty:
                uncertainty = torch.std(depth_aligned, dim=0, keepdim=True)
        elif reduction == "median":
            prediction = torch.median(depth_aligned.cpu(), dim=0, keepdim=True).values.to(depth_aligned.device)
            if return_uncertainty:
                uncertainty = torch.median(
                    torch.abs(depth_aligned.cpu() - prediction.cpu()), dim=0, keepdim=True
                ).values.to(depth_aligned.device)
        else:
            raise ValueError(f"Unrecognized reduction method: {reduction}.")
        return prediction, uncertainty

    def cost_fn(param: np.ndarray, depth: torch.Tensor) -> float:
        cost = 0.0
        depth_aligned = align(depth, param)

        for i, j in torch.combinations(torch.arange(ensemble_size)):
            diff = depth_aligned[i] - depth_aligned[j]
            cost += (diff**2).mean().sqrt().item()

        if regularizer_strength > 0:
            prediction, _ = ensemble(depth_aligned, return_uncertainty=False)
            err_near = (0.0 - prediction.min()).abs().item()
            err_far = (1.0 - prediction.max()).abs().item()
            cost += (err_near + err_far) * regularizer_strength

        return cost

    def compute_param(depth: torch.Tensor):
        import scipy

        depth_to_align = depth.to(torch.float32)
        if max_res is not None and max(depth_to_align.shape[2:]) > max_res:
            depth_to_align = resize_max_res(
                depth_to_align, max_res, get_tv_resample_method("nearest-exact")
            )

        param = init_param(depth_to_align)

        res = scipy.optimize.minimize(
            partial(cost_fn, depth=depth_to_align),
            param,
            method="BFGS",
            tol=tol,
            options={"maxiter": max_iter, "disp": False},
        )

        return res.x

    requires_aligning = scale_invariant or shift_invariant
    ensemble_size = depth.shape[0]

    if requires_aligning:
        param = compute_param(depth)
        depth = align(depth, param)

    depth, uncertainty = ensemble(depth, return_uncertainty=output_uncertainty)

    depth_max = depth.max()
    if scale_invariant and shift_invariant:
        depth_min = depth.min()
    elif scale_invariant:
        depth_min = 0
    else:
        raise ValueError("Unrecognized alignment.")
    depth_range = (depth_max - depth_min).clamp(min=1e-6)
    depth = (depth - depth_min) / depth_range
    if output_uncertainty:
        uncertainty /= depth_range

    return depth, uncertainty  # [1,1,H,W], [1,1,H,W]


def ensemble_normals(
    normals: torch.Tensor,
    output_uncertainty: bool = False,
    reduction: str = "closest",
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Ensembles the normals maps represented by the `normals` tensor with expected shape `(B, 3, H, W)`, where B is
    the number of ensemble members for a given prediction of size `(H x W)`.

    Args:
        normals (`torch.Tensor`):
            Input ensemble normals maps.
        output_uncertainty (`bool`, *optional*, defaults to `False`):
            Whether to output uncertainty map.
        reduction (`str`, *optional*, defaults to `"closest"`):
            Reduction method used to ensemble aligned predictions. The accepted values are: `"closest"` and
            `"mean"`.

    Returns:
        A tensor of aligned and ensembled normals maps with shape `(1, 3, H, W)` and optionally a tensor of
        uncertainties of shape `(1, 1, H, W)`.
    """
    if normals.dim() != 4 or normals.shape[1] != 3:
        raise ValueError(
            f"Expecting 4D tensor of shape [B,3,H,W]; got {normals.shape}."
        )
    if reduction not in ("closest", "mean"):
        raise ValueError(f"Unrecognized reduction method: {reduction}.")

    mean_normals = normals.mean(dim=0, keepdim=True)  # [1,3,H,W]
    norm = torch.norm(mean_normals, dim=1, keepdim=True)
    mean_normals /= norm.clamp(min=1e-6)  # [1,3,H,W]

    sim_cos = None
    if output_uncertainty or (reduction != "mean"):
        sim_cos = (mean_normals * normals).sum(dim=1, keepdim=True)  # [E,1,H,W]
        sim_cos = sim_cos.clamp(-1, 1)  # required to avoid NaN in uncertainty with fp16

    uncertainty = None
    if output_uncertainty:
        uncertainty = sim_cos.arccos()  # [E,1,H,W]
        uncertainty = uncertainty.mean(dim=0, keepdim=True) / np.pi  # [1,1,H,W]

    if reduction == "mean":
        return mean_normals, uncertainty  # [1,3,H,W], [1,1,H,W]

    closest_indices = sim_cos.argmax(dim=0, keepdim=True)  # [1,1,H,W]
    closest_indices = closest_indices.repeat(1, 3, 1, 1)  # [1,3,H,W]
    closest_normals = torch.gather(normals, 0, closest_indices)  # [1,3,H,W]

    return closest_normals, uncertainty  # [1,3,H,W], [1,1,H,W]


def ensemble_iid(
    targets: torch.Tensor,
    output_uncertainty: bool = False,
    reduction: str = "median",
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    uncertainty = None
    if reduction == "mean":
        prediction = torch.mean(targets, dim=0, keepdim=True)
        if output_uncertainty:
            uncertainty = torch.std(targets, dim=0, keepdim=True)
    elif reduction == "median":
        prediction = torch.median(targets.cpu(), dim=0, keepdim=True).values.to(targets.device)
        if output_uncertainty:
            uncertainty = torch.median(
                torch.abs(targets.cpu() - prediction.cpu()), dim=0, keepdim=True
            ).values.to(targets.device)
    else:
        raise ValueError(f"Unrecognized reduction method: {reduction}.")
    return prediction, uncertainty
