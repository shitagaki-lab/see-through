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
import os.path as osp
import logging
import numpy as np
import torch
from PIL import Image
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    LCMScheduler,
    UNet2DConditionModel,
)
from diffusers.utils import BaseOutput
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import pil_to_tensor, resize
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from typing import Dict, Optional, Union

from modules.layerdiffuse.layerdiff3d import UNetFrameConditionModel
from .util.batchsize import find_batch_size
from .util.ensemble import ensemble_depth
from .util.image_util import (
    chw2hwc,
    colorize_depth_maps,
    get_tv_resample_method,
    resize_max_res,
)
from utils.torchcv import pad_rgb_torch
from utils.torch_utils import img2tensor


def encode_rgb(vae, rgb_in: torch.Tensor, latent_scale_factor = 0.18215) -> torch.Tensor:
    """
    Encode RGB image into latent.

    Args:
        rgb_in (`torch.Tensor`):
            Input RGB image to be encoded.

    Returns:
        `torch.Tensor`: Image latent.
    """
    # encode
    h = vae.encoder(rgb_in)
    moments = vae.quant_conv(h)
    mean, logvar = torch.chunk(moments, 2, dim=1)
    # scale latent
    rgb_latent = mean * latent_scale_factor
    return rgb_latent


def encode_argb_list(vae, batched_img_list, dtype=torch.float32, pad_argb=False):
    batched_latent_list = []
    for img_list in batched_img_list:
        if pad_argb:
            img_list = pad_rgb_torch(img_list.float(), return_format='argb')
        img_list = img_list.to(device=vae.device, dtype=vae.dtype)
        rgb_in = img_list[:, 1:] * 2 - 1
        batched_latent_list.append(
            encode_rgb(vae, rgb_in).to(dtype=dtype)
        )
    return torch.stack(batched_latent_list)


def encode_depth(vae, depth_in):
    # stack depth into 3-channel
    stacked = stack_depth_images(depth_in)
    # encode using VAE encoder
    depth_latent = encode_rgb(vae, stacked)
    return depth_latent


def encode_depth_list(vae, batched_depth_list, dtype=torch.float32):
    batched_latent_list = []
    for depth_list in batched_depth_list:
        depth_list = depth_list.to(device=vae.device, dtype=vae.dtype)
        batched_latent_list.append(
            encode_depth(vae, depth_list).to(dtype=dtype)
        )
    return torch.stack(batched_latent_list)


def stack_depth_images(depth_in):
    if 4 == len(depth_in.shape):
        stacked = depth_in.repeat(1, 3, 1, 1)
    elif 3 == len(depth_in.shape):
        stacked = depth_in.unsqueeze(1).repeat(1, 3, 1, 1)
    return stacked


def encode_empty_text():

    """
    Encode text embedding for empty prompt
    """
    from safetensors.torch import load_file, save_file

    cached_empty_text_tensor = "workspace/empty_text_tensor.safetensors"
    if osp.exists(cached_empty_text_tensor):
        empty_text_embed = load_file(cached_empty_text_tensor)['tensors']
        return empty_text_embed
    
    text_encoder = CLIPTextModel.from_pretrained("prs-eth/marigold-depth-v1-1", subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained("prs-eth/marigold-depth-v1-1", subfolder="tokenizer")

    prompt = ""
    text_inputs = tokenizer(
        prompt,
        padding="do_not_pad",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(text_encoder.device)
    empty_text_embed = text_encoder(text_input_ids)[0]
    
    save_file({'tensors': empty_text_embed}, cached_empty_text_tensor)

    return empty_text_embed



class MarigoldDepthOutput(BaseOutput):
    """
    Output class for Marigold Monocular Depth Estimation pipeline.

    Args:
        depth_np (`np.ndarray`):
            Predicted depth map, with depth values in the range of [0, 1].
        depth_colored (`PIL.Image.Image`):
            Colorized depth map, with the shape of [H, W, 3] and values in [0, 255].
        uncertainty (`None` or `np.ndarray`):
            Uncalibrated uncertainty(MAD, median absolute deviation) coming from ensembling.
    """
    depth_tensor: torch.Tensor
    depth_np: np.ndarray
    depth_colored: Union[None, Image.Image]
    uncertainty: Union[None, np.ndarray]


class MarigoldDepthPipeline(DiffusionPipeline):
    """
    Pipeline for Marigold Monocular Depth Estimation: https://marigoldcomputervision.github.io.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        unet (`UNet2DConditionModel`):
            Conditional U-Net to denoise the prediction latent, conditioned on image latent.
        vae (`AutoencoderKL`):
            Variational Auto-Encoder (VAE) Model to encode and decode images and predictions
            to and from latent representations.
        scheduler (`DDIMScheduler`):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        text_encoder (`CLIPTextModel`):
            Text-encoder, for empty text embedding.
        tokenizer (`CLIPTokenizer`):
            CLIP tokenizer.
        scale_invariant (`bool`, *optional*):
            A model property specifying whether the predicted depth maps are scale-invariant. This value must be set in
            the model config. When used together with the `shift_invariant=True` flag, the model is also called
            "affine-invariant". NB: overriding this value is not supported.
        shift_invariant (`bool`, *optional*):
            A model property specifying whether the predicted depth maps are shift-invariant. This value must be set in
            the model config. When used together with the `scale_invariant=True` flag, the model is also called
            "affine-invariant". NB: overriding this value is not supported.
        default_denoising_steps (`int`, *optional*):
            The minimum number of denoising diffusion steps that are required to produce a prediction of reasonable
            quality with the given model. This value must be set in the model config. When the pipeline is called
            without explicitly setting `num_inference_steps`, the default value is used. This is required to ensure
            reasonable results with various model flavors compatible with the pipeline, such as those relying on very
            short denoising schedules (`LCMScheduler`) and those with full diffusion schedules (`DDIMScheduler`).
        default_processing_resolution (`int`, *optional*):
            The recommended value of the `processing_resolution` parameter of the pipeline. This value must be set in
            the model config. When the pipeline is called without explicitly setting `processing_resolution`, the
            default value is used. This is required to ensure reasonable results with various model flavors trained
            with varying optimal processing resolution values.
    """

    latent_scale_factor = 0.18215

    def __init__(
        self,
        unet,
        vae: AutoencoderKL,
        scheduler: Union[DDIMScheduler, LCMScheduler],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        scale_invariant: Optional[bool] = True,
        shift_invariant: Optional[bool] = True,
        default_denoising_steps: Optional[int] = 4,
        default_processing_resolution: Optional[int] = 768,
    ):
        super().__init__()
        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        self.register_to_config(
            scale_invariant=scale_invariant,
            shift_invariant=shift_invariant,
            default_denoising_steps=default_denoising_steps,
            default_processing_resolution=default_processing_resolution,
        )

        self.scale_invariant = scale_invariant
        self.shift_invariant = shift_invariant
        self.default_denoising_steps = default_denoising_steps
        self.default_processing_resolution = default_processing_resolution

        self.empty_text_embed = None

    @torch.no_grad()
    def __call__(
        self,
        input_image: Union[Image.Image, torch.Tensor] = None,
        cond_latent=None,
        denoising_steps: Optional[int] = None,
        ensemble_size: int = 1,
        processing_res: Optional[int] = None,
        match_input_res: bool = True,
        resample_method: str = "bilinear",
        batch_size: int = 0,
        generator: Union[torch.Generator, None] = None,
        color_map: str = "Spectral",
        ensemble_kwargs: Dict = None,
        img_list=None,
        **kwargs
    ) -> MarigoldDepthOutput:
        """
        Function invoked when calling the pipeline.

        Args:
            input_image (`Image`):
                Input RGB (or gray-scale) image.
            denoising_steps (`int`, *optional*, defaults to `None`):
                Number of denoising diffusion steps during inference. The default value `None` results in automatic
                selection.
            ensemble_size (`int`, *optional*, defaults to `1`):
                Number of predictions to be ensembled.
            processing_res (`int`, *optional*, defaults to `None`):
                Effective processing resolution. When set to `0`, processes at the original image resolution. This
                produces crisper predictions, but may also lead to the overall loss of global context. The default
                value `None` resolves to the optimal value from the model config.
            match_input_res (`bool`, *optional*, defaults to `True`):
                Resize the prediction to match the input resolution.
                Only valid if `processing_res` > 0.
            resample_method: (`str`, *optional*, defaults to `bilinear`):
                Resampling method used to resize images and predictions. This can be one of `bilinear`, `bicubic` or
                `nearest`, defaults to: `bilinear`.
            batch_size (`int`, *optional*, defaults to `0`):
                Inference batch size, no bigger than `num_ensemble`.
                If set to 0, the script will automatically decide the proper batch size.
            generator (`torch.Generator`, *optional*, defaults to `None`)
                Random generator for initial noise generation.
            color_map (`str`, *optional*, defaults to `"Spectral"`, pass `None` to skip colorized depth map generation):
                Colormap used to colorize the depth map.
            scale_invariant (`str`, *optional*, defaults to `True`):
                Flag of scale-invariant prediction, if True, scale will be adjusted from the raw prediction.
            shift_invariant (`str`, *optional*, defaults to `True`):
                Flag of shift-invariant prediction, if True, shift will be adjusted from the raw prediction, if False,
                near plane will be fixed at 0m.
            ensemble_kwargs (`dict`, *optional*, defaults to `None`):
                Arguments for detailed ensembling settings.
        Returns:
            `MarigoldDepthOutput`: Output class for Marigold monocular depth prediction pipeline, including:
            - **depth_np** (`np.ndarray`) Predicted depth map with depth values in the range of [0, 1]
            - **depth_colored** (`PIL.Image.Image`) Colorized depth map, with the shape of [H, W, 3] and values in [0, 255], None if `color_map` is `None`
            - **uncertainty** (`None` or `np.ndarray`) Uncalibrated uncertainty(MAD, median absolute deviation)
                    coming from ensembling. None if `ensemble_size = 1`
        """
        # Model-specific optimal default values leading to fast and reasonable results.


        def _np_transform(img):
            img = np.concatenate([img[..., 3:], img[..., :3]], axis=2).astype(np.float32) / 255.
            img = img2tensor(img=img, dim_order='chw')
            return img
        
        if img_list is not None:
            img_list_tensor = torch.stack([_np_transform(img) for img in img_list])
            cond_full_page = img_list_tensor[-1][None]
            ncls = img_list_tensor.shape[0]
            with torch.no_grad():
                vae = self.vae
                rgb_latent = [encode_argb_list(vae, img[None, None].to(device=vae.device, dtype=vae.dtype), pad_argb=True, dtype=vae.dtype) for img in img_list_tensor]
                rgb_latent = torch.cat(rgb_latent, dim=1)
                rgb_cond_latent = encode_argb_list(vae, cond_full_page[None], pad_argb=True, dtype=vae.dtype)
                rgb_latent = torch.cat(
                    [rgb_cond_latent.expand(-1, ncls, -1, -1, -1), rgb_latent], dim=2
                )
                cond_latent = rgb_latent[0]

        if denoising_steps is None or denoising_steps == -1:
            denoising_steps = self.default_denoising_steps
        if processing_res is None:
            processing_res = self.default_processing_resolution

        assert processing_res >= 0
        assert ensemble_size >= 1

        # Check if denoising step is reasonable
        self._check_inference_step(denoising_steps)

        resample_method: InterpolationMode = get_tv_resample_method(resample_method)

        if cond_latent is not None:
            rgb_latent = cond_latent
            input_size = [rgb_latent.shape[2] * 8, rgb_latent.shape[3] * 8]
        else:
            # ----------------- Image Preprocess -----------------
            # Convert to torch tensor
            if isinstance(input_image, Image.Image):
                input_image = input_image.convert("RGB")
                # convert to torch tensor [H, W, rgb] -> [rgb, H, W]
                rgb = pil_to_tensor(input_image)
                rgb = rgb.unsqueeze(0)  # [1, rgb, H, W]
            elif isinstance(input_image, torch.Tensor):
                rgb = input_image
            else:
                raise TypeError(f"Unknown input type: {type(input_image) = }")
            input_size = rgb.shape

            # Resize image
            if processing_res > 0:
                rgb = resize_max_res(
                    rgb,
                    max_edge_resolution=processing_res,
                    resample_method=resample_method,
                )

            # Normalize rgb values
            rgb_norm: torch.Tensor = rgb / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
            rgb_latent = self.encode_rgb(rgb_norm.to(dtype=self.vae.dtype, device=self.vae.device))

        # ----------------- Predicting depth -----------------
        # Batch repeated input image
        is_3d = isinstance(self.unet, UNetFrameConditionModel)
        if not is_3d:
            duplicated_rgb = rgb_latent.expand(ensemble_size, -1, -1, -1)
            single_rgb_dataset = TensorDataset(duplicated_rgb)
            if batch_size > 0:
                _bs = batch_size
            else:
                _bs = find_batch_size(
                    ensemble_size=ensemble_size,
                    input_res=max(rgb_norm.shape[1:]),
                    dtype=self.dtype,
                )

            single_rgb_loader = DataLoader(
                single_rgb_dataset, batch_size=_bs, shuffle=False
            )
        else:
            single_rgb_loader = [rgb_latent]

        # Predict depth maps (batched)
        target_pred_ls = []

        iterable = single_rgb_loader
        for batch in iterable:
            if not is_3d:
                (batched_img,) = batch
            else:
                batched_img = batch
            target_pred_raw = self.single_infer(
                cond_latent=batched_img,
                num_inference_steps=denoising_steps,
                generator=generator,
            )
            target_pred_ls.append(target_pred_raw.detach())
        target_preds = torch.concat(target_pred_ls, dim=0)
        torch.cuda.empty_cache()  # clear vram cache for ensembling

        # ----------------- Test-time ensembling -----------------
        if ensemble_size > 1:
            final_pred, pred_uncert = ensemble_depth(
                target_preds,
                scale_invariant=self.scale_invariant,
                shift_invariant=self.shift_invariant,
                **(ensemble_kwargs or {}),
            )
        else:
            final_pred = target_preds
            pred_uncert = None

        # Resize back to original resolution
        if match_input_res:
            final_pred = resize(
                final_pred,
                input_size[-2:],
                interpolation=resample_method,
                antialias=True,
            )

        # Convert to numpy
        final_pred = final_pred.squeeze()
        final_pred = final_pred.clip(0, 1)
        final_pred_tensor = final_pred
        final_pred = final_pred.to(device='cpu', dtype=torch.float32).numpy()
        if pred_uncert is not None:
            pred_uncert = pred_uncert.squeeze().to(device='cpu', dtype=torch.float32).numpy()

        # Clip output range
    
        # Colorize
        if color_map is not None:
            depth_colored = colorize_depth_maps(
                final_pred, 0, 1, cmap=color_map
            ).squeeze()  # [3, H, W], value in (0, 1)
            depth_colored = (depth_colored * 255).astype(np.uint8)
            depth_colored_hwc = chw2hwc(depth_colored)
            depth_colored_img = Image.fromarray(depth_colored_hwc)
        else:
            depth_colored_img = None

        return MarigoldDepthOutput(
            depth_tensor=final_pred_tensor,
            depth_np=final_pred,
            depth_colored=depth_colored_img,
            uncertainty=pred_uncert,
        )

    def _check_inference_step(self, n_step: int) -> None:
        """
        Check if denoising step is reasonable
        Args:
            n_step (`int`): denoising steps
        """
        assert n_step >= 1

        if isinstance(self.scheduler, DDIMScheduler):
            if "trailing" != self.scheduler.config.timestep_spacing:
                logging.warning(
                    f"The loaded `DDIMScheduler` is configured with `timestep_spacing="
                    f'"{self.scheduler.config.timestep_spacing}"`; the recommended setting is `"trailing"`. '
                    f"This change is backward-compatible and yields better results. "
                    f"Consider using `prs-eth/marigold-depth-v1-1` for the best experience."
                )
            else:
                if n_step > 10:
                    logging.warning(
                        f"Setting too many denoising steps ({n_step}) may degrade the prediction; consider relying on "
                        f"the default values."
                    )
            if not self.scheduler.config.rescale_betas_zero_snr:
                logging.warning(
                    f"The loaded `DDIMScheduler` is configured with `rescale_betas_zero_snr="
                    f"{self.scheduler.config.rescale_betas_zero_snr}`; the recommended setting is True. "
                    f"Consider using `prs-eth/marigold-depth-v1-1` for the best experience."
                )
        elif isinstance(self.scheduler, LCMScheduler):
            logging.warning(
                "DeprecationWarning: LCMScheduler will not be supported in the future. "
                "Consider using `prs-eth/marigold-depth-v1-1` for the best experience."
            )
            if n_step > 10:
                logging.warning(
                    f"Setting too many denoising steps ({n_step}) may degrade the prediction; consider relying on "
                    f"the default values."
                )
        else:
            raise RuntimeError(f"Unsupported scheduler type: {type(self.scheduler)}")

    def encode_empty_text(self):
        """
        Encode text embedding for empty prompt
        """
        prompt = ""
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype)

    @torch.no_grad()
    def single_infer(
        self,
        cond_latent: torch.Tensor,
        num_inference_steps: int,
        generator: Union[torch.Generator, None],
    ) -> torch.Tensor:
        """
        Perform a single prediction without ensembling.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image.
            num_inference_steps (`int`):
                Number of diffusion denoisign steps (DDIM) during inference.
            generator (`torch.Generator`)
                Random generator for initial noise generation.
        Returns:
            `torch.Tensor`: Predicted targets.
        """

        is_3d = isinstance(self.unet, UNetFrameConditionModel)
        device = self.device
        # rgb_in = rgb_in.to(device)

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps  # [T]

        # Encode image
        # rgb_latent = self.encode_rgb(rgb_in)  # [B, 4, h, w]
        b, c, h, w = cond_latent.shape

        # Noisy latent for outputs
        target_latent = torch.randn(
            (b, 4, h, w),
            device=device,
            dtype=self.unet.dtype,
            generator=generator,
        )  # [B, 4, h, w]

        # Batched empty text embedding
        if self.empty_text_embed is None:
            self.encode_empty_text()
        batch_empty_text_embed = self.empty_text_embed.repeat(
            (b, 1, 1)
        ).to(device)  # [B, 2, 1024]

        iterable = enumerate(timesteps)

        batch_empty_text_embed = batch_empty_text_embed.to(device=target_latent.device, dtype=target_latent.dtype)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in iterable:
                unet_input = torch.cat(
                    [cond_latent, target_latent], dim=1
                )  # this order is important

                if is_3d:
                    unet_input = unet_input[None]
            
                # predict the noise residual
                noise_pred = self.unet(
                    unet_input, t, encoder_hidden_states=batch_empty_text_embed
                ).sample  # [B, 4, h, w]

                if is_3d:
                    noise_pred = noise_pred[0]

                # compute the previous noisy sample x_t -> x_t-1
                target_latent = self.scheduler.step(
                    noise_pred, t, target_latent, generator=generator
                ).prev_sample
                progress_bar.update()

        if is_3d:
            depth = torch.cat([self.decode_depth(t[None]) for t in target_latent])
        else:
            depth = self.decode_depth(target_latent)  # [B,3,H,W]

        # clip prediction
        depth = torch.clip(depth, -1.0, 1.0)
        # shift to [0, 1]
        depth = (depth + 1.0) / 2.0

        return depth

    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image to be encoded.

        Returns:
            `torch.Tensor`: Image latent.
        """
        # encode
        rgb_in = rgb_in.to(device=self.vae.device, dtype=self.vae.dtype)
        h = self.vae.encoder(rgb_in)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale latent
        rgb_latent = mean * self.latent_scale_factor
        return rgb_latent

    def decode_depth(self, depth_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode depth latent into depth map.

        Args:
            depth_latent (`torch.Tensor`):
                Depth latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded depth map.
        """
        # scale latent
        depth_latent = depth_latent / self.latent_scale_factor
        # decode
        depth_latent = depth_latent.to(device=self.vae.device, dtype=self.vae.dtype)
        z = self.vae.post_quant_conv(depth_latent)
        stacked = self.vae.decoder(z)
        # mean of output channels
        depth_mean = stacked.mean(dim=1, keepdim=True)
        return depth_mean
