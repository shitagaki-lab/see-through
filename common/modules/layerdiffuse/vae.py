import torch.nn as nn
import torch
import cv2
import numpy as np
import safetensors.torch as sf

from tqdm import tqdm
from typing import Optional, Tuple
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2D, get_down_block, get_up_block
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution


from utils.cv import pad_rgb, checkerboard


class LatentTransparencyOffsetEncoder(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blocks = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.SiLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.SiLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
            nn.SiLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            torch.nn.Conv2d(256, 4, kernel_size=3, padding=1, stride=1),
        )

    def __call__(self, x):
        return self.blocks(x)


# 1024 * 1024 * 3 -> 16 * 16 * 512 -> 1024 * 1024 * 3
class UNet1024(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types: Tuple[str] = ("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        block_out_channels: Tuple[int] = (32, 32, 64, 128, 256, 512, 512),
        layers_per_block: int = 2,
        mid_block_scale_factor: float = 1,
        downsample_padding: int = 1,
        downsample_type: str = "conv",
        upsample_type: str = "conv",
        dropout: float = 0.0,
        act_fn: str = "silu",
        attention_head_dim: Optional[int] = 8,
        norm_num_groups: int = 4,
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        # input
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))
        self.latent_conv_in = nn.Conv2d(4, block_out_channels[2], kernel_size=1)

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=None,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=attention_head_dim if attention_head_dim is not None else output_channel,
                downsample_padding=downsample_padding,
                resnet_time_scale_shift="default",
                downsample_type=downsample_type,
                dropout=dropout,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            temb_channels=None,
            dropout=dropout,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift="default",
            attention_head_dim=attention_head_dim if attention_head_dim is not None else block_out_channels[-1],
            resnet_groups=norm_num_groups,
            attn_groups=None,
            add_attention=True,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=None,
                add_upsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=attention_head_dim if attention_head_dim is not None else output_channel,
                resnet_time_scale_shift="default",
                upsample_type=upsample_type,
                dropout=dropout,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def forward(self, x, latent):
        sample_latent = self.latent_conv_in(latent)
        sample = self.conv_in(x)
        emb = None

        down_block_res_samples = (sample,)
        for i, downsample_block in enumerate(self.down_blocks):
            if i == 3:
                sample = sample + sample_latent

            sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

        sample = self.mid_block(sample, emb)

        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            sample = upsample_block(sample, res_samples, emb)

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        return sample


def dist_sample_deterministic(dist: DiagonalGaussianDistribution, perturbation: torch.Tensor):
    # Modified from diffusers.models.autoencoders.vae.DiagonalGaussianDistribution.sample()
    x = dist.mean + dist.std * perturbation.to(dist.std)
    return x


class TransparentVAEDecoder(torch.nn.Module):
    def __init__(self, ckpt=None, in_channels=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = UNet1024(in_channels=in_channels, out_channels=4)
        if ckpt is not None:
            sd = sf.load_file(ckpt)
            self.model.load_state_dict(sd, strict=True)

    def estimate_single_pass(self, pixel, latent, rgb_cond=None):
        if rgb_cond is not None:
            pixel = torch.concat([pixel, rgb_cond], dim=-3)
        y = self.model(pixel, latent)
        return y

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def estimate_augmented(self, pixel, latent, rgb_cond=None):
        args = [
            [False, 0], [False, 1], [False, 2], [False, 3], [True, 0], [True, 1], [True, 2], [True, 3],
        ]

        result = []

        for flip, rok in args:
            feed_pixel = pixel.clone()
            feed_latent = latent.clone()

            if flip:
                feed_pixel = torch.flip(feed_pixel, dims=(3,))
                feed_latent = torch.flip(feed_latent, dims=(3,))

            feed_pixel = torch.rot90(feed_pixel, k=rok, dims=(2, 3))
            feed_latent = torch.rot90(feed_latent, k=rok, dims=(2, 3))

            if rgb_cond is not None:
                if flip:
                    rgb_cond = torch.flip(rgb_cond, dims=(3,))
                rgb_cond = torch.rot90(rgb_cond, k=rok, dims=(2, 3))

            eps = self.estimate_single_pass(feed_pixel, feed_latent, rgb_cond=rgb_cond).clip(0, 1)
            eps = torch.rot90(eps, k=-rok, dims=(2, 3))

            if flip:
                eps = torch.flip(eps, dims=(3,))

            result += [eps]
            break

        result = torch.stack(result, dim=0)
        median = torch.median(result.cpu(), dim=0).values.to(result.device)
        return median

    @torch.no_grad()
    def forward(self, sd_vae, latent, return_type='numpy', rgb_cond=None, mask=None, return_rgb=False):
        pixel = sd_vae.decode(latent).sample
        pixel = (pixel * 0.5 + 0.5).to(self.dtype)
        latent = latent.to(self.dtype)
        result_list = []
        vis_list = []

        for i in range(int(latent.shape[0])):
            y = self.estimate_augmented(pixel[i:i + 1], latent[i:i + 1], rgb_cond=rgb_cond)

            if return_type == 'tensor':
                result_list.append(y)
                continue

            y = y.clip(0, 1).movedim(1, -1)
            alpha = y[..., :1]
            if mask is not None:
                alpha = alpha * mask
            fg = y[..., 1:]

            B, H, W, C = fg.shape
            cb = checkerboard(shape=(H // 64, W // 64))
            cb = cv2.resize(cb, (W, H), interpolation=cv2.INTER_NEAREST)
            cb = (0.5 + (cb - 0.5) * 0.1)[None, ..., None]
            cb = torch.from_numpy(cb).to(fg)

            vis = (fg * alpha + cb * (1 - alpha))[0]
            vis = (vis * 255.0).detach().cpu().float().numpy().clip(0, 255).astype(np.uint8)
            vis_list.append(vis)

            png = torch.cat([fg, alpha], dim=3)[0]
            if return_type == 'numpy':
                png = (png * 255.0).detach().cpu().float().numpy().clip(0, 255).astype(np.uint8)
            result_list.append(png)

        if return_rgb:
            return pixel, result_list, vis_list
        return result_list, vis_list


class TransparentVAEEncoder(torch.nn.Module):
    def __init__(self, ckpt=None, alpha=300.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = LatentTransparencyOffsetEncoder()
        if ckpt is not None:
            sd = sf.load_file(ckpt)
            self.model.load_state_dict(sd, strict=True)
        # similar to LoRA's alpha to avoid initial zero-initialized outputs being too small
        self.alpha = alpha

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def encode(self, tensor, sd_vae, use_offset=True):
        '''
        tensor: shape(b c h w), argb range [0, 1]
        '''
        alpha = tensor[:, :1]
        vae_feed = (tensor[:, 1:] * 2 - 1) * alpha
        latent_dist = sd_vae.encode(vae_feed).latent_dist

        if use_offset:
            offset_feed = tensor
            offset = self.model(offset_feed) * self.alpha
            latent = dist_sample_deterministic(dist=latent_dist, perturbation=offset)
        else:
            latent = latent_dist.sample()
        return latent

    @torch.no_grad()
    def forward(self, sd_vae, list_of_np_rgba_hwc_uint8, use_offset=True):
        list_of_np_rgb_padded = [pad_rgb(x) for x in list_of_np_rgba_hwc_uint8]
        rgb_padded_bchw_01 = torch.from_numpy(np.stack(list_of_np_rgb_padded, axis=0)).float().movedim(-1, 1)
        rgba_bchw_01 = torch.from_numpy(np.stack(list_of_np_rgba_hwc_uint8, axis=0)).float().movedim(-1, 1) / 255.0
        rgb_bchw_01 = rgba_bchw_01[:, :3, :, :]
        a_bchw_01 = rgba_bchw_01[:, 3:, :, :]
        vae_feed = (rgb_bchw_01 * 2.0 - 1.0) * a_bchw_01
        vae_feed = vae_feed.to(device=sd_vae.device, dtype=sd_vae.dtype)
        latent_dist = sd_vae.encode(vae_feed).latent_dist
        offset_feed = torch.cat([a_bchw_01, rgb_padded_bchw_01], dim=1).to(device=sd_vae.device, dtype=self.dtype)
        offset = self.model(offset_feed) * self.alpha
        if use_offset:
            latent = dist_sample_deterministic(dist=latent_dist, perturbation=offset)
        else:
            latent = latent_dist.sample()
        return latent


class TransparentVAE(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
        self
    ):
        super().__init__()
        self.decoder = TransparentVAEDecoder()
        self.encoder = TransparentVAEEncoder()

    



@torch.inference_mode()
def vae_encode(vae, trans_vae_encoder: TransparentVAEEncoder, argb_tensor: torch.Tensor, use_offset=True, scale=True) -> torch.Tensor:
    latent = trans_vae_encoder.encode(argb_tensor.to(dtype=vae.dtype, device=vae.device), vae, use_offset=use_offset)
    if scale:
        latent = latent * vae.config.scaling_factor
    return latent
