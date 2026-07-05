from typing import Union, List, Optional
import os.path as osp
import argparse
import sys
import os
sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))

import torch
import numpy as np
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import *
from diffusers.utils.torch_utils import randn_tensor

from modules.layerdiffuse.vae import vae_encode, TransparentVAE
from modules.layerdiffuse.layerdiff3d import UNetFrameConditionModel
from utils.torch_utils import img2tensor
from modules.layerdiffuse.diffusers_kdiffusion_sdxl import KDiffusionStableDiffusionXLPipeline, LayerdiffPipelineOutput
from inference_psd_quantized import build_marigold_pipeline, run_layerdiff, run_marigold
from utils.inference_utils import further_extr

class KDiffusionStableDiffusionXLPipelineBlockSwap(KDiffusionStableDiffusionXLPipeline):
    def __init__(self, 
        vae,
        text_encoder,
        tokenizer,
        text_encoder_2,
        tokenizer_2,
        unet,
        scheduler=None,
        trans_vae=None,
        tag_list=None,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
        requires_aesthetics_score: bool = False,
        force_zeros_for_empty_prompt: bool = True,
        add_watermarker: Optional[bool] = None,
        ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
            trans_vae=trans_vae,
            tag_list=tag_list,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            requires_aesthetics_score=requires_aesthetics_score,
            force_zeros_for_empty_prompt=force_zeros_for_empty_prompt,
            add_watermarker=add_watermarker
        )
        
        # 4. Strict half-precision (bfloat16)
        self.to(dtype=torch.bfloat16)
        
        # 3. Permanent text encoder offloading
        if hasattr(self, 'text_encoder'):
            self.text_encoder.to(device='cpu')
        if hasattr(self, 'text_encoder_2'):
            self.text_encoder_2.to(device='cpu')
            
        self.blockswap_enabled = False
        self.blockswap_device = torch.device("cuda")
        self._blockswap_hooks = []

    @property
    def device(self) -> torch.device:
        if self.blockswap_enabled:
            return self.blockswap_device
        return self.unet.device

    def enable_blockswap(self, device="cuda"):
        """Enable block-level swapping for the UNet to save VRAM."""
        self.blockswap_device = torch.device(device)
        self.blockswap_enabled = True
        
        # Identification of blocks in UNetFrameConditionModel
        unet = self.unet
        blocks = []
        
        # Core components
        if hasattr(unet, 'conv_in'): blocks.append(unet.conv_in)
        if hasattr(unet, 'time_proj'): blocks.append(unet.time_proj)
        if hasattr(unet, 'time_embedding'): blocks.append(unet.time_embedding)
        if hasattr(unet, 'class_embedding'): blocks.append(unet.class_embedding)
        if hasattr(unet, 'add_embedding'): blocks.append(unet.add_embedding)
        if hasattr(unet, 'add_time_proj'): blocks.append(unet.add_time_proj)
        if hasattr(unet, 'encoder_hid_proj'): blocks.append(unet.encoder_hid_proj)
        
        # Hierarchical blocks
        if hasattr(unet, 'down_blocks'):
            for b in unet.down_blocks: blocks.append(b)
        if hasattr(unet, 'mid_block'): 
            blocks.append(unet.mid_block)
        if hasattr(unet, 'up_blocks'):
            for b in unet.up_blocks: blocks.append(b)
            
        # Group embeddings (if present in UNetFrameConditionModel)
        if hasattr(unet, 'group_embeds') and unet.group_embeds is not None:
            for b in unet.group_embeds: blocks.append(b)
        if hasattr(unet, 'group_embeds2') and unet.group_embeds2 is not None:
            for b in unet.group_embeds2: blocks.append(b)

        # Output components
        if hasattr(unet, 'conv_norm_out'): blocks.append(unet.conv_norm_out)
        if hasattr(unet, 'conv_act'): blocks.append(unet.conv_act)
        if hasattr(unet, 'conv_out'): blocks.append(unet.conv_out)
        
        # Ensure all registered blocks are on CPU initially
        for b in blocks:
            if b is not None:
                b.to("cpu")
                # Register hooks for automatic swapping
                h_pre = b.register_forward_pre_hook(self._blockswap_pre_hook)
                h_post = b.register_forward_hook(self._blockswap_post_hook)
                self._blockswap_hooks.extend([h_pre, h_post])
        
        # Optional: Move other large components to CPU if not already there
        # but keep them manageable.
        if hasattr(self, 'vae') and self.vae is not None:
            self.vae.to(dtype=torch.bfloat16, device="cpu")
        if hasattr(self, 'trans_vae') and self.trans_vae is not None:
            self.trans_vae.to(dtype=torch.bfloat16, device="cpu")

        print(f"Blockswap enabled on {device}. UNet blocks will be swapped in/out of VRAM.")

    def _blockswap_pre_hook(self, m, i):
        # Move module weights to VRAM
        m.to(self.blockswap_device)
        
        # Move inputs to VRAM
        def _to_device(obj):
            if isinstance(obj, torch.Tensor):
                return obj.to(self.blockswap_device)
            if isinstance(obj, list):
                return [_to_device(x) for x in obj]
            if isinstance(obj, tuple):
                return tuple(_to_device(x) for x in obj)
            if isinstance(obj, dict):
                return {k: _to_device(v) for k, v in obj.items()}
            return obj
            
        return tuple(_to_device(x) for x in i)

    def _blockswap_post_hook(self, m, i, o):
        # Move module weights back to CPU
        m.to("cpu")
        
        # Move outputs back to CPU or keep them in VRAM? 
        # Usually keeping activations in VRAM is okay if weights are swapped,
        # but for absolute minimum VRAM, we could move them to CPU.
        # Here we keep them in VRAM to avoid excessive transfer overhead 
        # as the next block will likely need them in VRAM.
        
        # Clear cache to free up VRAM from weights
        torch.cuda.empty_cache()
        return o

    def disable_blockswap(self):
        self.blockswap_enabled = False
        for h in self._blockswap_hooks:
            h.remove()
        self._blockswap_hooks = []

    @torch.inference_mode()
    def __call__(
            self,
            initial_latent: torch.FloatTensor = None,
            strength: float = 1.0,
            num_inference_steps: int = 25,
            guidance_scale: float = 5.0,
            batch_size: Optional[int] = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            c_concat=None,
            prompt=None,
            negative_prompt=None,
            show_progress=True,
            fullpage=None,
            group_index=None
    ):
        # Determine execution device
        device = self.blockswap_device if self.blockswap_enabled else self.unet.device
        dtype = torch.bfloat16# self.unet.dtype
        page_alpha = None
        if fullpage is not None:
            # VAE might be on CPU, vae_encode handles device internally or we move it
            vae_device = next(self.vae.parameters()).device
            page_alpha = img2tensor(fullpage[..., -1] / 255., device=vae_device, dtype=self.vae.dtype)[0][..., None]
            fullpage_rgb = fullpage[..., :3]
            c_concat_np = np.concatenate([np.full_like(fullpage_rgb[..., :1], fill_value=255), fullpage_rgb], axis=2)
            c_concat_t = img2tensor(c_concat_np, normalize=True).to(device=vae_device, dtype=self.vae.dtype)
            
            # vae_encode might need to onload VAE if it's offloaded
            self.vae.to(device)
            if self.trans_vae is not None: self.trans_vae.to(device)
            
            c_concat = vae_encode(self.vae, self.trans_vae.encoder, c_concat_t, use_offset=False).to(device=device, dtype=dtype)
            
            if self.blockswap_enabled:
                self.vae.to("cpu")
                if self.trans_vae is not None: self.trans_vae.to("cpu")
                torch.cuda.empty_cache()

        assert c_concat is not None
        c_concat = c_concat.to(device)

        self._guidance_scale = guidance_scale
        is_3d = isinstance(self.unet, UNetFrameConditionModel)
        lh, lw = c_concat.shape[-2:]

        num_frames = 1
        if is_3d:
            if prompt is not None:
                num_frames = len(prompt)
            if prompt_embeds is not None:
                num_frames = len(prompt_embeds)
            
        if initial_latent is None:
            initial_latent = torch.zeros((batch_size, 4, lh, lw), device=device, dtype=dtype)
        else:
            initial_latent = initial_latent.to(device)

        if is_3d and c_concat.ndim == 4:
            c_concat = c_concat[:, None].expand(-1, num_frames, -1, -1, -1)

        if is_3d and initial_latent.ndim == 4:
            initial_latent = initial_latent[:, None].expand(-1, num_frames, -1, -1, -1)

        is_te1_available = next(self.text_encoder.parameters(), None) is not None
        is_te2_available = next(self.text_encoder_2.parameters(), None) is not None
        if prompt is not None:
            # Text encoders might need to be on GPU
            if is_te1_available and is_te2_available:
                te_device = next(self.text_encoder.parameters()).device
                if self.blockswap_enabled:
                    self.text_encoder.to(device)
                    self.text_encoder_2.to(device)
            
            prompt_embeds, pooled_prompt_embeds = self.encode_cropped_prompt_77tokens_cached(prompt)
            
            if self.blockswap_enabled and is_te1_available and is_te2_available:
                self.text_encoder.to("cpu")
                self.text_encoder_2.to("cpu")

        if negative_prompt is not None and self.do_classifier_free_guidance:
            if self.blockswap_enabled and is_te1_available and is_te2_available:
                self.text_encoder.to(device)
                self.text_encoder_2.to(device)
            negative_prompt_embeds, negative_pooled_prompt_embeds = self.encode_cropped_prompt_77tokens(negative_prompt)
            if self.blockswap_enabled and is_te1_available and is_te2_available:
                self.text_encoder.to("cpu")
                self.text_encoder_2.to("cpu")

        # Initial latents
        noise = randn_tensor(initial_latent[:, [0]].shape, generator=generator, device=device, dtype=dtype).expand(-1, num_frames, -1, -1, -1)

        height = lh * self.vae_scale_factor
        width = lw * self.vae_scale_factor

        add_time_ids = list((height, width) + (0, 0) + (height, width))
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype, device=device)
        add_time_ids = add_time_ids.expand((prompt_embeds.shape[0], -1))

        # Batch
        add_time_ids = add_time_ids.repeat(batch_size, 1)
        prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1)
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(batch_size, 1)

        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.repeat(batch_size, 1, 1)
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(batch_size, 1)

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps=None, sigmas=None
        )

        latents = noise * self.scheduler.init_noise_sigma

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}
                if negative_pooled_prompt_embeds is not None:
                    # In CFG, prompt_embeds and added_cond_kwargs should contain both pos and neg
                    concated_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                    concated_pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
                    concated_add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)
                    added_cond_kwargs = {"text_embeds": concated_pooled_prompt_embeds, "time_ids": concated_add_time_ids}
                else:
                    concated_prompt_embeds = prompt_embeds

                # self.unet will use blockswap hooks if enabled
                if self.blockswap_enabled:
                    if hasattr(self.unet, 'group_embeds'):
                        for g in self.unet.group_embeds: g.to(device)
                    if hasattr(self.unet, 'group_embeds2'):
                        for g in self.unet.group_embeds2: g.to(device)
                
                noise_pred = self.unet(
                    torch.cat([latent_model_input, c_concat], dim=-3).to(device),
                    t,
                    encoder_hidden_states=concated_prompt_embeds.to(device),
                    added_cond_kwargs={k: v.to(device) for k, v in added_cond_kwargs.items()},
                    return_dict=False,
                    group_index=group_index
                )[0]
                
                if self.blockswap_enabled:
                    self.unet.to("cpu")
                    # Hooks might have already moved other parts back to CPU, 
                    # but ensure group_embeds stay consistent
                    if hasattr(self.unet, 'group_embeds'):
                        for g in self.unet.group_embeds: g.to("cpu")
                    if hasattr(self.unet, 'group_embeds2'):
                        for g in self.unet.group_embeds2: g.to("cpu")

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                if latents.dtype != latents_dtype:
                    latents = latents.to(latents_dtype)

                if i == len(timesteps) - 1 or (i + 1) % self.scheduler.order == 0:
                    progress_bar.update()

        if latents.ndim == 5:
            latents = latents[0]

        if self.trans_vae is None:
            return latents

        # Decoding
        self.vae.to(dtype=torch.bfloat16, device=device)
        self.trans_vae.to(dtype=torch.bfloat16, device=device)
        
        if fullpage is not None and page_alpha is not None:
            page_alpha = page_alpha.to(device)
        
        latents = latents.to(dtype=self.trans_vae.dtype, device=device) / self.vae.config.scaling_factor

        vis_list = []
        res_list = []
        for latent in latents:
            latent = latent[None]
            result_list, vis_list_batch = self.trans_vae.decoder(self.vae, latent, mask=page_alpha)
            vis_list += vis_list_batch
            res_list += result_list

        if self.blockswap_enabled:
            self.vae.to("cpu")
            self.trans_vae.to("cpu")
            torch.cuda.empty_cache()

        return LayerdiffPipelineOutput(images=res_list, vis_list=vis_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Blockswap inference: LayerDiff body+head -> Marigold depth -> PSD")
    parser.add_argument('--srcp', type=str, default='assets/test_image.png', help='input image')
    parser.add_argument('--save_dir', type=str, default='workspace/layerdiff_output_blockswap')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resolution', type=int, default=1280)
    parser.add_argument('--save_to_psd', action='store_true')
    parser.add_argument('--tblr_split', action='store_true')
    parser.add_argument('--repo_id_layerdiff', type=str, default='layerdifforg/seethroughv0.0.2_layerdiff3d')
    parser.add_argument('--num_inference_steps', type=int, default=30)
    parser.add_argument('--resolution_depth', type=int, default=768)
    args = parser.parse_args()

    srcname = osp.basename(osp.splitext(args.srcp)[0])
    saved = osp.join(args.save_dir, srcname)
    os.makedirs(saved, exist_ok=True)

    print(f"Building Blockswap pipeline (repo: {args.repo_id_layerdiff})...")
    trans_vae = TransparentVAE.from_pretrained(args.repo_id_layerdiff, subfolder='trans_vae', disable_mmap=True)
    unet = UNetFrameConditionModel.from_pretrained(args.repo_id_layerdiff, subfolder='unet', disable_mmap=True)
    pipeline = KDiffusionStableDiffusionXLPipelineBlockSwap.from_pretrained(
        args.repo_id_layerdiff, trans_vae=trans_vae, unet=unet, scheduler=None, disable_mmap=True
    )
    pipeline.enable_blockswap(device='cuda')
    pipeline.cache_tag_embeds()

    print('Running LayerDiff3D (body + head) with blockswap...')
    run_layerdiff(pipeline, args.srcp, args.save_dir, args.seed, args.num_inference_steps, args.resolution)
    
    del pipeline
    torch.cuda.empty_cache()

    print('\nBuilding Marigold depth pipeline...')
    marigold_args = argparse.Namespace(quant_mode='none', cpu_offload=False, repo_id_depth='24yearsold/seethroughv0.0.1_marigold')
    marigold_pipe = build_marigold_pipeline(marigold_args)

    print('Running Marigold depth...')
    run_marigold(marigold_pipe, args.srcp, args.save_dir, args.seed, args.resolution_depth)

    del marigold_pipe
    torch.cuda.empty_cache()

    print('\nRunning PSD assembly...')
    further_extr(saved, rotate=False, save_to_psd=args.save_to_psd, tblr_split=args.tblr_split)
    print("Done.")
