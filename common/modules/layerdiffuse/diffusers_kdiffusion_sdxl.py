from dataclasses import dataclass
from typing import Union, List, Optional
import gc

import numpy as np
from tqdm.auto import trange
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import *
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler, EulerDiscreteScheduler
from diffusers.utils.outputs import BaseOutput

from modules.layerdiffuse.vae import TransparentVAEDecoder, TransparentVAEEncoder, vae_encode
from .layerdiff3d import UNetFrameConditionModel
from utils.torch_utils import seed_everything, img2tensor, tensor2img

@dataclass
class LayerdiffPipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]
    vis_list: Union[List[PIL.Image.Image], np.ndarray]

@torch.no_grad()
def sample_dpmpp_2m(model, x, sigmas, extra_args=None, callback=None, show_progress=True, c_concat=None):
    """DPM-Solver++(2M)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None

    for i in trange(len(sigmas) - 1, disable=not show_progress):
        model_input = x
        denoised = model(model_input, sigmas[i] * s_in, c_concat=c_concat, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_denoised is None or sigmas[i + 1] == 0:
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
        old_denoised = denoised
    return x


class KDiffusionStableDiffusionXLPipeline(StableDiffusionXLImg2ImgPipeline):
    _optional_components = [
        "tokenizer",
        "tokenizer_2",
        "text_encoder",
        "text_encoder_2",
        "image_encoder",
        "feature_extractor",
    ]
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

        if scheduler is None:
            config_min = {"final_sigmas_type":"sigma_min"}
            config_min_euler = {"final_sigmas_type":"sigma_min", "euler_at_final": True }
            config_zero = {"final_sigmas_type":"zero"}
            schedulers = {
                "DPMPP_2M": {
                    "min": (DPMSolverMultistepScheduler, config_min),
                    "min_euler": (DPMSolverMultistepScheduler, config_min_euler),
                    "zero": (DPMSolverMultistepScheduler, config_zero),
                },
                "DPMPP_2M_K": {
                    "min": (DPMSolverMultistepScheduler, {"use_karras_sigmas": True, **config_min}),
                    "min_euler": (DPMSolverMultistepScheduler, {"use_karras_sigmas": True, **config_min_euler}),
                    "zero": (DPMSolverMultistepScheduler, {"use_karras_sigmas": True, **config_zero}),
                },
                "DPMPP_2M_SDE": {
                    "min": (DPMSolverMultistepScheduler, {"algorithm_type": "sde-dpmsolver++", **config_min}),
                    "min_euler": (DPMSolverMultistepScheduler, {"algorithm_type": "sde-dpmsolver++", **config_min_euler}),
                    "zero": (DPMSolverMultistepScheduler, {"algorithm_type": "sde-dpmsolver++", **config_zero}),
                },
                "DPMPP_2M_SDE_K": {
                    "min": (DPMSolverMultistepScheduler, {"algorithm_type": "sde-dpmsolver++", "use_karras_sigmas": True, **config_min}),
                    "min_euler": (DPMSolverMultistepScheduler, {"algorithm_type": "sde-dpmsolver++", "use_karras_sigmas": True, **config_min_euler}),
                    "zero": (DPMSolverMultistepScheduler, {"use_karras_sigmas": True, "algorithm_type": "sde-dpmsolver++", **config_zero}),
                },
                "DPMPP": {
                    "min": (DPMSolverSinglestepScheduler, config_min),
                    "min_euler": (DPMSolverSinglestepScheduler, config_min_euler),
                    "zero": (DPMSolverSinglestepScheduler, config_zero),
                },
                "DPMPP_K": {
                    "min": (DPMSolverSinglestepScheduler, {"use_karras_sigmas": True, **config_min}),
                    "min_euler": (DPMSolverSinglestepScheduler, {"use_karras_sigmas": True, **config_min_euler}),
                    "zero": (DPMSolverSinglestepScheduler, {"use_karras_sigmas": True, **config_zero}),
                },
            }
            model_id = "frankjoshua/juggernautXL_version6Rundiffusion"
            scheduler_name = "DPMPP_2M_SDE"
            scheduler_config_name = "zero"
            scheduler_configs = schedulers[scheduler_name]
            scheduler = scheduler_configs[scheduler_config_name][0].from_pretrained(
                    model_id,
                    subfolder="scheduler",
                    **scheduler_configs[scheduler_config_name][1],
            )

        super().__init__(
            vae=vae, text_encoder=text_encoder, text_encoder_2=text_encoder_2, tokenizer=tokenizer, tokenizer_2=tokenizer_2,
            unet=unet, scheduler=scheduler,feature_extractor=feature_extractor, image_encoder=image_encoder, requires_aesthetics_score=requires_aesthetics_score,
            force_zeros_for_empty_prompt=force_zeros_for_empty_prompt, add_watermarker=add_watermarker)
        # self.register_to_config(tag_list=tag_list)
        self.register_modules(trans_vae=trans_vae)

        self._cached_prompt_embeds = {}

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None


    @torch.inference_mode()
    def encode_cropped_prompt_77tokens(self, prompt: str):
        device = self.text_encoder.device
        tokenizers = [self.tokenizer, self.tokenizer_2]
        text_encoders = [self.text_encoder, self.text_encoder_2]

        pooled_prompt_embeds = None
        prompt_embeds_list = []

        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_input_ids = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids

            prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True, return_dict=False)

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds[-1][-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1).to(dtype=self.unet.dtype, device=device)
        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)

        # prompt_embeds = prompt_embeds.to(dtype=self.unet.dtype, device=device)

        return prompt_embeds, pooled_prompt_embeds

    def encode_cropped_prompt_77tokens_cached(self, prompt: Union[str, List]):
        if isinstance(prompt, str):
            prompt = [prompt]

        input_prompts = []
        _cached_prompt_embeds = {}
        for p in prompt:
            if p not in self._cached_prompt_embeds:
                input_prompts.append(p)
            else:
                _cached_prompt_embeds[p] = self._cached_prompt_embeds[p]
        if len(input_prompts) > 0:
            prompt_embeds, pooled_prompt_embeds = self.encode_cropped_prompt_77tokens(input_prompts)
            for ii in range(len(input_prompts)):
                _cached_prompt_embeds[input_prompts[ii]] = [prompt_embeds[[ii]].cpu(), pooled_prompt_embeds[[ii]].cpu()]
        
        prompt_embeds_out, pooled_prompt_embeds_out = [], []
        for ii in range(len(prompt)):
            prompt_embeds, pooled_prompt_embeds = _cached_prompt_embeds[prompt[ii]]
            prompt_embeds_out.append(prompt_embeds)
            pooled_prompt_embeds_out.append(pooled_prompt_embeds)

        pooled_prompt_embeds_out = torch.cat(pooled_prompt_embeds_out)
        prompt_embeds_out = torch.cat(prompt_embeds_out)
        return prompt_embeds_out, pooled_prompt_embeds_out

    def cache_tag_embeds(self, unload_textencoders=True):
        tag_version = self.unet.get_tag_version()
        if tag_version == 'v3' and len(self._cached_prompt_embeds) == 0:
            body_tag_list = ['front hair', 'back hair', 'head', 'neck', 'neckwear', 'topwear', 'handwear', 'bottomwear', 'legwear', 'footwear', 'tail', 'wings', 'objects']
            head_tag_list = ['headwear', 'face', 'irides', 'eyebrow', 'eyewhite', 'eyelash', 'eyewear', 'ears', 'earwear', 'nose', 'mouth']
            prompt_embeds, pooled_prompt_embeds = self.encode_cropped_prompt_77tokens(body_tag_list)
            for ii in range(len(body_tag_list)):
                self._cached_prompt_embeds[body_tag_list[ii]] = [prompt_embeds[[ii]].cpu(), pooled_prompt_embeds[[ii]].cpu()]
            prompt_embeds, pooled_prompt_embeds = self.encode_cropped_prompt_77tokens(head_tag_list)
            for ii in range(len(head_tag_list)):
                self._cached_prompt_embeds[head_tag_list[ii]] = [prompt_embeds[[ii]].cpu(), pooled_prompt_embeds[[ii]].cpu()]
        elif len(self._cached_prompt_embeds) == 0:
            body_tag_list = [
                'hair', 'headwear', 'face', 'eyes', 'eyewear', 'ears', 'earwear', 'nose', 'mouth', 
                'neck', 'neckwear', 'topwear', 'handwear', 'bottomwear', 'legwear', 'footwear', 
                'tail', 'wings', 'objects'
            ]
            prompt_embeds, pooled_prompt_embeds = self.encode_cropped_prompt_77tokens(body_tag_list)
            for ii in range(len(body_tag_list)):
                self._cached_prompt_embeds[body_tag_list[ii]] = [prompt_embeds[[ii]].cpu(), pooled_prompt_embeds[[ii]].cpu()]
        else:
            unload_textencoders = False

        if unload_textencoders:
            self.text_encoder.cpu()
            self.text_encoder_2.cpu()
            del self.text_encoder
            del self.text_encoder_2
            # to supress some warning msg
            self.text_encoder = self.text_encoder_2 = torch.nn.Identity()
            gc.collect()
            torch.cuda.empty_cache()
        
    
    def denoise_func(self, latents, add_text_embeds, add_time_ids, prompt_embeds, c_concat, num_inference_steps=50):

        # 4. Prepare timesteps
        device = self.unet.device
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps=None, sigmas=None
        )

        latents = latents * self.scheduler.init_noise_sigma

        for i, t in enumerate(timesteps):

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

            noise_pred = self.unet(
                torch.cat([latent_model_input, c_concat], dim=-3),
                t,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                # Based on 3.4. in https://huggingface.co/papers/2305.08891
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

            # compute the previous noisy sample x_t -> x_t-1
            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    latents = latents.to(latents_dtype)

        return latents


    @property
    def device(self) -> torch.device:
        return self.unet.device

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
            group_index=None,
            group_embd_indices=None,
    ):

        device = self.unet.device
        dtype = self.unet.dtype

        if fullpage is not None:
            page_alpha = img2tensor(fullpage[..., -1] / 255., device=self.vae.device, dtype=self.vae.dtype)[0][..., None]
            fullpage = fullpage[..., :3]
            c_concat = np.concatenate([np.full_like(fullpage[..., :1], fill_value=255), fullpage], axis=2)
            c_concat = img2tensor(c_concat, normalize=True)
            c_concat = vae_encode(self.vae, self.trans_vae.encoder, c_concat, use_offset=False).to(device=device, dtype=dtype)
            c_concat = c_concat.to(dtype=dtype)

        assert c_concat is not None

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
            initial_latent = torch.zeros((batch_size, 4, lh, lw), device=self.unet.device, dtype=self.unet.dtype)

        if is_3d and c_concat.ndim == 4:
            c_concat = c_concat[:, None].expand(-1, num_frames, -1, -1, -1)

        if is_3d and initial_latent.ndim == 4:
            initial_latent = initial_latent[:, None].expand(-1, num_frames, -1, -1, -1)

        if prompt is not None:
            prompt_embeds, pooled_prompt_embeds = self.encode_cropped_prompt_77tokens_cached(prompt)

        if negative_prompt is not None and self.do_classifier_free_guidance:
            negative_prompt_embeds, negative_pooled_prompt_embeds = self.encode_cropped_prompt_77tokens(negative_prompt)

        # Initial latents

        # noise = randn_tensor(initial_latent.shape, generator=generator, device=device, dtype=self.unet.dtype)
        noise = randn_tensor(initial_latent[:, [0]].shape, generator=generator, device=device, dtype=self.unet.dtype).expand(-1, num_frames, -1, -1, -1)
        # latents = initial_latent.to(noise) + noise * sigmas[0].to(noise)

        height = lh * self.vae_scale_factor
        width = lw * self.vae_scale_factor

        add_time_ids = list((height, width) + (0, 0) + (height, width))
        add_time_ids = torch.tensor([add_time_ids], dtype=self.unet.dtype)
        add_time_ids = add_time_ids.expand((prompt_embeds.shape[0], -1))
        add_neg_time_ids = add_time_ids.clone()

        # Batch

        # latents = latents.to(device)
        add_time_ids = add_time_ids.repeat(batch_size, 1).to(device)
        add_neg_time_ids = add_neg_time_ids.repeat(batch_size, 1).to(device)
        prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1).to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(batch_size, 1).to(device)

        sampler_kwargs = dict(
            cfg_scale=guidance_scale,
            positive=dict(
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids},)
        )

        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.repeat(batch_size, 1, 1).to(device)
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(batch_size, 1).to(device)
            sampler_kwargs['negative'] = dict(
                encoder_hidden_states=negative_prompt_embeds,
                added_cond_kwargs={"text_embeds": negative_pooled_prompt_embeds, "time_ids": add_neg_time_ids},
            )

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

                noise_pred = self.unet(
                    torch.cat([latent_model_input, c_concat], dim=-3),
                    t,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                    group_index=group_index,
                    group_embd_indices=group_embd_indices,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://huggingface.co/papers/2305.08891
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if i == len(timesteps) - 1 or (i + 1) % self.scheduler.order == 0:
                    progress_bar.update()


        if latents.ndim == 5:
            latents = latents[0]

        if self.trans_vae is None:
            return latents

        latents = latents.to(dtype=self.trans_vae.dtype, device=self.trans_vae.device) / self.vae.config.scaling_factor

        vis_list = []
        res_list = []
        for latent in latents:
            latent = latent[None]
            result_list, vis_list_batch = self.trans_vae.decoder(self.vae, latent, mask=page_alpha)
            vis_list += vis_list_batch
            res_list += result_list

        return LayerdiffPipelineOutput(images=res_list, vis_list=vis_list)
