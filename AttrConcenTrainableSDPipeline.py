from typing import Any, Callable, Dict, List, Optional, Union
import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg

from typing import Any, Callable, Dict, List, Optional, Union

from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    scale_lora_layers,
    unscale_lora_layers,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from TrainableSDPipeline import TrainableSDPipeline

import spacy
from attribute_concen_utils import *

from attn_utils.tc_attn_utils import get_cross_attn_map_from_unet


class AttrConcenTrainableSDPipeline(TrainableSDPipeline):
    def __init__(self, vae: AutoencoderKL, text_encoder: CLIPTextModel, tokenizer: CLIPTokenizer, unet: UNet2DConditionModel, scheduler: KarrasDiffusionSchedulers, safety_checker: StableDiffusionSafetyChecker, feature_extractor: CLIPImageProcessor, requires_safety_checker: bool = True):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor, requires_safety_checker)
        self.parser = spacy.load("en_core_web_trf")
        self.subtrees_indices = None
        self.doc = None
        self.contrast_loss = False
        self.attn_dict = {}
    
    # this is modified from the __call__ method of the StableDiffusionPipeline class
    def forward(
        self,
        prompt: Union[str, List[str]] = None,
        height: int = 512,
        width: int = 512,
        training_timesteps: Optional[List[int]] = [], # new training parameter
        early_exit: bool = False, # new training parameter
        detach_gradient: bool = True, # new training parameter
        train_text_encoder: bool = False, # new training parameter
        double_laststep: bool = False, # new training parameter
        bp_on_trained: bool = False, # new training parameter
        fast_training: bool = False, # new training parameter
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "image",
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        return_latents: bool = False, # new training parameter
        batch=None,
        attrcon_train_steps=None,
    ):
        # SYNGEN: NEW - use parsed_prompt instead of prompt
        self.doc = {}
        for p in prompt:
            self.doc[p] = self.parser(p)
        syn_gen_loss = negative_prompt_embeds.new_zeros(())
        self.subtrees_indices = dict()

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        torch.set_grad_enabled(train_text_encoder)
        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        prompt_embeds = prompt_embeds

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        if fast_training:
            self.scheduler.timesteps = [self.scheduler.timesteps[i] for i in training_timesteps]
            training_timesteps = range(len(self.scheduler.timesteps))
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            4,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Added
        text_embeddings = (
            prompt_embeds[batch_size * num_images_per_prompt:] if do_classifier_free_guidance else prompt_embeds
        )

        torch.set_grad_enabled(True)

        # training_steps
        attrcon_train_steps = attrcon_train_steps if attrcon_train_steps is not None else \
                                    training_timesteps

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        for i, t in enumerate(timesteps):            
            torch.set_grad_enabled((len(training_timesteps) == 0 or i > min(training_timesteps)) and not double_laststep)
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            torch.set_grad_enabled(i in training_timesteps and not double_laststep)

            do_detach = detach_gradient
            if i in training_timesteps and bp_on_trained:
                do_detach = False

            # predict the noise residual
            if i in training_timesteps and bp_on_trained and i in attrcon_train_steps:
                noise_pred = self._attrcon_forward(
                    latent_model_input.detach() if do_detach else latent_model_input, 
                    t,
                    prompt_embeds=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    prompt=prompt,
                    text_embeddings=text_embeddings
                )
            else:
                noise_pred = self.unet(
                    latent_model_input.detach() if do_detach else latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

            noise_pred = noise_pred.to(prompt_embeds.dtype)

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            if do_classifier_free_guidance and guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

            torch.set_grad_enabled((len(training_timesteps) == 0 or i >= min(training_timesteps)) and not double_laststep)

            # compute the previous noisy sample x_t -> x_t-1
            out = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=True)
            latents = out.prev_sample
            intermediate_result = out.pred_original_sample if hasattr(out, "pred_original_sample") else None

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

            if len(training_timesteps) > 0 and i == max(training_timesteps) and early_exit:
                latents = intermediate_result
                break
            
        if double_laststep:
            torch.set_grad_enabled(double_laststep)
            t = timesteps[training_timesteps[0]]
            noise = torch.randn_like(latents)
            noisy_latents = self.scheduler.add_noise(latents, noise, t)
            
            noisy_model_input = torch.cat([noisy_latents] * 2) if do_classifier_free_guidance else noisy_latents
            noisy_model_input = self.scheduler.scale_model_input(noisy_model_input, t)
            noise_pred = self.unet(
                noisy_model_input.detach() if do_detach else latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]
            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            if do_classifier_free_guidance and guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)
            
            out = self.scheduler.step(noise_pred, t, noisy_latents, **extra_step_kwargs, return_dict=True)
            latents = out.prev_sample
        
        if output_type == "image":
            image = self.vae.decode(latents.to(self.vae.dtype) / self.vae.config.scaling_factor, return_dict=False)[0]
            if return_latents:
                return (image / 2 + 0.5), latents
            return (image / 2 + 0.5) # .clamp(0, 1)
        elif output_type == "latent":
            return latents

    def _attrcon_forward(
            self,
            latents, # this should not detach
            t,
            prompt_embeds,
            cross_attention_kwargs=None,
            prompt=None,
            text_embeddings=None,
    ):  

        # forward with the whole
        self.controller.reset()
        noise_cond = self.unet(
            latents[latents.shape[0]//2:],
            t,
            encoder_hidden_states=prompt_embeds[latents.shape[0]//2:],
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=False,
        )[0]

        # Get attention maps
        attn_dict = get_cross_attn_map_from_unet(
                attention_store=self.controller, 
                is_training_sd21=False
            )
        self.attn_dict[str(t.cpu().item())] = attn_dict
        self.controller.reset()

        # the first half(uncond) can be directly forward
        noise_uncond = self.unet(
                latents[:latents.shape[0]//2],
                t,
                encoder_hidden_states=prompt_embeds[:latents.shape[0]//2],
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
        )[0]
        self.controller.reset()
        
        noise_pred = torch.cat([noise_uncond, noise_cond], dim=0)

        return noise_pred

    def _extract_attribution_indices(self, prompt):
        # extract standard attribution indices

        pairs = extract_attribution_indices(self.doc[prompt]) or []

        # extract attribution indices with verbs in between
        pairs_2 = extract_attribution_indices_with_verb_root(self.doc[prompt]) or []
        pairs_3 = extract_attribution_indices_with_verbs(self.doc[prompt]) or []
        # make sure there are no duplicates
        pairs = unify_lists(pairs, pairs_2, pairs_3)

        # MOD: filter too long pairs
        pairs = [p for p in pairs if len(p)<4]

        paired_indices = self._align_indices(prompt, pairs)
        return paired_indices    

    def _align_indices(self, prompt, spacy_pairs):
        wordpieces2indices = get_indices(self.tokenizer, prompt)
        paired_indices = []
        collected_spacy_indices = (
            set()
        )  # helps track recurring nouns across different relations (i.e., cases where there is more than one instance of the same word)

        for pair in spacy_pairs:
            curr_collected_wp_indices = (
                []
            )  # helps track which nouns and amods were added to the current pair (this is useful in sentences with repeating amod on the same relation (e.g., "a red red red bear"))
            for member in pair:
                for idx, wp in wordpieces2indices.items():
                    if wp in [start_token, end_token]:
                        continue

                    wp = wp.replace("</w>", "")
                    if member.text == wp:
                        if idx not in curr_collected_wp_indices and idx not in collected_spacy_indices:
                            curr_collected_wp_indices.append(idx)
                            break
                    # take care of wordpieces that are split up
                    elif member.text.startswith(wp) and wp != member.text:  # can maybe be while loop
                        wp_indices = align_wordpieces_indices(
                            wordpieces2indices, idx, member.text
                        )
                        # check if all wp_indices are not already in collected_spacy_indices
                        if wp_indices and (wp_indices not in curr_collected_wp_indices) and all([wp_idx not in collected_spacy_indices for wp_idx in wp_indices]):
                            curr_collected_wp_indices.append(wp_indices)
                            break

            for collected_idx in curr_collected_wp_indices:
                if isinstance(collected_idx, list):
                    for idx in collected_idx:
                        collected_spacy_indices.add(idx)
                else:
                    collected_spacy_indices.add(collected_idx)

            paired_indices.append(curr_collected_wp_indices)

        return paired_indices


    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
            else:
                scale_lora_layers(self.text_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )
            
            # MOD: accomodate for training text encoder
            if isinstance(self.text_encoder, DDP):
                if hasattr(self.text_encoder.module.config, "use_attention_mask") and self.text_encoder.module.config.use_attention_mask:
                    attention_mask = text_inputs.attention_mask.to(device)
                else:
                    attention_mask = None
            else:
                if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                    attention_mask = text_inputs.attention_mask.to(device)
                else:
                    attention_mask = None

            if clip_skip is None:
                prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
                prompt_embeds = prompt_embeds[0]
            else:
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
                )
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)

        if self.text_encoder is not None and isinstance(self.text_encoder, DDP):
            prompt_embeds_dtype = self.text_encoder.module.dtype
        elif self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            # MOD: accomodate for training text encoder
            if isinstance(self.text_encoder, DDP):
                if hasattr(self.text_encoder.module.config, "use_attention_mask") and self.text_encoder.module.config.use_attention_mask:
                    attention_mask = text_inputs.attention_mask.to(device)
                else:
                    attention_mask = None
            else:
                if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                    attention_mask = text_inputs.attention_mask.to(device)
                else:
                    attention_mask = None


            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if isinstance(self, LoraLoaderMixin) and USE_PEFT_BACKEND:
            # Retrieve the original scale by scaling back the LoRA layers
            unscale_lora_layers(self.text_encoder, lora_scale)

        return prompt_embeds, negative_prompt_embeds

def is_sublist(sub, main):
    # This function checks if 'sub' is a sublist of 'main'
    return len(sub) < len(main) and all(item in main for item in sub)

def unify_lists(lists_1, lists_2, lists_3):
    unified_list = lists_1 + lists_2 + lists_3
    sorted_list = sorted(unified_list, key=len)
    seen = set()

    result = []

    for i in range(len(sorted_list)):
        if tuple(sorted_list[i]) in seen:  # Skip if already added
            continue

        sublist_to_add = True
        for j in range(i + 1, len(sorted_list)):
            if is_sublist(sorted_list[i], sorted_list[j]):
                sublist_to_add = False
                break

        if sublist_to_add:
            result.append(sorted_list[i])
            seen.add(tuple(sorted_list[i]))

    return result