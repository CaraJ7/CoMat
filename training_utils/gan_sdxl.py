import torch.nn as nn
from training_utils.pipeline import *
from training_utils.pipeline import get_trainable_parameters as _get_trainable_parameters
import copy

class D_sd(nn.Module):
    def __init__(self, args, weight_dtype, device=None) -> None:
        super().__init__()
        self.D_args = copy.deepcopy(args)
        self.D_args.train_text_encoder_lora = False
        self.D_args.tune_text_encoder = False
        self.D_args.pretrain_model = "runwayml/stable-diffusion-v1-5"
        self.D_sd_pipeline = load_pipeline(self.D_args, args.gan_model_arch, weight_dtype, is_D=True).to(device)
        print("D_sd pipeline", self.D_sd_pipeline)

        self.weight_dtype = weight_dtype

        self.unet = self.D_sd_pipeline.unet

        if args.train_text_encoder_lora or args.tune_text_encoder:
            self.D_sd_pipeline.text_encoder.to(device)
            self.text_encoder = self.D_sd_pipeline.text_encoder
        
        self.ori_scheduler = copy.deepcopy(self.D_sd_pipeline.scheduler)
        
        # for classification
        if args.gan_unet_lastlayer_cls:
            ori_last_conv = self.D_sd_pipeline.unet.conv_out
            self.mlp = nn.Conv2d(ori_last_conv.in_channels, 1, ori_last_conv.kernel_size, ori_last_conv.padding, ori_last_conv.stride)
            self.unet.conv_out = self.mlp # to hack in accelerator.prepare
        else: # MLP
            self.mlp = nn.Sequential(
                nn.Linear(4, 1)
            )
        self.cls_loss_fn = nn.BCEWithLogitsLoss()
    
    def get_trainable_parameters(self):
        self.D_parameters = _get_trainable_parameters(self.D_args, self.D_sd_pipeline, is_D=True)
        self.D_parameters.extend([p for p in self.mlp.parameters()])
        return self.D_parameters
    
    def set_D_sd_pipeline_lora(self, requires_grad=True):
        for p in self.D_parameters:
            p.requires_grad = requires_grad

    def get_D_gt_noise(self, device, **kwargs):
        ori_latents = kwargs['batch']['latents'].to(device, dtype=self.weight_dtype)
        return ori_latents

    def D_sd_pipeline_forward(self, training_latents, side='G',**kwargs):
        device = training_latents.device
        if side == 'G':
            
            # set D_sd_pipeline no grad
            self.unet.eval()
            self.set_D_sd_pipeline_lora(requires_grad=False)

            # get discriminator condition
            if self.D_args.condition_discriminator:
                D_cond = self.pipeline.encode_prompt(
                    prompt=kwargs['prompt'], 
                    device=device, 
                    num_images_per_prompt=1, 
                    do_classifier_free_guidance=False)[0]
            else:
                D_cond = kwargs['negative_prompt_embeds']

            self.ori_scheduler.set_timesteps(kwargs['num_inference_steps'], device=device)
            timesteps = self.ori_scheduler.timesteps
            training_latents = self.ori_scheduler.scale_model_input(training_latents, timesteps[-1]) # identity in ddpm

            noise_pred = self.unet(
                training_latents,
                timesteps[-1], # marking its own step, could be out of domain
                encoder_hidden_states=D_cond,
                cross_attention_kwargs=None,
                return_dict=False,
            )[0]
            
            noise_pred = noise_pred.permute(0,2,3,1) # -> (bs, h, w, 4)
            if self.D_args.gan_unet_lastlayer_cls:
                pred = noise_pred # noise_pred -> (bs, h, w, 1)
            else:
                pred = self.mlp(noise_pred) # -> (bs, h, w, 2)
            target = torch.ones_like(pred) # -> (bs, h, w, 2)

            with torch.autocast('cuda'):
                gan_loss = self.cls_loss_fn(pred, target)
            return gan_loss

            
        elif side == 'D':

            self.unet.train()
            self.set_D_sd_pipeline_lora(requires_grad=True)
            training_latents.requires_grad_(False)

            D_cond = torch.cat(
                [kwargs['negative_prompt_embeds'], kwargs['negative_prompt_embeds']]
            )

            with torch.no_grad():
                ori_latents = self.get_D_gt_noise(device, **kwargs)
            

            self.ori_scheduler.set_timesteps(kwargs['num_inference_steps'], device=device)
            timesteps = self.ori_scheduler.timesteps
            
            input_latents = torch.cat([training_latents, ori_latents])
            input_latents = self.ori_scheduler.scale_model_input(input_latents, timesteps[-1]) # identity in ddpm

            noise_pred = self.unet(
                input_latents,
                timesteps[-1], # marking its own step, could be out of domain
                encoder_hidden_states=D_cond,
                cross_attention_kwargs=None,
                return_dict=False,
            )[0] # (2*bs, 4, h, w)

                
            noise_pred = noise_pred.permute(0,2,3,1) # -> (2*bs, h, w, 4)
            if self.D_args.gan_unet_lastlayer_cls:
                pred = noise_pred # noise_pred -> (bs, h, w, 1)
            else:
                pred = self.mlp(noise_pred) # -> (bs, h, w, 2)
            target = torch.ones_like(pred) # -> (2*bs, h, w, 1)

            target[:target.shape[0]//2] = 0 # label = 0 for generated_image

            with torch.autocast('cuda'):
                gan_loss = self.cls_loss_fn(pred, target)
            return gan_loss

    @torch.no_grad()
    def encode_prompt(self, prompt, device, batch_size, do_classifier_free_guidance=False):
        if isinstance(self.D_sd_pipeline, TrainableSDPipeline) or isinstance(self.D_sd_pipeline, SynTrainableSDPipeline):
            null_embed = self.D_sd_pipeline.encode_prompt(
                prompt, 
                device, 
                batch_size, 
                do_classifier_free_guidance=do_classifier_free_guidance
            )[0]
            pooled_null_embed = None
        elif isinstance(self.D_sd_pipeline, TrainableSDXLPipeline):
            null_embed, _, pooled_null_embed, _ = self.D_sd_pipeline.encode_prompt(
                prompt, 
                device=device, 
                num_images_per_prompt=batch_size, 
                do_classifier_free_guidance=do_classifier_free_guidance
            )
        # assume this function will be only called once
        self.D_sd_pipeline.text_encoder.to('cpu')
        self.D_sd_pipeline.vae.to('cpu')

        return null_embed, pooled_null_embed


class D_sdxl(D_sd):
    def __init__(self, args, weight_dtype, device=None) -> None:
        super().__init__()
        self.D_args = copy.deepcopy(args)
        self.D_args.train_text_encoder_lora = False
        self.D_args.tune_text_encoder = False
        self.D_args.pretrain_model = 'stabilityai/stable-diffusion-xl-base-1.0'
        self.D_sd_pipeline = load_pipeline(self.D_args, args.gan_model_arch, weight_dtype, is_D=True).to(device)

        self.weight_dtype = weight_dtype

        self.unet = self.D_sd_pipeline.unet

        if args.train_text_encoder_lora or args.tune_text_encoder:
            self.D_sd_pipeline.text_encoder.to(device)
            self.text_encoder = self.D_sd_pipeline.text_encoder
        
        self.ori_scheduler = copy.deepcopy(self.D_sd_pipeline.scheduler)
        
        # for classification
        if args.gan_unet_lastlayer_cls:
            ori_last_conv = self.D_sd_pipeline.unet.conv_out
            self.mlp = nn.Conv2d(ori_last_conv.in_channels, 1, ori_last_conv.kernel_size, ori_last_conv.padding, ori_last_conv.stride)
            self.unet.conv_out = self.mlp # to hack in accelerator.prepare
        else: # MLP
            self.mlp = nn.Sequential(
                nn.Linear(4, 1)
            )
        self.cls_loss_fn = nn.BCEWithLogitsLoss()

        if args.train_text_encoder_lora or args.tune_text_encoder:
            self.D_sd_pipeline.text_encoder_2.to(device)
            self.text_encoder_2 = self.D_sd_pipeline.text_encoder_2

        # used in add_time_ids
        height = args.resolution or self.D_sd_pipeline.default_sample_size * self.D_sd_pipeline.vae_scale_factor
        width = args.resolution or self.D_sd_pipeline.default_sample_size * self.D_sd_pipeline.vae_scale_factor

        self.original_size = (height, width)
        self.target_size = (height, width)
        self.crops_coords_top_left = (0, 0)
        self.add_time_ids = self._get_add_time_ids(
            self.original_size,
            self.crops_coords_top_left,
            self.target_size,
            weight_dtype
        ).to(device)


    def D_sd_pipeline_forward(self, training_latents, side='G',**kwargs):
        device = training_latents.device
        bs = training_latents.shape[0]

        if side == 'G':
            
            # set D_sd_pipeline no grad
            self.unet.eval()
            self.set_D_sd_pipeline_lora(requires_grad=False)

            D_cond = kwargs['negative_prompt_embeds']

            add_time_ids = self.add_time_ids.repeat(bs, 1)
            # Predict the noise residual
            unet_added_conditions = {"time_ids": add_time_ids}
            unet_added_conditions.update({"text_embeds": kwargs['negative_pooled_prompt_embeds']})

            self.ori_scheduler.set_timesteps(kwargs['num_inference_steps'], device=device)
            timesteps = self.ori_scheduler.timesteps
            training_latents = self.ori_scheduler.scale_model_input(training_latents, timesteps[-1]) # identity in ddpm

            noise_pred = self.unet(
                training_latents,
                timesteps[-1], # marking its own step, could be out of domain
                encoder_hidden_states=D_cond,
                added_cond_kwargs=unet_added_conditions,
                return_dict=False,
            )[0]
            
            # noise_pred = noise_pred.to(kwargs['negative_prompt_embeds'].dtype).flatten(-2).permute(0,2,1) # -> (2*bs,h*w,4)
            noise_pred = noise_pred.permute(0,2,3,1) # -> (bs, h, w, 4)
            if self.D_args.gan_unet_lastlayer_cls:
                pred = noise_pred # noise_pred -> (bs, h, w, 1)
            else:
                pred = self.mlp(noise_pred) # -> (bs, h, w, 2)
            target = torch.ones_like(pred) # -> (bs, h, w, 2)

            with torch.autocast('cuda'):
                gan_loss = self.cls_loss_fn(pred, target)
            return gan_loss

            
        elif side == 'D':

            self.unet.train()
            self.set_D_sd_pipeline_lora(requires_grad=True)
            training_latents.requires_grad_(False)

            D_cond = torch.cat(
                [kwargs['negative_prompt_embeds'], kwargs['negative_prompt_embeds']]
            )

            with torch.no_grad():
                ori_latents = self.get_D_gt_noise(device, **kwargs)

            add_time_ids = self.add_time_ids.repeat(2*bs, 1)
            # Predict the noise residual
            unet_added_conditions = {"time_ids": add_time_ids}
            unet_added_conditions.update(
                {"text_embeds": torch.cat([
                    kwargs['negative_pooled_prompt_embeds'], kwargs['negative_pooled_prompt_embeds']
                ], dim=0)}
            )

            self.ori_scheduler.set_timesteps(kwargs['num_inference_steps'], device=device)
            timesteps = self.ori_scheduler.timesteps

            input_latents = torch.cat([training_latents, ori_latents])
            input_latents = self.ori_scheduler.scale_model_input(input_latents, timesteps[-1]) # identity in ddpm

            noise_pred = self.unet(
                input_latents,
                timesteps[-1], # marking its own step, could be out of domain
                encoder_hidden_states=D_cond,
                added_cond_kwargs=unet_added_conditions,
                return_dict=False,
            )[0] # (2*bs, 4, h, w)

            noise_pred = noise_pred.permute(0,2,3,1) # -> (2*bs, h, w, 4)
            if self.D_args.gan_unet_lastlayer_cls:
                pred = noise_pred # noise_pred -> (bs, h, w, 1)
            else:
                pred = self.mlp(noise_pred) # -> (bs, h, w, 2)
            target = torch.ones_like(pred) # -> (2*bs, h, w, 1)
            target[:target.shape[0]//2] = 0 # label = 0 for generated_image

            with torch.autocast('cuda'):
                gan_loss = self.cls_loss_fn(pred, target)
            return gan_loss

    @torch.no_grad()
    def encode_prompt(self, prompt, device, batch_size, do_classifier_free_guidance=False):
        null_embed, pooled_null_embed = super().encode_prompt(prompt, device, batch_size, do_classifier_free_guidance)
        self.D_sd_pipeline.text_encoder_2.to('cpu')

        return null_embed, pooled_null_embed

    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype):
        from torch.nn.parallel import DistributedDataParallel
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        if isinstance(self.unet, DistributedDataParallel):
            passed_add_embed_dim = (
                self.unet.module.config.addition_time_embed_dim * len(add_time_ids) + self.D_sd_pipeline.text_encoder_2.config.projection_dim
            )
            expected_add_embed_dim = self.unet.module.add_embedding.linear_1.in_features
        else:
            passed_add_embed_dim = (
                self.unet.config.addition_time_embed_dim * len(add_time_ids) + self.D_sd_pipeline.text_encoder_2.config.projection_dim
            )
            expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids