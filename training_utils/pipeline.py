import torch

from TrainableSDPipeline import TrainableSDPipeline, TrainableSDXLPipeline
from AttrConcenTrainableSDPipeline import AttrConcenTrainableSDPipeline
from AttrConcenTrainableSDXLPipeline import AttrConcenTrainableSDXLPipeline
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, DPMSolverMultistepScheduler
from diffusers.models.lora import LoRALinearLayer
from diffusers.loaders import (
    LoraLoaderMixin,
    text_encoder_lora_state_dict,
)
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    SlicedAttnAddedKVProcessor,
)


def _load_diffusion_pipeline(model_path, model_name, revision, weight_dtype, args=None):
    if model_name == 'sd_1_5':
        pipeline = TrainableSDPipeline.from_pretrained(model_path, revision=revision, torch_type=weight_dtype)
    elif model_name == 'sd_1_5_attrcon':
        pipeline = AttrConcenTrainableSDPipeline.from_pretrained(model_path, revision=revision, torch_type=weight_dtype)
    elif 'sdxl' in model_name:
        vae_path = "madebyollin/sdxl-vae-fp16-fix"
        vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16)
        if 'unet' in model_name:
            unet = UNet2DConditionModel.from_pretrained(args.sdxl_unet_path, revision=revision)
        else:
            unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", revision=revision)
        if 'attrcon' in model_name:
            PIPELINE_NAME = AttrConcenTrainableSDXLPipeline
        else:
            PIPELINE_NAME = TrainableSDXLPipeline
        pipeline = PIPELINE_NAME.from_pretrained(model_path, revision=revision, vae=vae, unet=unet, torch_type=weight_dtype)
            
    else:
        raise NotImplementedError("This model is not supported yet")
    return pipeline


def load_pipeline(args, model_name, weight_dtype, is_D=False):
    # Load pipeline
    pipeline = _load_diffusion_pipeline(args.pretrain_model, model_name, args.revision, weight_dtype, args)

    scheduler_args = {}

    if args.scheduler == "DPM++":
         pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    elif args.scheduler == "DDPM":
        if "variance_type" in  pipeline.scheduler.config:
            variance_type =  pipeline.scheduler.config.variance_type

            if variance_type in ["learned", "learned_range"]:
                variance_type = "fixed_small"

            scheduler_args["variance_type"] = variance_type

        pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config, **scheduler_args)
    if args.full_finetuning:
         pipeline.unet.to(dtype=torch.float)
    else:
        pipeline.unet.to(dtype=weight_dtype)
    pipeline.vae.to(dtype=weight_dtype)
    pipeline.text_encoder.to(dtype=weight_dtype)
    # set grad
    # Freeze vae and text_encoder
    pipeline.vae.requires_grad_(args.tune_vae)
    pipeline.text_encoder.requires_grad_(args.tune_text_encoder)
    pipeline.unet.requires_grad_(False)

    # gradient checkpoint
    if args.gradient_checkpointing:
        pipeline.unet.enable_gradient_checkpointing()
        if args.tune_text_encoder or args.train_text_encoder_lora:
            pipeline.text_encoder.gradient_checkpointing_enable()
            pipeline.text_encoder_2.gradient_checkpointing_enable() if hasattr(pipeline, "text_encoder_2") else None
    
    # set trainable lora
    pipeline = set_pipeline_trainable_module(args, pipeline, is_D=is_D)
    
    return pipeline

def set_pipeline_trainable_module(args, pipeline, is_D=False):
    if not args.full_finetuning:
        # Set correct lora layers
        for attn_processor_name, attn_processor in pipeline.unet.attn_processors.items():
            # Parse the attention module.
            attn_module = pipeline.unet
            for n in attn_processor_name.split(".")[:-1]:
                attn_module = getattr(attn_module, n)

            # Set the `lora_layer` attribute of the attention-related matrices.
            attn_module.to_q.set_lora_layer(
                LoRALinearLayer(
                    in_features=attn_module.to_q.in_features, out_features=attn_module.to_q.out_features, rank=args.lora_rank
                )
            )
            attn_module.to_k.set_lora_layer(
                LoRALinearLayer(
                    in_features=attn_module.to_k.in_features, out_features=attn_module.to_k.out_features, rank=args.lora_rank
                )
            )
            attn_module.to_v.set_lora_layer(
                LoRALinearLayer(
                    in_features=attn_module.to_v.in_features, out_features=attn_module.to_v.out_features, rank=args.lora_rank
                )
            )
            attn_module.to_out[0].set_lora_layer(
                LoRALinearLayer(
                    in_features=attn_module.to_out[0].in_features,
                    out_features=attn_module.to_out[0].out_features,
                    rank=args.lora_rank,
                )
            )

    if args.train_text_encoder_lora and not is_D:
        # ensure that dtype is float32, even if rest of the model that isn't trained is loaded in fp16
        text_lora_parameters = LoraLoaderMixin._modify_text_encoder(pipeline.text_encoder, dtype=torch.float32, rank=args.lora_rank)

    return pipeline

def get_trainable_parameters(args, pipeline, is_D=False):
    # load unet parameters
    if args.full_finetuning:
        G_parameters = list(pipeline.unet.parameters())
    else:
        G_parameters = []
        for attn_processor_name, attn_processor in pipeline.unet.attn_processors.items():

            attn_module = pipeline.unet
            for n in attn_processor_name.split(".")[:-1]:
                attn_module = getattr(attn_module, n)

            attn_module.to_q.lora_layer.to(torch.float)
            attn_module.to_k.lora_layer.to(torch.float)
            attn_module.to_v.lora_layer.to(torch.float)
            attn_module.to_out[0].lora_layer.to(torch.float)

            # Accumulate the LoRA params to optimize.
            G_parameters.extend(attn_module.to_q.lora_layer.parameters())
            G_parameters.extend(attn_module.to_k.lora_layer.parameters())
            G_parameters.extend(attn_module.to_v.lora_layer.parameters())
            G_parameters.extend(attn_module.to_out[0].lora_layer.parameters())

            if isinstance(attn_processor, (AttnAddedKVProcessor, SlicedAttnAddedKVProcessor, AttnAddedKVProcessor2_0)):
                import pdb
                pdb.set_trace() # should not enter here, otherwise the resume process is useless
                attn_module.add_k_proj.set_lora_layer(
                    LoRALinearLayer(
                        in_features=attn_module.add_k_proj.in_features,
                        out_features=attn_module.add_k_proj.out_features,
                        rank=args.lora_rank,
                    )
                )
                attn_module.add_v_proj.set_lora_layer(
                    LoRALinearLayer(
                        in_features=attn_module.add_v_proj.in_features,
                        out_features=attn_module.add_v_proj.out_features,
                        rank=args.lora_rank,
                    )
                )
                G_parameters.extend(attn_module.add_k_proj.lora_layer.parameters())
                G_parameters.extend(attn_module.add_v_proj.lora_layer.parameters())
    
    
    # load other parameters, not for D
    if not is_D:
        text_lora_parameters = []
        if args.tune_vae:
            G_parameters.extend(pipeline.vae.parameters())
        
        if args.tune_text_encoder:
            G_parameters.extend(pipeline.text_encoder.parameters())
        
        if args.train_text_encoder_lora:
            for n, p in pipeline.text_encoder.named_parameters():
                if 'lora' in n:
                    if not p.requires_grad:
                        import pdb
                        pdb.set_trace()
                    if p.dtype != torch.float:
                        p.data = p.data.to(torch.float)
                    text_lora_parameters.append(p)
    
        return G_parameters, text_lora_parameters
    else:
        return G_parameters