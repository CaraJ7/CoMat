import math
import os
import random
import json

import numpy as np
import torch
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from tqdm.auto import tqdm

from PIL import Image

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from diffusers import DDPMScheduler, UNet2DConditionModel, DPMSolverMultistepScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from diffusers.loaders import (
    LoraLoaderMixin,
    text_encoder_lora_state_dict,
)

from TrainableSDPipeline import TrainableSDPipeline, TrainableSDXLPipeline

# import training_utils
from training_utils.arguments import parse_args
from training_utils.logging import set_logger
from training_utils.pipeline import *
from training_utils.gan_sd_model import load_discriminator
from training_utils.dataset import get_dataset_dataloader

from attribute_concen_utils import get_attention_map_index_to_wordpiece
from concept_mat_utils.load_captionmodel import load_model

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.16.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def unet_lora_state_dict(unet: UNet2DConditionModel):
    r"""
    Returns:
        A state dict containing just the LoRA parameters.
    """
    lora_state_dict = {}

    for name, module in unet.named_modules():
        if hasattr(module, "set_lora_layer"):
            lora_layer = getattr(module, "lora_layer")
            if lora_layer is not None:
                current_lora_layer_sd = lora_layer.state_dict()
                for lora_layer_matrix_name, lora_param in current_lora_layer_sd.items():
                    # The matrix name can either be "down" or "up".
                    lora_state_dict[f"unet.{name}.lora.{lora_layer_matrix_name}"] = lora_param

    return lora_state_dict
    

class CaptionModelWrapper(torch.nn.Module):
    def __init__(self, caption_model, weights, device, args, dtype):
        super().__init__()

        self.caption_model = caption_model
        self.model_name = caption_model
        
        self.device = device
        self.dtype = dtype
        self.caption_model_dict = {}
        load_device = device

        self.weights = {}
        self.args = args
        for model, weight in zip(caption_model, weights):
            self.weights[model] = weight


        load_model(self, caption_model, load_device, args)

    def forward(self, images, prompts, text_encoder=None, return_feature=False, step=-1, batch=None):
        caption_rewards = {}

        if 'Blip' in self.model_name:
            caption_reward = self.blip_model.score(images, prompts, **batch)
            caption_rewards['Blip'] = caption_reward * self.weights['Blip']

        caption_rewards["total"] = sum([caption_rewards[k] for k in self.model_name])
        return caption_rewards

class Trainer(object):
    
    def __init__(self, pretrained_model_name_or_path, args):
            
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

        logging_dir = os.path.join(args.output_dir, args.logging_dir)

        accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit, logging_dir=logging_dir)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with=args.report_to,
            project_config=accelerator_project_config,
        )

        # If passed along, set the training seed now.
        if args.seed is not None:
            set_seed(args.seed)

        set_logger(args, self.accelerator, logger)

        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        self.model_name = args.pretrain_model_name
        self.pipeline = load_pipeline(args, self.model_name, self.weight_dtype)

        self.caption_model = CaptionModelWrapper(args.caption_model, weights=args.reward_weights, device=self.accelerator.device, args=args, dtype=self.weight_dtype)

        if args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers

                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    logger.warn(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                self.pipeline.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        if args.gan_loss:
            self.D = load_discriminator(args, self.weight_dtype, device=self.accelerator.device)

        self.global_step = 0
        self.first_epoch = 0
        self.resume_step = 0
        resume_global_step = 0
        # Potentially load in the weights and states from a previous save
        if args.resume_from_checkpoint:
            path = None
            if args.resume_from_checkpoint != "latest":
                # assert False, "not implemented"
                path = os.path.basename(args.resume_from_checkpoint)
                load_dir = os.path.dirname(args.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(args.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None
                load_dir = args.output_dir

            if path is None:
                self.accelerator.print(
                    f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                args.resume_from_checkpoint = None
            else:
                self.accelerator.print(f"Resuming from checkpoint {path}")
                if args.full_finetuning:
                    self.pipeline.unet.load_state_dict(torch.load(os.path.join(load_dir, path, "unet.pt"), map_location="cpu"))
                else:
                    lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(os.path.join(load_dir, path, "pytorch_lora_weights.safetensors"))
                    LoraLoaderMixin.load_lora_into_unet(lora_state_dict, network_alphas=network_alphas, unet=self.accelerator.unwrap_model(self.pipeline.unet))
                    if args.train_text_encoder_lora:
                        LoraLoaderMixin.load_lora_into_text_encoder(
                            lora_state_dict, network_alphas=network_alphas, text_encoder=self.accelerator.unwrap_model(self.pipeline.text_encoder)
                        )
                
                if args.tune_vae:
                    self.pipeline.vae.load_state_dict(torch.load(os.path.join(load_dir, path, "vae.pt"), map_location="cpu"))
                if args.tune_text_encoder:
                    self.pipeline.text_encoder.load_state_dict(torch.load(os.path.join(load_dir, path, "text_encoder.pt"), map_location="cpu"))
                if args.gan_loss and args.resume_from_checkpoint == 'latest':
                    print("Loading D_SD Lora")
                    lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(os.path.join(load_dir, path, 'D_sd', "pytorch_lora_weights.safetensors"))
                    LoraLoaderMixin.load_lora_into_unet(lora_state_dict, network_alphas=network_alphas, unet=self.accelerator.unwrap_model(self.D.unet))
                    print("Loading D_SD MLP")
                    self.D.mlp.load_state_dict(torch.load(os.path.join(load_dir, path, 'D_sd', "mlp.pt"), map_location="cpu"))
                    for p in self.D.mlp.parameters():
                        p.data = p.data.float()
                    if args.gan_unet_lastlayer_cls:
                        self.D.unet.conv_out = self.mlp
                    

                self.global_step = int(path.split("-")[1])

                resume_global_step = self.global_step * args.gradient_accumulation_steps

        # load_trainable parameters, should be done after resume
        G_parameters, text_lora_parameters = get_trainable_parameters(args, self.pipeline, is_D=False)
        
        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        # Initialize the optimizer
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
                )
            optimizer_cls = bnb.optim.AdamW8bit
        elif args.optimizer_class == 'AdamW':
            optimizer_cls = torch.optim.AdamW

        if args.train_text_encoder_lora:
            if args.textenc_lora_lr is None: # share lr with text lora
                G_parameters.extend(text_lora_parameters)
                self.G_parameters = G_parameters
                self.optimizer = optimizer_cls(
                    self.G_parameters,
                    lr=args.learning_rate,
                    betas=(args.adam_beta1, args.adam_beta2),
                    weight_decay=args.adam_weight_decay,
                    eps=args.adam_epsilon,
                )
            else:
                self.optimizer = optimizer_cls([
                    dict(
                        params=G_parameters, 
                        lr=args.learning_rate,
                        betas=(args.adam_beta1, args.adam_beta2),
                        weight_decay=args.adam_weight_decay,
                        eps=args.adam_epsilon),
                    dict(
                        params=text_lora_parameters, 
                        lr=args.textenc_lora_lr,
                        betas=(args.adam_beta1, args.adam_beta2),
                        weight_decay=args.adam_weight_decay,
                        eps=args.adam_epsilon)
                ])
                self.G_parameters = G_parameters
                self.G_parameters.extend(text_lora_parameters)

        else:
            self.G_parameters = G_parameters
            self.optimizer = optimizer_cls(
                self.G_parameters,
                lr=args.learning_rate,
                betas=(args.adam_beta1, args.adam_beta2),
                weight_decay=args.adam_weight_decay,
                eps=args.adam_epsilon,
            )

        self.D_optimizer = None
        if args.gan_loss:
            self.D_parameters = self.D.get_trainable_parameters()
            self.D_optimizer = optimizer_cls(
                self.D_parameters,
                lr=args.learning_rate_D,
                betas=(args.adam_beta1_D, args.adam_beta2_D),
                weight_decay=args.adam_weight_decay,
                eps=args.adam_epsilon,
            )
        
        # get train_dataset and train_dataloader
        self.train_dataset, self.train_dataloader = get_dataset_dataloader(args, self.accelerator)

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        self.num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * self.num_update_steps_per_epoch
            overrode_max_train_steps = True

        self.first_epoch = self.global_step // self.num_update_steps_per_epoch
        self.resume_step = resume_global_step % (self.num_update_steps_per_epoch * args.gradient_accumulation_steps)

        self.lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        )

        self.pipeline.to(torch_device=self.accelerator.device)

        self.caption_model.to(self.accelerator.device, dtype=self.weight_dtype)

        if args.tune_vae:
            self.pipeline.vae.to(dtype=torch.float)

        if args.tune_text_encoder:
            self.pipeline.text_encoder.to(dtype=torch.float)
        

        if 'attrcon' in args.pretrain_model_name :
            from attr_concen_utils.load_segmodel import load_seg_model
            if 'sdxl' in args.pretrain_model_name:
                from attn_utils.tc_sdxl_attn_utils import AttentionStore, register_attention_control
                train_layer_ls = ['mid_16', 'up_16', 'up_32']
            else:
                from attn_utils.tc_attn_utils import AttentionStore, register_attention_control
                train_layer_ls = ['mid_8', 'up_16', 'up_32', 'up_64']

                
            self.seg_model = load_seg_model(args, self.accelerator.device, train_layer_ls)
            self.pipeline.controller = AttentionStore(train_layer_ls)
            register_attention_control(self.pipeline.unet, self.pipeline.controller)

        # Prepare everything with our `self.accelerator`.
        if not args.gan_loss:
            self.pipeline.unet, self.optimizer, self.train_dataloader, self.lr_scheduler, self.D_optimizer, self.pipeline.text_encoder = self.accelerator.prepare(
                self.pipeline.unet, self.optimizer, self.train_dataloader, self.lr_scheduler, self.D_optimizer, self.pipeline.text_encoder
            )
        else: # add D_sd_pipeline
            self.pipeline.unet, self.optimizer, self.train_dataloader, self.lr_scheduler, self.D_optimizer, self.pipeline.text_encoder, self.D.unet, self.D.mlp = self.accelerator.prepare(
                self.pipeline.unet, self.optimizer, self.train_dataloader, self.lr_scheduler, self.D_optimizer, self.pipeline.text_encoder, self.D.unet, self.D.mlp
            )


        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        self.num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / args.gradient_accumulation_steps)
        if overrode_max_train_steps:
            args.max_train_steps = args.num_train_epochs * self.num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        args.num_train_epochs = math.ceil(args.max_train_steps / self.num_update_steps_per_epoch)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if self.accelerator.is_main_process:
            tracker_config = dict(vars(args))
            tracker_config.pop("validation_prompts")
            tracker_config.pop("caption_model")
            tracker_config.pop("reward_weights")
            tracker_config.pop("seg_model")
            none_keys = []
            for k, v in tracker_config.items():
                if v is None:
                    none_keys.append(k)
                # print(f"{k}: {type(v)}")
                
            for k in none_keys:
                tracker_config.pop(k)
            for k, v in tracker_config.items():
                print(f"{k}: {type(v)}")

            self.accelerator.init_trackers(args.tracker_project_name, tracker_config)
    
    def train(self, args):
            
        # Train!
        total_batch_size = args.train_batch_size * self.accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        
        global_step = self.global_step
        first_epoch = self.first_epoch
        resume_step = self.resume_step

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps), disable=not self.accelerator.is_local_main_process)
        progress_bar.set_description("Steps")

        def save_and_evaluate(output_dir, n_iter, save=True):
            unet = self.accelerator.unwrap_model(self.pipeline.unet)
            unet_lora_layers = unet_lora_state_dict(unet)

            text_encoder_lora_layers = None
            if args.train_text_encoder_lora:
                text_encoder = self.accelerator.unwrap_model(self.pipeline.text_encoder)
                text_encoder_lora_layers = text_encoder_lora_state_dict(text_encoder)

            if save:
                if args.full_finetuning:
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    torch.save(unet.state_dict(), os.path.join(output_dir, "unet.pt"))
                else:
                    LoraLoaderMixin.save_lora_weights(
                        save_directory=output_dir,
                        unet_lora_layers=unet_lora_layers,
                        text_encoder_lora_layers=text_encoder_lora_layers,
                    )
                if args.tune_vae:
                    torch.save(self.accelerator.unwrap_model(self.pipeline.vae).state_dict(), os.path.join(output_dir, "vae.pt"))
                
                if args.tune_text_encoder:
                    torch.save(self.accelerator.unwrap_model(self.pipeline.text_encoder).state_dict(), os.path.join(output_dir, "text_encoder.pt"))


                if args.gan_loss:
                    os.makedirs(os.path.join(output_dir, 'D_sd'), exist_ok=True)
                    D_sd_unet = self.accelerator.unwrap_model(self.D.unet)
                    D_sd_unet_lora_layers = unet_lora_state_dict(D_sd_unet)

                    D_sd_text_encoder_lora_layers = None
                    if args.full_finetuning:
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        torch.save(D_sd_unet.state_dict(), os.path.join(output_dir, 'D_sd', "unet.pt"))
                    else:
                        LoraLoaderMixin.save_lora_weights(
                            save_directory=os.path.join(output_dir, 'D_sd'),
                            unet_lora_layers=D_sd_unet_lora_layers,
                            text_encoder_lora_layers=D_sd_text_encoder_lora_layers,
                        )
                    # save mlp
                    torch.save(self.accelerator.unwrap_model(self.D.mlp).state_dict(), os.path.join(output_dir, 'D_sd', "mlp.pt"))

            def dummy_checker(image, device, dtype):
                return image, None
            
            # Load previous pipeline
            self.pipeline.run_safety_checker = dummy_checker
            ori_scheduler = self.pipeline.scheduler
            ori_unet = self.pipeline.unet
            self.pipeline.unet = unet
            if args.train_text_encoder_lora:
                ori_text_encoder = self.pipeline.text_encoder
                self.pipeline.text_encoder = self.accelerator.unwrap_model(self.pipeline.text_encoder)

            if args.scheduler == "DPM++":
                self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self.pipeline.scheduler.config)
            elif args.scheduler == "DDPM":
                # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
                scheduler_args = {}

                if "variance_type" in self.pipeline.scheduler.config:
                    variance_type = self.pipeline.scheduler.config.variance_type

                    if variance_type in ["learned", "learned_range"]:
                        variance_type = "fixed_small"

                    scheduler_args["variance_type"] = variance_type

                self.pipeline.scheduler = DDPMScheduler.from_config(self.pipeline.scheduler.config, **scheduler_args)

            images = []
            if args.validation_prompts and args.num_validation_images > 0:
                if args.validation_prompts_file is not None:
                    with open(args.validation_prompts_file, 'r') as f:
                        val_prompts_from_file = f.readlines()
                    validation_prompts = args.validation_prompts + val_prompts_from_file
                    validation_prompts = [p.strip() for p in validation_prompts]
                else:
                    validation_prompts = args.validation_prompts
                generator = torch.Generator(device=self.accelerator.device).manual_seed(args.seed) if args.seed else None
                # avoid oom by shrinking bs
                all_images = [[] for _ in range(args.num_validation_images)]
                for start in range(0, len(validation_prompts), 1):
                    prompts = validation_prompts[start: start+1]
                    with torch.autocast(device_type='cuda'):
                        images = [
                            self.pipeline(prompts, num_inference_steps=args.total_step, generator=generator, guidance_scale=args.cfg_scale, guidance_rescale=args.cfg_rescale).images
                            for _ in range(args.num_validation_images)
                        ]
                    for i, img in enumerate(images):
                        all_images[i].extend(img)

                images = all_images

                new_images = [[] for _ in validation_prompts]
                for image in images:
                    for i, img in enumerate(image):
                        new_images[i].append(img)

                for tracker in self.accelerator.trackers:
                    if tracker.name == "tensorboard":
                        for i, image in enumerate(new_images):
                            np_images = np.stack([np.asarray(img) for img in image])
                            tracker.writer.add_images(f"test_{i}", np_images, n_iter, dataformats="NHWC")

            self.pipeline.scheduler = ori_scheduler
            self.pipeline.unet = ori_unet
            if args.train_text_encoder_lora:
                self.pipeline.text_encoder = ori_text_encoder

        # evaluate before training
        if self.accelerator.is_main_process and not self.accelerator.is_last_process and global_step == 0 and resume_step == 0:
            with torch.no_grad():
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                save_and_evaluate(save_path, global_step)
                torch.cuda.empty_cache()
        self.accelerator.wait_for_everyone()

        # evaluate after resume
        if self.accelerator.is_main_process and not self.accelerator.is_last_process and global_step%100 == 0:
            with torch.no_grad():
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                save_and_evaluate(save_path, global_step, save=False)
                torch.cuda.empty_cache()
        self.accelerator.wait_for_everyone()


        if args.do_classifier_free_guidance:
            if args.gan_loss:
                with torch.no_grad():
                    gan_null_embed, gan_pooled_null_embed = self.D.encode_prompt("", self.accelerator.device, args.train_batch_size, do_classifier_free_guidance=False)

            # embed for pipeline
            with torch.no_grad():
                if isinstance(self.pipeline, TrainableSDPipeline):
                    null_embed = self.pipeline.encode_prompt("", self.accelerator.device, args.train_batch_size, do_classifier_free_guidance=False)[0]
                elif isinstance(self.pipeline, TrainableSDXLPipeline):
                    null_embed, _, pooled_null_embed, _ = self.pipeline.encode_prompt("", device=self.accelerator.device, num_images_per_prompt=args.train_batch_size, do_classifier_free_guidance=False)
                else:
                    raise NotImplementedError("This model is not supported yet")

        # remove unnecessary D pipeline vae and text encoders
        if args.gan_loss:
            del self.D.D_sd_pipeline.vae
            del self.D.D_sd_pipeline.text_encoder
            if isinstance(self.D.D_sd_pipeline, TrainableSDXLPipeline):
                del self.D.D_sd_pipeline.text_encoder_2

            torch.cuda.empty_cache()
        
        step_count = 0
        for epoch in range(first_epoch, args.num_train_epochs):
            self.pipeline.unet.train()
            if args.tune_text_encoder or args.train_text_encoder_lora:
                self.pipeline.text_encoder.train()
                self.pipeline.text_encoder_2.train() if hasattr(self.pipeline, "text_encoder_2") else None
            train_loss = 0.0
            for step, batch in enumerate(self.train_dataloader):
                # Skip steps until we reach the resumed step
                if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue

                if args.batch_repeat > 1:
                    batch['text'] = batch['text'] * args.batch_repeat

                total_step = args.total_step

                # train diffusion model
                with self.accelerator.accumulate(self.pipeline.unet):
                    # setting of backward
                    bp_on_trained = True
                    early_exit = False
                    double_laststep = False
                    fast_training = False

                    interval = total_step // args.K
                    max_start = total_step - interval * (args.K - 1) - 1
                    start = random.randint(0, max_start)
                    training_steps = list(range(start, total_step, interval))
                    detach_gradient = True
 
                    if args.tune_text_encoder or args.train_text_encoder_lora:
                        if isinstance(self.pipeline, TrainableSDPipeline):
                            null_embed = self.pipeline.encode_prompt("", self.accelerator.device, args.train_batch_size, do_classifier_free_guidance=False)[0]
                        elif isinstance(self.pipeline, TrainableSDXLPipeline):
                            null_embed, _, pooled_null_embed, _ = self.pipeline.encode_prompt("", device=self.accelerator.device, num_images_per_prompt=args.train_batch_size, do_classifier_free_guidance=False)

                    kwargs = dict(
                        prompt=batch["text"],
                        height=args.resolution, 
                        width=args.resolution, 
                        training_timesteps=training_steps, 
                        detach_gradient=detach_gradient,
                        train_text_encoder=args.tune_text_encoder or args.train_text_encoder_lora, 
                        num_inference_steps=total_step, 
                        guidance_scale=args.cfg_scale, 
                        guidance_rescale=args.cfg_rescale,
                        negative_prompt_embeds=null_embed if args.do_classifier_free_guidance else None,
                        early_exit=early_exit,
                        return_latents=True if args.gan_loss else False,
                    )
                    if 'attrcon' in args.pretrain_model_name:
                        kwargs['attrcon_train_steps'] = random.choices(training_steps, k=min(args.attrcon_train_steps, len(training_steps)))

                    if isinstance(self.pipeline, TrainableSDPipeline):
                        if args.gan_loss:
                            image, training_latents = self.pipeline.forward(bp_on_trained=bp_on_trained, double_laststep=double_laststep, fast_training=fast_training, **kwargs)
                        else:
                            image = self.pipeline.forward(bp_on_trained=bp_on_trained, double_laststep=double_laststep, fast_training=fast_training, **kwargs)
                    elif isinstance(self.pipeline, TrainableSDXLPipeline):
                        if args.gan_loss:
                            image, training_latents = self.pipeline.forward(negative_pooled_prompt_embeds=pooled_null_embed if args.do_classifier_free_guidance else None, **kwargs)
                        else:
                            image = self.pipeline.forward(negative_pooled_prompt_embeds=pooled_null_embed if args.do_classifier_free_guidance else None, **kwargs)
                    else:
                        raise NotImplementedError("This model is not supported yet")
                    
                    # reward 
                    offset_range = args.resolution // 224
                    random_offset_x = random.randint(0, offset_range)
                    random_offset_y = random.randint(0, offset_range)
                    size = args.resolution - offset_range
                    caption_rewards = self.caption_model(
                        image[:,:,random_offset_x:random_offset_x + size, random_offset_y:random_offset_y + size].to(self.weight_dtype), 
                        batch['text'], 
                        step=step_count,
                        text_encoder=self.pipeline.text_encoder,
                        batch=batch)
                    step_count += 1

                    loss = - caption_rewards["total"].mean()

                    if args.gan_loss:
                        kwargs['negative_prompt_embeds'] = gan_null_embed # used in D_sd
                        kwargs['negative_pooled_prompt_embeds'] = gan_pooled_null_embed    

                        G_loss = self.D.D_sd_pipeline_forward(training_latents, side='G' ,**kwargs)
                        loss += args.gan_loss_weight * G_loss

                    if 'attrcon' in args.pretrain_model_name:

                        all_subtree_indices = [self.pipeline._extract_attribution_indices(p) for p in batch['text']]
                        attn_map_idx_to_wp_all = [get_attention_map_index_to_wordpiece(self.pipeline.tokenizer, p) for p in batch['text'] ]
                        attn_map = self.pipeline.attn_dict

                        token_loss, pixel_loss, grounding_loss_dict = self.seg_model.get_mask_loss(
                            image.clamp(0, 1), # this does not require grad, so clamp(0,1) is better
                            batch['text'],
                            all_subtree_indices, 
                            attn_map_idx_to_wp_all,
                            attn_map)
                        loss += args.mask_token_loss_weight * token_loss
                        loss += args.mask_pixel_loss_weight * pixel_loss

                        self.pipeline.attn_dict = {} # clear the attn_dict after usage, or it will cause error in next iter

                    norm = {}
                    def record_grad(grad):
                        norm['reward_norm'] = grad.norm(2).item()
                        if args.norm_grad:
                            grad = grad / (norm['reward_norm'] / 1e4) # 1e4 for numerical stability
                        return grad
                    
                    image.register_hook(record_grad)

                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = self.accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                    train_loss += avg_loss.item() / args.gradient_accumulation_steps

                    # Backpropagate
                    self.optimizer.zero_grad()
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.G_parameters, args.max_grad_norm)
                    self.optimizer.step()
                    self.lr_scheduler.step()


                logs = {"step_loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0]}
                logs.update({k: self.accelerator.gather(v.detach()).mean().item() for k, v in caption_rewards.items()})
                
                if args.gan_loss:
                    logs.update({'G_loss': self.accelerator.gather(G_loss.detach()).mean().item()})
            
                if 'attrcon' in args.pretrain_model_name:
                    logs.update({'token_loss': self.accelerator.gather(token_loss.detach()).mean().item()}) 
                    logs.update({'pixel_loss': self.accelerator.gather(pixel_loss.detach()).mean().item()})                      

                logs.update(norm)

                if args.gan_loss:
                    with self.accelerator.accumulate(self.D):
                        kwargs['batch'] = batch 
                            
                        D_loss = self.D.D_sd_pipeline_forward(training_latents.detach(), side='D', **kwargs)
                        avg_D_loss = self.accelerator.gather(D_loss.detach()).mean().item()
                        logs.update(dict(
                            D_loss=avg_D_loss,
                        ))

                        self.D_optimizer.zero_grad()
                        self.accelerator.backward(D_loss)

                        if self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(self.D_parameters, args.max_grad_norm_D)
                        self.D_optimizer.step()


                # Checks if the self.accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:

                    progress_bar.update(1)
                    global_step += 1
                    self.accelerator.log({"train_loss": train_loss}, step=global_step)
                    self.accelerator.log(logs, step=global_step)
                    train_loss = 0.0

                    logger.info(f"{global_step}: {json.dumps(logs, sort_keys=False, indent=4)}")

                progress_bar.set_postfix(**logs)

                if global_step % args.validation_steps == 0 and self.accelerator.sync_gradients:
                    if self.accelerator.is_main_process:
                        with torch.no_grad():
                            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                            save_and_evaluate(save_path, global_step)
                    torch.cuda.empty_cache()
                self.accelerator.wait_for_everyone()

                if global_step >= args.max_train_steps:
                    break


        # Save the lora layers
        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args.pretrain_model, args=args)
    trainer.train(args=args)