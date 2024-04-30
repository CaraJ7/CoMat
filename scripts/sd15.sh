python -u -m accelerate.commands.launch --config_file node8.yaml --main_process_port 12213 \
training_script.py \
--pretrain_model runwayml/stable-diffusion-v1-5 --resolution 512 \
--train_batch_size 4 --gradient_accumulation_steps 1 --max_train_steps 2000 \
--learning_rate 5e-5 --max_grad_norm 0.1 --lr_scheduler constant --lr_warmup_steps 0 \
--output_dir output/sd15 \
--caption_model "Blip" --gradient_checkpointing \
--mixed_precision=fp16 --validation_prompts "A man walking on street" \
--seed 42 --K 5 --lora_rank 128 --training_prompts train_data/gan_abc5k_t2icomp_hrs_20k_sd15.jsonl \
--total_step 50 --scheduler DDPM \
--validation_prompts_file valid_15k.txt \
--gan_loss --gan_loss_weight 1 --learning_rate_D 2e-5 --adam_beta1_D 0 --max_grad_norm_D 1 \
--validation_steps 200 --pretrain_model_name sd_1_5_attrcon \
--mask_token_loss_weight 1e-3 --mask_pixel_loss_weight 5e-5 --attrcon_train_steps 2 \
--gan_model_arch gansd_1_5 --seg_model gsam 