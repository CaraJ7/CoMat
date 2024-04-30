import argparse
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:18000"
import random
import json

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from packaging import version
from tqdm.auto import tqdm

from PIL import Image

from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
from torchvision.transforms.functional import pil_to_tensor
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, DPMSolverMultistepScheduler
from diffusers.utils.import_utils import is_xformers_available

from TrainableSDPipeline import TrainableSDPipeline, TrainableSDXLPipeline
from torchvision import transforms as T

import shortuuid
import time

import threading

# build a lock
lock = threading.Lock()

def write_data(file_name, data):
    # obtain the lock
    with lock:
        with open(file_name, 'a') as f:
            f.write(data + '\n')

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default="0")
parser.add_argument("--end", type=int, default="10000")
parser.add_argument("--unet-path", type=str)
parser.add_argument("--use-cache", action='store_true')
parser.add_argument("--prompt-path", type=str, default='merged_data/abc5k_hrs10k_t2icompall_20k.txt')
parser.add_argument("--save-prompt-path", type=str, default='train_data/gan_train_data.jsonl')
parser.add_argument("--model-path", type=str, default="runwayml/stable-diffusion-v1-5")
parser.add_argument("--model-type", type=str, default='sd_1_5')
parser.add_argument("--batch-size", type=int, default=8)
args = parser.parse_args()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# random seed
timestamp = int(time.time())
setup_seed(timestamp)


def load_diffusion_pipeline(model_path, revision, weight_dtype, args):
    if args.model_type == 'sd_1_5':
        pipeline = TrainableSDPipeline.from_pretrained(model_path, revision=revision, torch_type=weight_dtype)
    elif args.model_type == 'sdxl':
        vae_path = "madebyollin/sdxl-vae-fp16-fix"
        vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16)
        pipeline = TrainableSDXLPipeline.from_pretrained(model_path, revision=revision, vae=vae, torch_type=weight_dtype)
    elif args.model_type == 'sdxl_unet': # for a UNet fine-tuned on 512
        vae_path = "madebyollin/sdxl-vae-fp16-fix"
        vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16)
        unet_path = args.unet_path
        unet = UNet2DConditionModel.from_pretrained(unet_path)
        pipeline = TrainableSDXLPipeline.from_pretrained(model_path, revision=revision, vae=vae, unet=unet, torch_type=weight_dtype)
    else:
        raise NotImplementedError("This model is not supported yet")
    return pipeline

def read_jsonl(save_path):
    ret_list = []
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            for line in f:
                ret_list.append(json.loads(line))
    return ret_list

class Prompt_dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, save_path, start, end) -> None:
        super().__init__()
        if 'txt' in data_path:
            self.ann = list()
            with open(data_path, 'r') as f:
                for line in f:
                    self.ann.append(line.strip())
        elif 'json' in data_path:
            self.ann = json.load(open(data_path, 'r'))
        self.ann = self.ann[start:end]

        if args.use_cache:
            ann_done = read_jsonl(save_path)
            prompts_done = [inst['prompt'] for inst in ann_done]
            self.ann = list(set(self.ann)-set(prompts_done))
    
    def __getitem__(self, idx):
        return self.ann[idx]
    
    def __len__(self):
        return len(self.ann)

# load dataset
prompt_path = args.prompt_path
save_prompt_path = args.save_prompt_path

save_dir = os.path.dirname(save_prompt_path)
os.makedirs(save_dir, exist_ok=True)
os.makedirs(os.path.join(save_dir, 'latents'), exist_ok=True)

prompt_dataset = Prompt_dataset(prompt_path, save_prompt_path, args.start, args.end)
batch_size = args.batch_size
prompt_loader = torch.utils.data.DataLoader(
    prompt_dataset,
    shuffle=False,
    batch_size=batch_size,
    num_workers=4
)

# load model
model_path = args.model_path
pipeline = load_diffusion_pipeline(model_path, None, torch.float16, args)
pipeline.to('cuda')
pipeline.vae.requires_grad_(False)
pipeline.text_encoder.requires_grad_(False)
pipeline.unet.requires_grad_(False)
pipeline.text_encoder.to(torch.float16)
pipeline.unet.to(torch.float16)
if isinstance(pipeline, TrainableSDXLPipeline):
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_2.to(torch.float16)


if is_xformers_available():
    import xformers

    xformers_version = version.parse(xformers.__version__)
    if xformers_version == version.parse("0.0.16"):
        print(
            "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
        )
    pipeline.enable_xformers_memory_efficient_attention()
else:
    raise ValueError("xformers is not available. Make sure it is installed correctly")


scheduler_args = {}
if "variance_type" in pipeline.scheduler.config:
    variance_type = pipeline.scheduler.config.variance_type

    if variance_type in ["learned", "learned_range"]:
        variance_type = "fixed_small"

    scheduler_args["variance_type"] = variance_type

pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config, **scheduler_args)
generator = torch.Generator(device='cuda').manual_seed(int(time.time()))

with torch.no_grad():
    for idx, prompts in tqdm(enumerate(prompt_loader)):
        latents = pipeline(
            prompts, 
            height=512,
            width=512,
            num_inference_steps=50, 
            generator=generator, 
            guidance_scale=7.5, 
            guidance_rescale=0.0,
            output_type='latent').images
        
        save_list = []
        for idx in range(latents.shape[0]):
            uid = shortuuid.uuid()
            torch.save(latents[idx].cpu().float(), os.path.join(save_dir, 'latents', f"{uid}.pt"))
            save_dict = json.dumps({
                'prompt': prompts[idx],
                'file_path': os.path.join(save_dir, 'latents', f"{uid}.pt")
            })
            save_list.append(save_dict)

        write_data(save_prompt_path, '\n'.join(save_list))




