import torch.nn as nn
from training_utils.pipeline import *
from training_utils.pipeline import get_trainable_parameters as _get_trainable_parameters
from training_utils.gan_sdxl import D_sd, D_sdxl
import copy
import pdb

def load_discriminator(args, weight_dtype, device):
    args.gan_model_arch = args.gan_model_arch.replace('gan', '')
    
    if args.gan_model_arch == 'sd_1_5':
        return D_sd(args, weight_dtype, device)
    elif 'sdxl' in args.gan_model_arch :
        return D_sdxl(args, weight_dtype, device)

