
import numpy
import json
import matplotlib.pyplot as plt 
from PIL import Image
import os
# from petrel_client.client import Client
from aoss_client.client import Client

import torch
from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
import os
import io
import json
import random

def read_jsonl(save_path):
    ret_list = []
    with open(save_path, 'r') as f:
        for line in f:
            ret_list.append(json.loads(line))
    return ret_list


class Gan_Dataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        args,
    ):
        self.args = args
        
        if 'txt' in args.training_prompts:
            self.ann = list()
            with open(args.training_prompts, 'r') as f:
                for line in f:
                    self.ann.append(line.strip())
        elif 'jsonl' in args.training_prompts:
            self.ann = read_jsonl(args.training_prompts)
        elif 'json' in args.training_prompts:
            self.ann = json.load(open(args.training_prompts, 'r'))
        
        self.client = Client('~/aoss.conf')

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        example = {}
        example['text'] = self.ann[index]['prompt']
        latent_path = self.ann[index]['file_path'] if not isinstance(self.ann[index]['file_path'], list) else random.choice(self.ann[index]['file_path'])
        # use ceph
        with io.BytesIO(self.client.get(latent_path)) as f:
            latents = torch.load(f)
        # use disk
        # latents = torch.load(f)
        
        example['latents'] = latents
        
        # add another potential key in example
        eliminate_keys = ['prompt', 'file_path', 'image']
        for k in self.ann[index].keys():
            if k not in eliminate_keys:
                example[k] = self.ann[index][k]

        return example


