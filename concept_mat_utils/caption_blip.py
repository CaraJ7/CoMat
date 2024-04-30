import torch
import copy
from dataclasses import dataclass, field
from PIL import Image

from torchvision.transforms import Normalize, Compose, RandomResizedCrop, \
                                    InterpolationMode, ToTensor, Resize, CenterCrop

from transformers import BlipForConditionalGeneration
from concept_mat_utils.processing_blip import BlipProcessor

IGNORE_INDEX = -100

class Blip(torch.nn.Module):
    def __init__(self, model_path, device, args=None, score_qa=False) -> None:
        super(Blip, self).__init__()
        self.processor = BlipProcessor.from_pretrained(model_path)
        self.model = BlipForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16).to(device)

        for n, p in self.model.named_parameters():
            p.requires_grad = False

        self.mean = [
                0.48145466,
                0.4578275,
                0.40821073
            ]
        self.std = [
                0.26862954,
                0.26130258,
                0.27577711
            ]
        self.transforms = Compose([
            Resize(size=(384,384), interpolation=InterpolationMode.BICUBIC, antialias=True),
            Normalize(mean=self.mean, std=self.std),
        ])

        self.prompt = 'a photography of'
        self.prompt_length = len(self.processor.tokenizer(self.prompt).input_ids) - 1

        self.args = args

    def score(self, images, prompts, **kwargs):

        images = torch.stack([self.transforms(image) for image in images])

        text = [self.prompt + ' ' + prompt.lower() for prompt in prompts]
        inputs = self.processor(images=images, text=text, return_tensors="pt", padding='longest')
        device = images.device
        inputs = {key: inputs[key].to(device) for key in inputs.keys()}
        inputs['labels'] = inputs['input_ids'].masked_fill(
            inputs['input_ids'] == self.processor.tokenizer.pad_token_id, -100
        )
        inputs['labels'][:, :self.prompt_length] = -100

        with torch.autocast(device_type="cuda"):
            outputs = self.model(**inputs)
            reward = -outputs.loss
        return reward
    