from ultralytics import YOLO
from collections import defaultdict

from PIL import Image
import numpy as np

import torch
import torch.nn as nn
from seg_model.gsam.EfficientSAM.FastSAM.tools import *
from seg_model.gsam.GroundingDINO.groundingdino.util.inference import load_model, predict
from torchvision.ops import box_convert
from attn_utils.tc_loss_utils import get_grounding_loss_by_layer
import groundingdino.datasets.transforms as T

class GsamSegModel(nn.Module):
    def __init__(self, args, device, train_layer_ls=['mid_8', 'up_16', 'up_32', 'up_64']) -> None:
        super().__init__()

        self.device = device

        self.train_layer_ls = train_layer_ls
        self.args = args
        # load sam model
        sam_model_path = 'pretrained_model/FastSAM-x.pt'
        self.sam = YOLO(sam_model_path)
        self.sam.model.to(device)
        for p in self.sam.model.parameters():
            p.requires_grad = False

        # load gdino
        groundingdino_config = "seg_model/gsam/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        groundingdino_ckpt_path = "pretrained_model/groundingdino_swint_ogc.pth"

        self.gdino = load_model(groundingdino_config, groundingdino_ckpt_path, device='cpu')
        self.gdino.to(device)
        for n, p in self.gdino.named_parameters():
            p.requires_grad = False

        self.gdino_transform_np = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )  

        self.gdino_transform_tensor = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    
    @torch.no_grad()
    def get_mask(self, image, nouns):
        width = image.shape[1]
        height = image.shape[2]
        # use np_image
        image=image.mul(255).clamp(0,255).to(torch.uint8)  
        np_image = image.cpu().permute(1,2,0).numpy()
        # use tensor image
        # tensor_image = image.unsqueeze(0)

        seg_results = self.sam(
            np_image,
            # tensor_image,
            imgsz=(width, height),
            device=image.device,
            retina_masks=True, # draw high-resolution segmentation masks
            iou=0.9, # iou threshold for filtering the annotations
            conf=0.4, # object confidence threshold
            max_det=100,
            verbose=False
        )

        # sometimes it will have no mask
        if seg_results[0].masks is None:
            print("No mask is detected")
            return None

        
        # use np_image
        pil_image = Image.fromarray(np_image.astype(np.uint8)).convert("RGB")
        image_transformed, _ = self.gdino_transform_np(pil_image, None)
        image_transformed = image_transformed.to(image.device)

        # use tensor image
        # image_transformed, _ = self.gdino_transform_tensor(tensor_image, None)

        caption = ' . '.join(nouns)

        boxes, logits, phrases = predict(
            model=self.gdino,
            image=image_transformed.squeeze(),
            caption=caption,
            box_threshold=0.3,
            text_threshold=0.25,
            # device=image.device,
            device=image.device
        )

        ori_h = width # due to square image, this is fine
        ori_w = height

        # Save each frame due to the post process from FastSAM
        boxes = boxes * torch.Tensor([ori_w, ori_h, ori_w, ori_h])

        boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy().tolist()
        
        noun_box_dict = defaultdict(list)
        for box_idx in range(len(boxes)):
            try:
                noun_idx = nouns.index(phrases[box_idx].strip())
            except: # the box does not belong to single noun, just pass
                # print(f"not detected: {phrases[box_idx]}; nouns: {nouns}")
                continue

            mask, _ = box_prompt(
                seg_results[0].masks.data,
                boxes[box_idx],
                ori_h,
                ori_w,
                to_numpy=False,
            )
            noun_box_dict[str(noun_idx)].append(mask.squeeze())

        mask_list = []
        for i in range(len(nouns)):
            if str(i) in noun_box_dict: # if the object is detected
                noun_mask = torch.sum(torch.stack(noun_box_dict[str(i)]), dim=0).squeeze()
                noun_mask = noun_mask > 0 # make this to be a [0,1] mask
            else: # no object is detected
                noun_mask = image.new_zeros((width, height)) > 0

            mask_list.append(noun_mask.unsqueeze(0).unsqueeze(0))

        return mask_list

    
    def get_mask_loss(self, images, prompt, all_subtree_indices, attn_map_idx_to_wp_all, attn_map):
        images = images.detach().requires_grad_(False)

        token_loss = images.new_zeros(()) # could not directly be 0, int has no detach
        pixel_loss = images.new_zeros(())
        grounding_loss_dict = defaultdict(int)

        # re-organize the attn map, split each batch
        bs = images.shape[0]
        attn_map_per_sample = []
        for i in range(bs):
            attn_map_per_timestep = {}

            for timestep in attn_map:
                attn_map_per_timestep[timestep] = {}
                for place in attn_map[timestep]:
                    attn_map_per_timestep[timestep][place] = []
                    for inst in attn_map[timestep][place]:
                        inst = inst.reshape(bs, inst.shape[0]//bs, inst.shape[1], inst.shape[2], inst.shape[3])
                        attn_map_per_timestep[timestep][place].append(inst[i])

            attn_map_per_sample.append(attn_map_per_timestep)

        for idx, subtree_indices in enumerate(all_subtree_indices):
            attn_map_idx_to_wp = attn_map_idx_to_wp_all[idx]
            image = images[idx]

            nouns = []
            attributes = []
            for subtree in subtree_indices:
                if len(subtree) < 1: # should also collect single object
                    continue
                noun_indices = subtree[-1] if isinstance(subtree[-1], list) else [subtree[-1]] 
                noun_char = [attn_map_idx_to_wp[i] for i in noun_indices]
                noun = ''.join(noun_char)
                nouns.append(noun)
                # attribute: [[17, 18], 19] -> [17, 18, 19]
                attribute = []
                for attribute_char in subtree[:-1]:
                    if isinstance(attribute_char, list):
                        attribute.extend(attribute_char)
                    else:
                        attribute.append(attribute_char)
                # also add nouns into mask loss
                attribute.extend(noun_indices)

                attributes.append(attribute)
            
            if len(nouns) == 0:
                continue
            
            # nouns: [noun1, noun2, noun3]
            # attributes: [[1,2,3], [4,5], [18,19]]
            nouns, attributes = self.update_nouns_attributes(nouns, attributes)

            if len(nouns) == 0:
                continue
            
            with torch.autocast(device_type='cuda'):
                mask = self.get_mask(image, nouns)

            if mask == None:
                continue 

            for timestep in attn_map_per_sample[idx]:
                for train_layer in self.train_layer_ls:
                    layer_res = int(train_layer.split("_")[1])
                    attn_loss_dict = \
                        get_grounding_loss_by_layer(
                        _gt_seg_list=mask,
                        word_token_idx_ls=attributes,
                        res=layer_res,
                        input_attn_map_ls=attn_map_per_sample[idx][timestep][train_layer],
                        is_training_sd21=False,
                    )

                    layer_token_loss = attn_loss_dict["token_loss"]
                    layer_pixel_loss = attn_loss_dict["pixel_loss"]

                    grounding_loss_dict[f"token/{timestep}/{train_layer}"] += layer_token_loss
                    grounding_loss_dict[f"pixel/{timestep}/{train_layer}"] += layer_pixel_loss

                    token_loss += layer_token_loss
                    pixel_loss += layer_pixel_loss
        
        token_loss = token_loss / len(all_subtree_indices)
        pixel_loss = pixel_loss / len(all_subtree_indices)
            
        return token_loss, pixel_loss, grounding_loss_dict

    
    # rm not qualified nouns
    def update_nouns_attributes(self, nouns, attributes):
        new_nouns, new_attributes = [], []
        # rm duplicate nouns, do not calculate loss if there is duplicate nouns
        nouns2idx = defaultdict(list)
        for idx, n in enumerate(nouns):
            nouns2idx[n].append(idx)
        for n in nouns2idx:
            if len(nouns2idx[n]) > 1:
                continue
            else:
                new_nouns.append(n)
                new_attributes.append(attributes[nouns2idx[n][0]])
        
        # rm invalid nouns
        filtered_nouns, filtered_attributes = [], []
        invalid_nouns = set(['scene', 'surface', 'area', 'atmosphere', 'noise', 'place', 'kitchen', 'dream', 'interior', 'exterior', 
        'meal', 'background', 'bathroom', 'room', 'scent', 'street', 'hillside', 'mountain', 'sky', 'sea', 'ocean', 'lost',
        'language', 'skill', 'one', 'night', 'day', 'morning', 'space', 'environment', 'conditions', 'field', 'shore', 'restroom',
        'party', 'grass', 'snow', 'meadow', 'water', 'shadow', 'waves', 'song', 'cycle', 'sunlight', 'mysteries', 'wall', 'salon',
        'range', 'cry', 'speech', 'tone', 'thing', 'about', 'activity', 'air', 'advertisement', 'airport', 'also'])
        
        for idx, n in enumerate(new_nouns):
            if n in invalid_nouns or n[:-1] in invalid_nouns:
                continue
            else:
                filtered_nouns.append(n)
                filtered_attributes.append(new_attributes[idx])

        
        return filtered_nouns, filtered_attributes
