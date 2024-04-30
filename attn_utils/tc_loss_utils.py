'''
Modified from
https://github.com/mlpc-ucsd/TokenCompose/blob/main/train/src/loss_utils.py

'''
import torch
import torch.nn as nn
from torchvision import transforms
from copy import deepcopy
import pdb

SD14_TO_SD21_RATIO = 1.5

# get token index in text
def get_word_idx(text: str, tgt_word, tokenizer):

    tgt_word = tgt_word.lower()

    # ignore the first and last token
    encoded_text = tokenizer.encode(text)[1:-1]
    encoded_tgt_word = tokenizer.encode(tgt_word)[1:-1]

    # find the idx of target word in text
    first_token_idx = -1
    for i in range(len(encoded_text)):
        if encoded_text[i] == encoded_tgt_word[0]:

            if len(encoded_text) > 0:
                # check the following 
                following_match = True
                for j in range(1, len(encoded_tgt_word)):
                    if encoded_text[i + j] != encoded_tgt_word[j]:
                        following_match = False
                if not following_match:
                    continue
            # for a single encoded idx, just take it
            first_token_idx = i

            break

    assert first_token_idx != -1, "word not in text"

    # add 1 for sot token
    tgt_word_tokens_idx_ls = [i + 1 + first_token_idx for i in range(len(encoded_tgt_word))]

    # sanity check
    encoded_text = tokenizer.encode(text)

    decoded_token_ls = []

    for word_idx in tgt_word_tokens_idx_ls:
        text_decode = tokenizer.decode([encoded_text[word_idx]]).strip("#")
        decoded_token_ls.append(text_decode)

    decoded_tgt_word = "".join(decoded_token_ls)
    
    tgt_word_ls = tgt_word.split(" ")
    striped_tgt_word = "".join(tgt_word_ls).strip("#")

    assert decoded_tgt_word == striped_tgt_word, "decode_text != striped_tar_wd"

    return tgt_word_tokens_idx_ls

# get attn loss by resolution

def get_grounding_loss_by_layer(_gt_seg_list, word_token_idx_ls, res, 
                                input_attn_map_ls, is_training_sd21):
    """
        _gt_seg_list (List[Tensor]): (1, 1, res, res)
        input_attn_map_ls (List[Tensor]): (b*head, res, res, 77), sum = b*head*res*res
        
    """
    if is_training_sd21:
        # training with sd21, using resolution 768 = 512 * 1.5
        res = int(SD14_TO_SD21_RATIO * res)
    
    if len(word_token_idx_ls) == 0:
        return {
            "token_loss" : 0,
            "pixel_loss": 0,
        }

    gt_seg_list = deepcopy(_gt_seg_list)

    # reszie gt seg map to the same size with attn map
    # assert gt_seg_list[0].min() == 0 and gt_seg_list[0].max() == 1

    resize_transform = transforms.Resize((res, res), antialias=True)

    valid_gt_seg = 0
    for i in range(len(gt_seg_list)):
        gt_seg_list[i] = resize_transform(gt_seg_list[i])
        gt_seg_list[i] = gt_seg_list[i].squeeze(0) # 1, 1, res, res => 1, res, res
        # add binary
        gt_seg_list[i] = (gt_seg_list[i] > 0.0).float()
        # if there is truly a mask
        if gt_seg_list[i].sum() > 0.1:
            valid_gt_seg += 1
    

    ################### token loss start ###################
    # Following code is adapted from
    # https://github.com/silent-chen/layout-guidance/blob/08b687470f911c7f57937012bdf55194836d693e/utils.py#L27
    token_loss = 0.0
    for attn_map in input_attn_map_ls:
        b, H, W, j = attn_map.shape
        for i in range(len(word_token_idx_ls)): # [[word1 token_idx1, word1 token_idx2, ...], [word2 token_idx1, word2 token_idx2, ...]]
            obj_loss = 0.0
            single_word_idx_ls = word_token_idx_ls[i] #[token_idx1, token_idx2, ...]
            mask = gt_seg_list[i]

            for obj_position in single_word_idx_ls:
                # ca map obj shape 8 * 16 * 16
                try:
                    ca_map_obj = attn_map[:, :, :, obj_position].reshape(b, H, W)
                except:
                    print(_gt_seg_list, word_token_idx_ls, res, input_attn_map_ls, attn_map.shape)

                # calculate pixel value inside mask
                activation_value = (ca_map_obj * mask).reshape(b, -1).sum(dim=-1)/ca_map_obj.reshape(b, -1).sum(dim=-1)
                obj_loss += (1.0 - torch.mean(activation_value)) ** 2

            token_loss += (obj_loss/len(single_word_idx_ls))

    token_loss = token_loss / len(word_token_idx_ls) 

    ################## token loss end ##########################

    ################## pixel loss start ######################
    # average cross attention map on different layers
    avg_attn_map_ls = []
    for i in range(len(input_attn_map_ls)):
        avg_attn_map_ls.append(
            input_attn_map_ls[i].reshape(-1, res, res, input_attn_map_ls[i].shape[-1]).mean(0)
        ) # avg on each head
    avg_attn_map = torch.stack(avg_attn_map_ls, dim=0)
    avg_attn_map = avg_attn_map.sum(0) / avg_attn_map.shape[0]
    avg_attn_map = avg_attn_map.unsqueeze(0) # (1, res, res, 77)

    bce_loss_func = nn.BCELoss()
    pixel_loss = 0.0

    for i in range(len(word_token_idx_ls)):
        
        word_cross_attn_ls = []
        for token_idx in word_token_idx_ls[i]:
            word_cross_attn_ls.append(
                avg_attn_map[..., token_idx]
            )

        word_cross_attn_ls = torch.stack(word_cross_attn_ls, dim=0).sum(dim=0) # this should be sum, tokens of one object should belong to one meta class 
        try:
            pixel_loss += bce_loss_func(
                word_cross_attn_ls, 
                gt_seg_list[i]
            )
        except:
            print(word_cross_attn_ls.shape, word_cross_attn_ls.device)
            print(gt_seg_list[i].shape, gt_seg_list[i].device)
            pdb.set_trace()
            print(gt_seg_list)
        # contrastive loss

    
    # average with len word_token_idx_ls
    # pixel_loss = pixel_loss  / valid_gt_seg 
    pixel_loss = pixel_loss / len(word_token_idx_ls)
    ################## pixel loss end #########################

    return {
        "token_loss" : token_loss,
        "pixel_loss": pixel_loss,
    }
