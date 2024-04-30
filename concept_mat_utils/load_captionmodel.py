from concept_mat_utils.caption_blip import Blip

def load_model(reward_wrapper, caption_model, load_device, args):
    if 'Blip' in caption_model:
        model_path = 'Salesforce/blip-image-captioning-large'
        reward_wrapper.blip_model = Blip(model_path, load_device, args)
        reward_wrapper.caption_model_dict[str(caption_model.index('Blip'))] = reward_wrapper.blip_model
    else:
        raise NotImplementedError