from attr_concen_utils.gsam_interface import GsamSegModel

def load_seg_model(args, device, train_layer_ls):
    if 'gsam' in args.seg_model:
        seg_model = GsamSegModel(args, device, train_layer_ls)
    else:
        raise NotImplementedError

    return seg_model
