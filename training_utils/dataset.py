import os
import torch
from torchvision import transforms
from datasets import load_dataset
from PIL import Image

from training_utils.gan_dataset import Gan_Dataset


def get_dataset_dataloader(args, accelerator):
    if args.gan_loss:
        dataset = Gan_Dataset(args)
    elif args.training_prompts.endswith("txt"):
        dataset = load_dataset("text", data_files=dict(train=args.training_prompts))
    elif args.training_prompts.endswith("json"):
        dataset = load_dataset("json", data_files=dict(train=args.training_prompts))

    image_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.ToTensor()
        ]
    )

    def preprocess_train(instance):
        if 'file_name' in instance:
            filenames = instance.pop('file_name')
            images = [Image.open(os.path.join(args.image_folder, filename)).convert("RGB") for filename in filenames]
            images = [image_transforms(image) for image in images]
            instance['image'] = torch.stack(images)
        return instance
            

    with accelerator.main_process_first():
        if args.gan_loss:
            train_dataset = dataset
        else:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed + accelerator.process_index)
            if args.max_train_samples is not None:
                dataset["train"] = dataset["train"].select(range(args.max_train_samples))
            # Set the training transforms
            train_dataset = dataset["train"].with_transform(preprocess_train)


    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    return train_dataset, train_dataloader