import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class UnetDataset(Dataset):
    def __init__(self, annotation_lines, train=True, transforms=None):
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.train = train
        self.transforms = transforms

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        line = self.annotation_lines[index].split()
        image_path = line[0]
        label_path = line[1]
        img = Image.open(image_path).convert('RGB')
        mask = Image.open(label_path).convert('RGB')

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def get_height_and_width(self, index):
        line = self.annotation_lines[index].split()
        h, w = int(line[2]), int(line[3])
        return h, w

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs
