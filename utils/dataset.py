import math
import os

import cv2
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
        image_path_list = (line[2], line[1], line[0])
        image = np.stack([cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) for img_path in image_path_list], axis=2)
        label_path = line[3]
        img = self.Pre_pic(image)
        mask = Image.open(label_path).convert('RGB')

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def get_height_and_width(self, index):
        line = self.annotation_lines[index].split()
        h, w = int(line[-2]), int(line[-1])
        return h, w

    def Pre_pic(self, png):
        if not (png == 0).all():
            png = png * 5
            png[png > 255] = 255
            png = self.gamma_trans(png, math.log10(0.5) / math.log10(np.mean(png[png > 0]) / 255))
        image = Image.fromarray(cv2.cvtColor(png, cv2.COLOR_BGR2RGB)).convert('RGB')
        return image

    @staticmethod
    def gamma_trans(img, gamma):
        gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
        return cv2.LUT(img, gamma_table)

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
