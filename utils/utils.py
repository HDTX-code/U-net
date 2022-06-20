import math
from functools import partial

import torch
import utils.transforms as T
from net.res50_unet import Res50UNet

from net.unet import UNet
from net.vgg_unet import VGG16UNet
from utils.group_by_aspect_ratio import create_aspect_ratio_groups, GroupedBatchSampler


# ---------------------------------------------------#
#   获得高宽分类间隔抽取
# ---------------------------------------------------#
def get_dataloader_with_aspect_ratio_group(train_dataset, aspect_ratio_group_factor, batch_size, num_workers):
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    # 统计所有图像高宽比例在bins区间中的位置索引
    group_ids = create_aspect_ratio_groups(train_dataset, k=aspect_ratio_group_factor)
    # 每个batch图片从同一高宽比例区间中取
    train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, batch_size)

    # 如果按照图片高宽比采样图片，dataloader中需要使用batch_sampler
    gen = torch.utils.data.DataLoader(train_dataset,
                                      batch_sampler=train_batch_sampler,
                                      pin_memory=True,
                                      num_workers=num_workers,
                                      collate_fn=train_dataset.collate_fn)
    return gen


class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)

        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    base_size = 565
    crop_size = 480

    if train:
        return SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        return SegmentationPresetEval(mean=mean, std=std)


def create_model(num_classes, backbone, pretrained):
    if backbone == 'vgg':
        model = VGG16UNet(num_classes=num_classes, pretrain_backbone=pretrained)
    elif backbone == 'res50':
        model = Res50UNet(num_classes=num_classes, pretrain_backbone=pretrained)
    else:
        model = UNet(in_channels=3, num_classes=num_classes, base_c=32)
    return model


# ---------------------------------------------------#
#   lr 下降函数
# ---------------------------------------------------#
def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
    if iters <= warmup_total_iters:
        # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
        lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
    elif iters >= total_iters - no_aug_iter:
        lr = min_lr
    else:
        lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(
            math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
        )
    return lr


def step_lr(lr, decay_rate, step_size, iters):
    if step_size < 1:
        raise ValueError("step_size must above 1.")
    n = iters // step_size
    out_lr = lr * decay_rate ** n
    return out_lr


def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio=0.05, warmup_lr_ratio=0.1,
                     no_aug_iter_ratio=0.05, step_num=10):
    if lr_decay_type == "cos":
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)
    return func


def get_lr_fun(optimizer_type, batch_size, Init_lr, Min_lr, Epoch, lr_decay_type):
    # 判断当前batch_size，自适应调整学习率
    nbs = 16
    lr_limit_max = 1e-4 if optimizer_type == 'adam' else 5e-2
    lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    #   获得学习率下降的公式
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, Epoch)

    return lr_scheduler_func, Init_lr_fit, Min_lr_fit


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
