import colorsys
import math
from functools import partial

import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as F
import utils.transforms as T
from net.build_model import build
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
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), size=480):
        self.size = size
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        img = F.resize(img, (self.size, self.size))
        target = F.resize(target, (self.size, self.size),
                          interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        return self.transforms(img, target)


def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), crop_size=480, cb_per=1.2):
    base_size = crop_size * cb_per
    if train:
        return SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        return SegmentationPresetEval(mean=mean, std=std, size=crop_size)


def get_model(model_name, num_classes, pre="", pre_b=True, bilinear=True):
    model = build(model_name=model_name, num_classes=num_classes, pretrain_backbone=pre_b, bilinear=bilinear)
    if pre != "":
        model = load_model(model, pre)
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


def get_lr_fun(optimizer_type, batch_size, Init_lr, Min_lr, Epoch, lr_decay_type, Auto=False):
    # 判断当前batch_size，自适应调整学习率
    if Auto:
        nbs = 16
        lr_limit_max = 1e-4 if optimizer_type == 'adam' else 5e-2
        lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
    else:
        Init_lr_fit = Init_lr
        Min_lr_fit = Min_lr

    #   获得学习率下降的公式
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, Epoch)

    return lr_scheduler_func, Init_lr_fit, Min_lr_fit


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# ---------------------------------------------------#
#   画框设置不同的颜色
# ---------------------------------------------------#
def get_color(num_classes):
    if num_classes <= 21:
        colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                  (0, 128, 128),
                  (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128),
                  (192, 0, 128),
                  (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
                  (0, 64, 128),
                  (128, 64, 12)]
    else:
        hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    return colors


# ---------------------------------------------------#
#   加载model
# ---------------------------------------------------#
def load_model(model, model_path):
    print('Loading weights into state dict...')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location='cpu')['model']
    a = {}
    no_load = 0
    for k, v in pretrained_dict.items():
        try:
            if np.shape(model_dict[k]) == np.shape(v):
                a[k] = v
            else:
                no_load += 1
        except:
            print("模型加载出错")
            no_load = -1
            pass
    model_dict.update(a)
    model.load_state_dict(model_dict)
    print("No_load: {}".format(no_load))
    print('Finished!')
    return model


def find_next(data, item):
    day = int(data.loc[item, 'day_id'])
    slice = int(data.loc[item, 'slice_id'])
    case = int(data.loc[item, 'case_id'])
    if len(data.loc[
           data[(((data['day_id']) == day) & ((data['slice_id']) == (slice + 2)) &
                 ((data['case_id']) == case))].index.tolist(), :]) == 1:
        follow = data.loc[data[(((data['day_id']) == day) & ((data['slice_id']) == (slice + 2)) &
                                ((data['case_id']) == case))].index.tolist()[0], 'id']
    elif len(data.loc[
             data[(((data['day_id']) == day) & ((data['slice_id']) == (slice + 1)) &
                   ((data['case_id']) == case))].index.tolist(), :]) == 1:
        follow = data.loc[data[(((data['day_id']) == day) & ((data['slice_id']) == (slice + 1)) &
                                ((data['case_id']) == case))].index.tolist()[0], 'id']
    else:
        follow = data.loc[item, 'id']
    return follow


def find_last(data, item):
    day = data.loc[item, 'day_id']
    slice = data.loc[item, 'slice_id']
    case = data.loc[item, 'case_id']
    if len(data.loc[
           data[(((data['day_id']) == day) & ((data['slice_id']) == (slice - 2)) &
                 ((data['case_id']) == case))].index.tolist(), :]) == 1:
        last = data.loc[data[(((data['day_id']) == day) & ((data['slice_id']) == (slice - 2)) &
                              ((data['case_id']) == case))].index.tolist()[0], 'id']
    elif len(data.loc[
             data[(((data['day_id']) == day) & ((data['slice_id']) == (slice - 1)) &
                   ((data['case_id']) == case))].index.tolist(), :]) == 1:
        last = data.loc[data[(((data['day_id']) == day) & ((data['slice_id']) == (slice - 1)) &
                              ((data['case_id']) == case))].index.tolist()[0], 'id']
    else:
        last = data.loc[item, 'id']
    return last


if __name__ == '__main__':
    model = get_model("mit_PLD_b2", 21)
    print(model)
