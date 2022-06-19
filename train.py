import argparse
import os
import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset import UnetDataset
from utils.utils import get_dataloader_with_aspect_ratio_group, get_transform, create_model, get_lr_fun


def main(args):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                       训练相关准备                            #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M%S')
    log_dir = os.path.join(args.sd, "loss_" + str(time_str))
    # 检查保存文件夹是否存在
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                    训练参数设置相关准备                         #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    torch.cuda.set_device(args.GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # num_workers
    args.num_workers = min(min([os.cpu_count(), args.fbs if args.fbs > 1 else 0, 8]), args.num_workers)
    # 混合精度
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                 dataset dataloader model                    #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    with open(args.train, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(args.val, encoding='utf-8') as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    # using compute_mean_std.py
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # dataset
    train_dataset = UnetDataset(train_lines, train=True, transforms=get_transform(train=True, mean=mean, std=std))
    val_dataset = UnetDataset(val_lines, train=False, transforms=get_transform(train=False, mean=mean, std=std))
    # 是否按图片相似高宽比采样图片组成batch, 使用的话能够减小训练时所需GPU显存，默认使用
    if args.aspect_ratio_group_factor != -1:
        gen_Freeze = get_dataloader_with_aspect_ratio_group(train_dataset, args.aspect_ratio_group_factor,
                                                            args.Freeze_batch_size, args.num_workers)
        gen_UnFreeze = get_dataloader_with_aspect_ratio_group(train_dataset, args.aspect_ratio_group_factor,
                                                              args.UnFreeze_batch_size, args.num_workers)
    else:
        gen_Freeze = torch.utils.data.DataLoader(train_dataset,
                                                 batch_size=args.Freeze_batch_size,
                                                 shuffle=True,
                                                 pin_memory=True,
                                                 num_workers=args.num_workers,
                                                 collate_fn=train_dataset.collate_fn)
        gen_UnFreeze = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.UnFreeze_batch_size,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=args.num_workers,
                                                   collate_fn=train_dataset.collate_fn)
    gen_val = torch.utils.data.DataLoader(val_dataset,
                                          batch_size=1,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=args.num_workers,
                                          collate_fn=val_dataset.collate_fn)

    # model初始化
    model = create_model(num_classes=args.num_classes + 1)
    model.to(device)

    # 获取lr下降函数
    lr_scheduler_func_Freeze, Init_lr_fit_Freeze, Min_lr_fit_Freeze = get_lr_fun(args.optimizer_type_Freeze,
                                                                                 args.Freeze_batch_size,
                                                                                 args.Init_lr,
                                                                                 args.Init_lr*0.01,
                                                                                 args.Freeze_Epoch,
                                                                                 args.lr_decay_type_Freeze)
    lr_scheduler_func_UnFreeze, Init_lr_fit_UnFreeze, Min_lr_fit_UnFreeze = get_lr_fun(args.optimizer_type_UnFreeze,
                                                                                       args.UnFreeze_batch_size,
                                                                                       args.Init_lr,
                                                                                       args.Init_lr*0.01,
                                                                                       args.UnFreeze_Epoch,
                                                                                       args.lr_decay_type_UnFreeze)

    # 记录loss lr map
    train_loss = []
    learning_rate = []
    val_dice = []
