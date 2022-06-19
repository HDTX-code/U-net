import argparse
import os
import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.utils import get_dataloader_with_aspect_ratio_group


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
                                          num_workers=num_workers,
                                          collate_fn=val_dataset.collate_fn)

    # model初始化
    model = get_model(backbone, num_classes + 1, model_path=model_path, pretrained=pretrained).to(device)
