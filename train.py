import argparse
import os
import datetime
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset import UnetDataset
from utils.plot_curve import plot_loss_and_lr, plot_dice
from utils.train_one_epoch import train_one_epoch, evaluate
from utils.utils import get_dataloader_with_aspect_ratio_group, get_transform, create_model, get_lr_fun, \
    set_optimizer_lr


def main(args):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                       训练相关准备                            #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M%S')
    log_dir = os.path.join(args.save_dir, "loss_" + str(time_str))
    # 检查保存文件夹是否存在
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 用来保存训练以及验证过程中信息
    results_file = os.path.join(log_dir, "results.txt")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                    训练参数设置相关准备                         #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    torch.cuda.set_device(args.GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # num_workers
    args.num_workers = min(min([os.cpu_count(), args.UnFreeze_batch_size if args.UnFreeze_batch_size > 1 else 0, 8]),
                           args.num_workers)
    # 混合精度
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # 权重
    if args.cls_weights is None:
        args.cls_weights = np.ones([2], np.float32)
    else:
        args.cls_weights = np.array(args.cls_weights, np.float32)
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
    train_dataset = UnetDataset(train_lines, train=True, transforms=get_transform(train=True, mean=mean,
                                                                                  std=std, crop_size=args.size))
    val_dataset = UnetDataset(val_lines, train=False, transforms=get_transform(train=False, mean=mean,
                                                                               std=std, crop_size=args.size))
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
    model = create_model(num_classes=args.num_classes + 1, backbone=args.backbone, pretrained=args.pretrained)
    model.to(device)

    # 获取lr下降函数
    lr_scheduler_func_Freeze, Init_lr_fit_Freeze, Min_lr_fit_Freeze = get_lr_fun(args.optimizer_type_Freeze,
                                                                                 args.Freeze_batch_size,
                                                                                 args.Init_lr,
                                                                                 args.Init_lr * 0.01,
                                                                                 args.Freeze_Epoch,
                                                                                 args.lr_decay_type_Freeze)
    lr_scheduler_func_UnFreeze, Init_lr_fit_UnFreeze, Min_lr_fit_UnFreeze = get_lr_fun(args.optimizer_type_UnFreeze,
                                                                                       args.UnFreeze_batch_size,
                                                                                       args.Init_lr,
                                                                                       args.Init_lr * 0.01,
                                                                                       args.UnFreeze_Epoch,
                                                                                       args.lr_decay_type_UnFreeze)

    # 记录loss lr map
    train_loss = []
    learning_rate = []
    val_dice = []

    best_dice = 0.
    start_time = time.time()

    print(args)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #  first frozen backbone train 5 epochs                       #
    #  首先冻结前置特征提取网络权重（backbone），训练rpn以及最终预测网络部分 #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # 冻结训练
    if args.Freeze_Epoch != 0 and args.resume == '':
        for param in model.backbone.parameters():
            param.requires_grad = False
        params = [p for p in model.parameters() if p.requires_grad]

        #   根据optimizer_type选择优化器
        optimizer = {
            'adam': optim.Adam(params, Init_lr_fit_Freeze, betas=(args.momentum, 0.999), weight_decay=0),
            'sgd': optim.SGD(params, Init_lr_fit_Freeze, momentum=args.momentum, nesterov=True,
                             weight_decay=args.weight_decay)
        }[args.optimizer_type_Freeze]

        for epoch in range(args.Init_Epoch + 1, args.Freeze_Epoch + 1):
            set_optimizer_lr(optimizer, lr_scheduler_func_Freeze, epoch - 1)
            mean_loss, lr = train_one_epoch(model, optimizer, gen_Freeze, device, epoch, args.num_classes + 1,
                                            print_freq=int((num_train / args.Freeze_batch_size) // 5), scaler=scaler,
                                            cls_weights=args.cls_weights)
            confmat, dice = evaluate(model, gen_val, device=device, num_classes=args.num_classes + 1)
            val_info = str(confmat)
            train_loss.append(mean_loss)
            learning_rate.append(lr)
            val_dice.append(dice)
            print(val_info)
            print(f"dice coefficient: {dice:.3f}")
            # write into txt
            with open(results_file, "a") as f:
                # 记录每个epoch对应的train_loss、lr以及验证集各指标
                train_info = f"[epoch: {epoch}]\n" \
                             f"train_loss: {mean_loss:.4f}\n" \
                             f"lr: {lr:.6f}\n" \
                             f"dice coefficient: {dice:.3f}\n"
                f.write(train_info + val_info + "\n\n")

            save_file = {"model": model.state_dict()}
            if args.amp:
                save_file["scaler"] = scaler.state_dict()

            if args.save_best is True:
                if best_dice < val_dice[-1]:
                    torch.save(save_file, "save_weights/best_model.pth")
                    best_dice = val_dice[-1]
                    print('save best dice {}'.format(val_dice[-1]))
            else:
                torch.save(save_file, "save_weights/dice_{}.pth".format(dice))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #  second unfrozen backbone and train all network     #
    # 解冻前置特征提取网络权重（backbone），接着训练整个网络权重   #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    for param in model.backbone.parameters():
        param.requires_grad = True
    params = [p for p in model.parameters() if p.requires_grad]

    #   根据optimizer_type选择优化器
    optimizer = {
        'adam': optim.Adam(params, Init_lr_fit_UnFreeze, betas=(args.momentum, 0.999), weight_decay=0),
        'sgd': optim.SGD(params, Init_lr_fit_UnFreeze, momentum=args.momentum,
                         nesterov=True, weight_decay=args.weight_decay)
    }[args.optimizer_type_UnFreeze]

    if args.resume != '':
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.Init_Epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    UnFreeze_start_Epoch = args.Init_Epoch + args.Freeze_Epoch if args.resume else args.Freeze_Epoch + 1

    for epoch in range(UnFreeze_start_Epoch, args.UnFreeze_Epoch + args.Freeze_Epoch + 1):
        set_optimizer_lr(optimizer, lr_scheduler_func_UnFreeze, epoch - args.Freeze_Epoch)
        mean_loss, lr = train_one_epoch(model, optimizer, gen_UnFreeze, device, epoch, args.num_classes + 1,
                                        print_freq=int((num_train / args.UnFreeze_batch_size) // 5), scaler=scaler,
                                        cls_weights=args.cls_weights)
        confmat, dice = evaluate(model, gen_val, device=device, num_classes=args.num_classes + 1)
        val_info = str(confmat)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)
        val_dice.append(dice)
        print(val_info)
        print(f"dice coefficient: {dice:.3f}")
        # write into txt
        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n" \
                         f"dice coefficient: {dice:.3f}\n"
            f.write(train_info + val_info + "\n\n")

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        if args.save_best is True:
            if best_dice < val_dice[-1]:
                torch.save(save_file, "save_weights/best_model.pth")
                best_dice = val_dice[-1]
                print('save best dice {}'.format(val_dice[-1]))
        else:
            torch.save(save_file, "save_weights/dice_{}.pth".format(dice))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))
    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        plot_loss_and_lr(train_loss, learning_rate, log_dir)
    if len(val_dice) != 0:
        plot_dice(val_dice, log_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training parameter setting')
    parser.add_argument('--backbone', type=str, default='res50')
    parser.add_argument('--save_dir', type=str, default="./weights")
    parser.add_argument('--resume', type=str, default="", help='resume')
    parser.add_argument('--GPU', type=int, default=6, help='GPU_ID')
    parser.add_argument('--size', type=int, default=256, help='pic size')
    parser.add_argument('--train', type=str, default=r"weights/val.txt", help="train_txt_path")
    parser.add_argument('--val', type=str, default=r"weights/val.txt", help="val_txt_path")
    parser.add_argument('--optimizer_type_Freeze', type=str, default='adam', help='adam or sgd')
    parser.add_argument('--optimizer_type_UnFreeze', type=str, default='adam', help='adam or sgd')
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--Freeze_batch_size', type=int, default=12)
    parser.add_argument('--UnFreeze_batch_size', type=int, default=24)
    parser.add_argument('--aspect_ratio_group_factor', type=int, default=3)
    parser.add_argument('--lr_decay_type_Freeze', type=str, default='cos', help="'step' or 'cos'")
    parser.add_argument('--lr_decay_type_UnFreeze', type=str, default='cos', help="'step' or 'cos'")
    parser.add_argument('--num_workers', type=int, default=24, help="num_workers")
    parser.add_argument('--Init_lr', type=float, default=1e-4, help="max lr")
    parser.add_argument('--momentum', type=float, default=0.9, help="momentum")
    parser.add_argument('--weight_decay', type=float, default=0, help="adam is 0")
    parser.add_argument('--Freeze_Epoch', type=int, default=12, help="Freeze_Epoch")
    parser.add_argument('--UnFreeze_Epoch', type=int, default=24, help="UnFreeze_Epoch")
    parser.add_argument('--Init_Epoch', type=int, default=0, help="Init_Epoch")
    parser.add_argument('--pretrained', default=False, action='store_true', help="pretrained")
    parser.add_argument('--amp', default=True, action='store_true', help="amp or Not")
    parser.add_argument('--cls_weights', nargs='+', type=float, default=None, help='交叉熵loss系数')
    args = parser.parse_args()

    main(args)
