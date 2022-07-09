import torch
from torch import nn
from utils.other_loss import SoftIoULoss, lovasz_hinge
import utils.distributed_utils as utils
from utils.dice_coefficient_loss import build_target, dice_loss


def Focal_Loss(inputs, target, cls_weights, num_classes=-100, alpha=0.5, gamma=2):
    logpt = -nn.functional.cross_entropy(inputs, target, ignore_index=num_classes, weight=cls_weights)
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    loss = loss.mean()
    return loss


def criterion(inputs, target, IoULoss, loss_weight=None, num_classes: int = 2, dice: bool = True,
              ignore_index: int = -100):
    loss_ce = 0
    loss_focal = 0
    loss_dice = 0
    loss_IoU = 0
    loss_lv = 0

    for name, x in inputs.items():
        # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
        for item in range(target.shape[-1]):
            loss_ce += nn.functional.cross_entropy(x[:, [2 * item, 2 * item + 1], ...], target[..., item],
                                                   ignore_index=ignore_index, weight=loss_weight)
            loss_focal += Focal_Loss(x[:, [2 * item, 2 * item + 1]], target[..., item],
                                     cls_weights=loss_weight, num_classes=ignore_index)
            if dice is True:
                dice_target = build_target(target[..., item], 2, ignore_index)
                loss_dice += dice_loss(x[:, [2 * item, 2 * item + 1], ...], dice_target,
                                       multiclass=False, ignore_index=ignore_index)
            loss_IoU += IoULoss(x[:, [2 * item, 2 * item + 1], ...], target[..., item])
            loss_lv += lovasz_hinge(x[:, [2 * item, 2 * item + 1], ...], target[..., item])
    return loss_ce, loss_focal, loss_dice, loss_IoU, loss_lv


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(2)
    dice = utils.DiceCoefficient(num_classes=2, ignore_index=255)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output['out']

            for item in range(target.shape[-1]):
                confmat.update(target[..., item].flatten(),
                               output[:, [2 * item, 2 * item + 1], ...].argmax(1).flatten())
                dice.update(output[:, [2 * item, 2 * item + 1]], target[..., item])

        confmat.reduce_from_all_processes()
        dice.reduce_from_all_processes()

    return confmat, dice.value.item()


def train_one_epoch(model, optimizer, data_loader, device, epoch, num_classes, cls_weights,
                    print_freq=10, scaler=None, CE=True, FOCAL=False):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    IoULoss = SoftIoULoss(2, device, ignore_index=255).to(device)

    with torch.no_grad():
        cls_weights = torch.from_numpy(cls_weights).type(torch.FloatTensor).to(device)

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss_ce, loss_focal, loss_dice, loss_IoU, loss_lv = criterion(output, target, IoULoss,
                                                                          num_classes=num_classes,
                                                                          ignore_index=255,
                                                                          loss_weight=cls_weights)
            loss = loss_dice + loss_IoU + loss_lv
            if CE:
                loss += loss_ce
            if FOCAL:
                loss += loss_focal

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr,
                             loss_ce=loss_ce.item(),
                             loss_focal=loss_focal.item(),
                             loss_dice=loss_dice.item(),
                             loss_lv=loss_lv.item(),
                             loss_IoU=loss_IoU.item())

    return metric_logger.meters["loss"].global_avg, lr
