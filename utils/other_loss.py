import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


# 排除掉忽略的index
def deal_ignore(pred, target, ignore_index):
    index_ignore = torch.eq(target.reshape(-1), ignore_index)
    target_ = target.reshape(-1)[index_ignore]
    pred_ = pred.permute([0, 2, 3, 1]).reshape([-1, 2])[index_ignore, :]
    return pred_, target_


class SoftIoULoss(nn.Module):
    def __init__(self, n_classes, device, ignore_index):
        super(SoftIoULoss, self).__init__()
        self.n_classes = n_classes
        self.device = device
        self.ignore_index = ignore_index

    @staticmethod
    def to_one_hot(tensor, n_classes, device):
        n = tensor.size()
        # one_hot = nn.functional.one_hot(tensor).permute(0, 3, 1, 2).to(device)
        one_hot = torch.zeros([n, n_classes], device=device).scatter_(1, tensor.view(n, 1), 1)
        return one_hot

    def forward(self, input, target):
        # logit => N x Classes x H x W
        # target => N x H x W

        N = len(input)

        pred, label = deal_ignore(F.softmax(input, dim=1), target, self.ignore_index)

        target_onehot = self.to_one_hot(label, self.n_classes, self.device)

        # Numerator Product
        inter = pred * target_onehot
        # Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.n_classes, -1).sum(2)

        # Denominator
        union = pred + target_onehot - (pred * target_onehot)
        # Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.n_classes, -1).sum(2)

        loss = inter / (union + 1e-16)

        # Return average loss over classes and batch
        return 1 - loss.mean()


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge(input, labels, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, C, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    logits, _ = torch.max(F.softmax(input, dim=1), dim=1)
    loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels
