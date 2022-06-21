from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn
import torchvision.models
from torch import Tensor

# from .unet import Up, OutConv
from net.unet import Up, OutConv


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    """
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        # 重新构建backbone，将没有使用到的模块全部删掉
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class EfficientNetUNet(nn.Module):
    def __init__(self, num_classes, pretrain_backbone: bool = False, bilinear: bool = True):
        super(EfficientNetUNet, self).__init__()
        backbone = torchvision.models.efficientnet_b5(pretrained=pretrain_backbone)
        backbone = backbone.features
        self.stage_out_channels = [48, 40, 128, 304, 2048]
        self.backbone = IntermediateLayerGetter(backbone, return_layers={'0': '0',
                                                                         '2': '1',
                                                                         '4': '2',
                                                                         '6': '3',
                                                                         '8': '4'})
        # print(self.backbone(torch.ones([1, 3, 224, 224]))['5'].shape)
        # print(backbone)

        c = self.stage_out_channels[4] + self.stage_out_channels[3]
        self.up1 = Up(c, self.stage_out_channels[3], bilinear=bilinear)
        c = self.stage_out_channels[3] + self.stage_out_channels[2]
        self.up2 = Up(c, self.stage_out_channels[2], bilinear=bilinear)
        c = self.stage_out_channels[2] + self.stage_out_channels[1]
        self.up3 = Up(c, self.stage_out_channels[1], bilinear=bilinear)
        c = self.stage_out_channels[1] + self.stage_out_channels[0]
        self.up4 = Up(c, self.stage_out_channels[0], bilinear=bilinear)
        self.up5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = OutConv(self.stage_out_channels[0], num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        backbone_out = self.backbone(x)
        x = self.up1(backbone_out['4'], backbone_out['3'])
        x = self.up2(x, backbone_out['2'])
        x = self.up3(x, backbone_out['1'])
        x = self.up4(x, backbone_out['0'])
        x = self.up5(x)
        x = self.conv(x)

        return {"out": x}


# if __name__ == '__main__':
#     res = EfficientNetUNet(21)
#     print(res(torch.ones([1, 3, 224, 224]))['out'].shape)
