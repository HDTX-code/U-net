from net.EfficientNet_unet import EfficientNetUNet
from net.res50_unet import Res50UNet


def build(num_classes, model_name: str = 'res50', pretrain_backbone: bool = True, bilinear: bool = False):
    if model_name == 'res50':
        model = Res50UNet(num_classes=num_classes, pretrain_backbone=pretrain_backbone, bilinear=bilinear)
        return model
    elif model_name == 'effb7':
        model = EfficientNetUNet(num_classes=num_classes, pretrain_backbone=pretrain_backbone, bilinear=bilinear)
        return model

