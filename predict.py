import argparse
import os
import time

import numpy as np
import torch
from PIL import Image
import cv2

from torchvision import transforms
from utils.utils import create_model
from torchvision.transforms import functional as F
from torchvision import transforms as T


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main(args):
    # get devices
    # torch.cuda.set_device(args.GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # model初始化
    model = create_model(num_classes=args.num_classes + 1, backbone=args.backbone, pretrained=False)
    model.to(device)

    # load train weights
    assert os.path.exists(args.weights_path), "{} file dose not exist.".format(args.weights_path)
    model.load_state_dict(torch.load(args.weights_path, map_location='cpu')['model'])
    model.to(device)

    # using compute_mean_std.py
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # load image
    original_img = Image.open(args.pic_path)
    original_size = (original_img.size[1], original_img.size[0])
    if args.gt_path != "":
        gt_img = Image.blend(original_img,
                             Image.fromarray(cv2.cvtColor(cv2.imread(args.gt_path) * 255, cv2.COLOR_BGR2RGB)), 0.5)
        gt_img.show("ground truth")
        print(gt_img.size)
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std),
                                         transforms.Resize((480, 480))])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        predictions = model(img.to(device))['out']
        t_end = time_synchronized()
        print("inference time: {}".format(t_end - t_start))

        predictions = F.resize(torch.stack(
            [predictions[0, [0, item + 1], ...].argmax(0)
             for item in range(args.num_classes)], dim=0), original_size,
            interpolation=T.InterpolationMode.NEAREST).permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 255
        label_img = Image.fromarray(predictions)
        show_img = Image.blend(original_img, label_img, 0.5)
        show_img.show("predict")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict parameter setting')
    parser.add_argument('--backbone', type=str, default='res50')
    parser.add_argument('--GPU', type=int, default=0, help='GPU_ID')
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--weights_path', default='weights/loss_20220620181720/best_model.pth', type=str,
                        help='training weights')
    parser.add_argument('--pic_path',
                        default=r'D:\work\project\DATA\Kaggle-uw\train_pic/case130_day20_slice_0030.png',
                        type=str, help='pic_path')
    parser.add_argument('--gt_path',
                        default=r'D:\work\project\DATA\Kaggle-uw\label_pic/case130_day20_slice_0030.png',
                        type=str, help='gt_path')
    args = parser.parse_args()

    main(args)
