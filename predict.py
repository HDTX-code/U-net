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
    model = create_model(num_classes=args.num_classes + 1,
                         backbone=args.backbone, pretrained=False, bilinear=args.bilinear)
    model.to(device)

    # load train weights
    assert os.path.exists(args.weights_path), "{} file dose not exist.".format(args.weights_path)
    model.load_state_dict(torch.load(args.weights_path, map_location='cpu')['model'])
    model.to(device)

    # using compute_mean_std.py
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # load image
    original_img = Image.open(args.pic_path).convert("RGB")
    original_size = (original_img.size[1], original_img.size[0])
    if args.gt_path != "":
        gt_img = Image.blend(original_img,
                             Image.fromarray(cv2.cvtColor(cv2.imread(args.gt_path) * 255, cv2.COLOR_BGR2RGB)), 0.5)
        gt_img.save("./gt.jpg")

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=mean[0], std=std[0]),
                                         transforms.Resize((args.size, args.size))
                                         ])
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
        prediction = model(img.to(device))['out']
        t_end = time_synchronized()
        print("inference time: {}".format(t_end - t_start))

        predictions = F.resize(torch.stack(
            [prediction[0][[0, item + 1], ...].argmax(0)
             for item in range(args.num_classes)], dim=0), original_size,
            interpolation=T.InterpolationMode.NEAREST).permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 255
        label_img = Image.fromarray(predictions)
        show_img = Image.blend(original_img, label_img, 0.5)
        show_img.save("./pre.jpg")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict parameter setting')
    parser.add_argument('--backbone', type=str, default='eff_b7')
    parser.add_argument('--GPU', type=int, default=0, help='GPU_ID')
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--weights_path', default='weights/loss_20220623130417/best_model_eff_b7.pth', type=str,
                        help='training weights')
    parser.add_argument('--pic_path',
                        default=r'/Home/atr2/homefun/zhf/DATA/UW/train_pic/case101_day20_slice_0107.png',
                        type=str, help='pic_path')
    parser.add_argument('--gt_path',
                        default=r'/Home/atr2/homefun/zhf/DATA/UW/label_pic/case101_day20_slice_0107.png',
                        type=str, help='gt_path')
    parser.add_argument('--size', type=int, default=256, help='pic size')
    parser.add_argument('--bilinear', default=True, action='store_true', help="bilinear or conv")
    args = parser.parse_args()

    main(args)
