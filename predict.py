import argparse
import math
import os
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
import cv2
from matplotlib import pyplot as plt

from torchvision import transforms

from torchvision.transforms import functional as F
from torchvision import transforms as T

from utils.utils import get_model, find_next, find_last


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def plt_show_Image_image(image: Image):
    plt.figure()
    plt.imshow(image)
    plt.show()


def Pre_pic(png):
    if not (png == 0).all():
        png = png * 5
        png[png > 255] = 255
        png = gamma_trans(png, math.log10(0.5) / math.log10(np.mean(png[png > 0]) / 255))
    image = Image.fromarray(cv2.cvtColor(png, cv2.COLOR_BGR2RGB)).convert('RGB')
    return image


def gamma_trans(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)


def main(args):
    # get devices
    torch.cuda.set_device(args.GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # model初始化
    model = get_model(model_name=args.model_name, num_classes=args.num_classes * 2,
                      pre="", pre_b=False, bilinear=args.bilinear)
    model.to(device)

    # load train weights
    assert os.path.exists(args.weights_path), "{} file dose not exist.".format(args.weights_path)
    model.load_state_dict(torch.load(args.weights_path, map_location='cpu')['model'])
    model.to(device)

    # using compute_mean_std.py
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # load image
    data_csv = pd.read_csv(args.csv_path)
    original_img = Pre_pic(cv2.imread(os.path.join(args.pic_path, 'train_pic', args.id + '.png')))
    img = cv2.imread(os.path.join(args.pic_path, 'train_pic', args.id + '.png'), cv2.IMREAD_GRAYSCALE)
    next_img = cv2.imread(os.path.join(args.pic_path, 'train_pic',
                                       find_next(data_csv,
                                                 data_csv[data_csv["id"] == args.id].index.tolist()[0]) + '.png'),
                          cv2.IMREAD_GRAYSCALE)
    last_img = cv2.imread(os.path.join(args.pic_path, 'train_pic',
                                       find_last(data_csv,
                                                 data_csv[data_csv["id"] == args.id].index.tolist()[0]) + '.png'),
                          cv2.IMREAD_GRAYSCALE)
    image = Pre_pic(np.stack([next_img, img, last_img], axis=2))
    original_size = (image.size[1], image.size[0])
    if os.path.exists(os.path.join(args.pic_path, 'label_pic', args.id + '.png')):
        gt = Image.fromarray(cv2.cvtColor(cv2.imread(os.path.join(args.pic_path,
                                                                  'label_pic', args.id + '.png')) * 255,
                                          cv2.COLOR_BGR2RGB))
        gt_img = Image.blend(original_img, gt, 0.5)
        # gt_img.show(title="./gt.jpg")
        plt_show_Image_image(gt_img)

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std),
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
            [prediction[0][[2 * item, 2 * item + 1], ...].argmax(0)
             for item in range(args.num_classes)], dim=0), original_size,
            interpolation=T.InterpolationMode.NEAREST).permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 255
        label_img = Image.fromarray(predictions)
        show_img = Image.blend(original_img, label_img, 0.5)
        # show_img.show(title="./pre.jpg")
        plt_show_Image_image(show_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict parameter setting')
    parser.add_argument('--model_name', type=str, default='res50')
    parser.add_argument('--GPU', type=int, default=2, help='GPU_ID')
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--weights_path', default='weights/loss_20220704234728/best_model_mit_PLD_b4.pth', type=str,
                        help='training weights')
    parser.add_argument('--pic_path', default=r"/Home/atr2/homefun/zhf/DATA/UW/", type=str, help='pic_path')
    parser.add_argument('--id', default=r'case88_day38_slice_0091', type=str, help='gt_path')
    parser.add_argument('--csv_path', type=str, default=r"/Home/atr2/homefun/zhf/DATA/UW/data_csv.csv")
    parser.add_argument('--size', type=int, default=224, help='pic size')
    parser.add_argument('--bilinear', default=True, action='store_true')
    args = parser.parse_args()

    main(args)
