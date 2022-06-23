import argparse
import os
import random

import pandas as pd


def main(args):
    print(args)
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    data_csv = pd.read_csv(args.csv_path)
    data_csv = data_csv.loc[data_csv[~((data_csv['segmentation_s'] == "0") & (data_csv['segmentation_sb'] == "0") &
                                       (data_csv['segmentation_lb'] == "0"))].index.tolist(), :]
    data_csv.index = list(range(len(data_csv)))
    num_list = random.sample(range(len(data_csv)), int(args.num_per * len(data_csv)))
    num = len(num_list)
    train = random.sample(num_list, int(num * args.train_per))
    f_train = open(os.path.join(args.save_path, 'train.txt'), 'w')
    f_val = open(os.path.join(args.save_path, 'val.txt'), 'w')
    for item in num_list:
        if item in train:
            f_train.write(os.path.join(args.data_path, 'train_pic', data_csv.loc[item, 'id']) + '.png' + ' '
                          + os.path.join(args.data_path, 'label_pic', data_csv.loc[item, 'id']) + '.png' + ' '
                          + str(data_csv.loc[item, 'slice_h']) + ' '
                          + str(data_csv.loc[item, 'slice_w']) + '\n')
        else:
            f_val.write(os.path.join(args.data_path, 'train_pic', data_csv.loc[item, 'id']) + '.png' + ' '
                        + os.path.join(args.data_path, 'label_pic', data_csv.loc[item, 'id']) + '.png' + ' '
                        + str(data_csv.loc[item, 'slice_h']) + ' '
                        + str(data_csv.loc[item, 'slice_w']) + '\n')
    f_train.close()
    f_val.close()
    print('train: {}'.format(len(train)) + '\n' + "val: {}".format(num - len(train)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make train and val')
    parser.add_argument('--data_path', type=str, default=r"/Home/atr2/homefun/zhf/DATA/UW/")
    parser.add_argument('--save_path', type=str, default=r"./weights")
    parser.add_argument('--csv_path', type=str, default=r"/Home/atr2/homefun/zhf/U-net/weights/data_csv.csv")
    parser.add_argument('--num_per', type=float, default=1, help='The proportion of photos used')
    parser.add_argument('--train_per', type=float, default=0.9, help='The proportion of photos train')
    args = parser.parse_args()

    main(args)
