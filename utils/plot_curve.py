import datetime
import os.path

import matplotlib.pyplot as plt


def plot_loss_and_lr(train_loss, learning_rate, log_dir):
    try:
        x = list(range(len(train_loss)))
        fig, ax1 = plt.subplots(1, 1)
        ax1.plot(x, train_loss, 'r', label='loss')
        ax1.set_xlabel("step")
        ax1.set_ylabel("loss")
        ax1.set_title("Train Loss and lr")
        plt.legend(loc='best')

        ax2 = ax1.twinx()
        ax2.plot(x, learning_rate, label='lr')
        ax2.set_ylabel("learning rate")
        ax2.set_xlim(0, len(train_loss))  # 设置横坐标整数间隔
        plt.legend(loc='best')

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(handles1 + handles2, labels1 + labels2, loc='upper right')

        fig.subplots_adjust(right=0.8)  # 防止出现保存图片显示不全的情况
        fig.savefig(os.path.join(log_dir, './loss_and_lr.png'))
        plt.close()
        print("successful save loss curve! ")
    except Exception as e:
        print(e)


def plot_dice(dice, log_dir):
    try:
        x = list(range(len(dice)))
        plt.plot(x, dice, label='dice')
        plt.xlabel('epoch')
        plt.ylabel('dice')
        plt.title('Eval dice')
        plt.xlim(0, len(dice))
        plt.legend(loc='best')
        plt.savefig(os.path.join(log_dir, './dice.png'))
        plt.close()
        print("successful save dice curve!")
    except Exception as e:
        print(e)
