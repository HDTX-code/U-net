# source /Home/atr2/homefun/zhf/U-net/train_sh/train.sh
# source /devdata/home/homefun/U-net/train_sh/train.sh
conda activate homefun
# cd /devdata/home/homefun/U-net/
cd /Home/atr2/homefun/zhf/U-net
nohup python train.py --GPU 2 \
                      --Freeze_batch_size 148 \
                      --UnFreeze_batch_size 28 \
                      --train weights/all/train.txt \
                      --val weights/all/val.txt \
                      --cls_weights 0.3 0.7 \
                      --amp \
                      --save_best \
                      --pretrain_backbone \
                      --bilinear > weights/logres50_1.txt 2>&1 &