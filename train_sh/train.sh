# source /Home/atr2/homefun/zhf/U-net/train_sh/train.sh
# source /devdata/home/homefun/U-net/train_sh/train.sh
conda activate homefun
# cd /devdata/home/homefun/U-net/
cd /Home/atr2/homefun/zhf/U-net
nohup python train.py --GPU 0 \
                      --Freeze_batch_size 912 \
                      --UnFreeze_batch_size 256 \
                      --train weights/all/train.txt \
                      --val weights/all/val.txt \
                      --cls_weights 0.3 0.7 \
                      --resume weights/loss_20220706125457/best_model_res50.pth \
                      --Freeze_Epoch 0 \
                      --UnFreeze_Epoch 320 \
                      --amp \
                      --save_best \
                      --bilinear > weights/logres50_3.txt 2>&1 &

#                       --pretrain_backbone \