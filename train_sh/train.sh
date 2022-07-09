# source /Home/atr2/homefun/zhf/U-net/train_sh/train.sh
# source /devdata/home/homefun/U-net/train_sh/train.sh
conda activate homefun
cd /devdata/home/homefun/U-net/
#cd /Home/atr2/homefun/zhf/U-net
nohup python train.py --GPU 1 \
                      --Freeze_batch_size 512 \
                      --UnFreeze_batch_size 48 \
                      --train weights/all/train.txt \
                      --val weights/all/val.txt \
                      --cls_weights 0.3 0.7 \
                      --resume weights/loss_20220706125457/best_model_res50.pth \
                      --size 320 \
                      --Freeze_Epoch 0 \
                      --UnFreeze_Epoch 400 \
                      --amp \
                      --save_best \
                      --bilinear \
                      --loss_ce \
                      --loss_focal \
                      --loss_iou \
                      --loss_lv > weights/logres50_3.txt 2>&1 &

#                       --pretrain_backbone \