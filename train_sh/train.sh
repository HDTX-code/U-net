# source /Home/atr2/homefun/zhf/U-net/train_sh/train.sh
# source /devdata/home/homefun/U-net/train_sh/train.sh
conda activate homefun
cd /devdata/home/homefun/U-net/
#cd /Home/atr2/homefun/zhf/U-net
nohup python train.py --GPU 0 \
                      --Freeze_batch_size 512 \
                      --UnFreeze_batch_size 128 \
                      --train weights/all/train.txt \
                      --val weights/all/val.txt \
                      --cls_weights 0.5 0.5 \
                      --resume weights/loss_20220708114311/best_model_res50.pth \
                      --size 320 \
                      --Freeze_Epoch 0 \
                      --UnFreeze_Epoch 320 \
                      --Init_lr_Freeze 1e-4 \
                      --Init_lr_UnFreeze 1e-3 \
                      --Min_lr_Freeze 1e-5 \
                      --Min_lr_UnFreeze 1e-5 \
                      --amp \
                      --save_best \
                      --bilinear \
                      --loss_ce \
                      --loss_focal \
                      --loss_lv > weights/logres50_4.txt 2>&1 &
#                      --Auto_Freeze \
#                      --Auto_UnFreeze
#                     --loss_iou \
#                     --pretrain_backbone \