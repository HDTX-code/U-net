# source /Home/atr2/homefun/zhf/U-net/train_sh/train_mutil_GPU.sh
# source /devdata/home/homefun/U-net/train_sh/train_mutil_GPU.sh
conda activate homefun
# cd /devdata/home/homefun/U-net/
cd /Home/atr2/homefun/zhf/U-net
CUDA_VISIBLE_DEVICES=0,2 nohup \
python -m torch.distributed.launch --nproc_per_node=2 \
train_mutil_GPU.py  --Freeze_batch_size 512 \
                    --UnFreeze_batch_size 128 \
                    --train weights/all/train.txt \
                    --val weights/all/val.txt \
                    --cls_weights 0.5 0.5 \
                    --resume weights/loss_20220708114311/best_model_res50.pth \
                    --size 320 \
                    --Freeze_Epoch 0 \
                    --UnFreeze_Epoch 400 \
                    --Init_lr_Freeze 1e-4 \
                    --Init_lr_UnFreeze 1e-4 \
                    --Min_lr_Freeze 1e-5 \
                    --Min_lr_UnFreeze 1e-6 \
                    --amp \
                    --save_best \
                    --bilinear \
                    --loss_ce \
                    --loss_focal \
                    --loss_lv \
                    --Auto_Freeze \
                    --Auto_UnFreeze > weights/logres50_4.txt 2>&1 &