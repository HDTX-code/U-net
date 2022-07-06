# source /Home/atr2/homefun/zhf/U-net/train_sh/train_mutil_GPU.sh
# source /devdata/home/homefun/U-net/train_sh/train_mutil_GPU.sh
conda activate homefun
# cd /devdata/home/homefun/U-net/
cd /Home/atr2/homefun/zhf/U-net
CUDA_VISIBLE_DEVICES=0,2 nohup \
python -m torch.distributed.launch --nproc_per_node=2 \
train_mutil_GPU.py  --Freeze_batch_size 148 \
                    --UnFreeze_batch_size 28 \
                    --train weights/all/train.txt \
                    --val weights/all/val.txt \
                    --cls_weights 0.3 0.7 \
                    --resume weights/loss_20220705182252/best_model_res50.pth \
                    --Freeze_Epoch 0 \
                    --UnFreeze_Epoch 220 \
                    --amp \
                    --save_best \
                    --pretrain_backbone \
                    --bilinear >weights/logres50_2.txt 2>&1 &