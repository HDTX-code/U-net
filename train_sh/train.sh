conda activate homefun
cd homefun/zhf/U-net
nohup python train.py --backbone 'eff_b7' --GPU 1 --Freeze_Epoch 0 --UnFreeze_Epoch 160 --UnFreeze_batch_size 36 --size 256 --resume weights/loss_20220622155335/best_model_eff_b7.pth --bilinear --cls_weights 0.1 0.9 >weights/log100.txt 2>&1 &