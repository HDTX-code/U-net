conda activate homefun
cd homefun/zhf/U-net
nohup CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node=2 train_mutil_GPU.py >weights/log.out 2>&1 &