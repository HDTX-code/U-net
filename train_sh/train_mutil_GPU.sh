conda activate homefun
cd homefun/zhf/U-net
CUDA_VISIBLE_DEVICES=0,6,7 python -m torch.distributed.launch --nproc_per_node=3 train_mutil_GPU.py