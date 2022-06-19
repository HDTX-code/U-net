import torch

from utils.group_by_aspect_ratio import create_aspect_ratio_groups, GroupedBatchSampler


# ---------------------------------------------------#
#   获得高宽分类间隔抽取
# ---------------------------------------------------#
def get_dataloader_with_aspect_ratio_group(train_dataset, aspect_ratio_group_factor, batch_size, num_workers):
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    # 统计所有图像高宽比例在bins区间中的位置索引
    group_ids = create_aspect_ratio_groups(train_dataset, k=aspect_ratio_group_factor)
    # 每个batch图片从同一高宽比例区间中取
    train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, batch_size)

    # 如果按照图片高宽比采样图片，dataloader中需要使用batch_sampler
    gen = torch.utils.data.DataLoader(train_dataset,
                                      batch_sampler=train_batch_sampler,
                                      pin_memory=True,
                                      num_workers=num_workers,
                                      collate_fn=train_dataset.collate_fn)
    return gen
