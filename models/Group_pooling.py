from typing import List

import torch
from torch import nn, Tensor


class GroupPooling(nn.Module):
    def __init__(self, group_sizes = [20, 10, 5]):
        super().__init__()
        self.group_sizes = group_sizes

    def _group_pool(self, x: Tensor, group_size: int):
        """
        对单个尺度的特征进行分组平均池化
        Args:
            x: 输入特征 (B, L, C)
            group_size: 每组包含的token数
        Returns:
            pooled: 池化后特征 (B, num_groups, C)
        """
        B, L, C = x.shape
        
        # 计算分组数（向上取整）
        num_groups = (L + group_size - 1) // group_size
        
        pooled_list = []
        for g in range(num_groups):
            start = g * group_size
            end = min(start + group_size, L)
            group_feat = x[:, start:end, :]  # (B, actual_group_size, C)
            pooled = group_feat.mean(dim=1)  # (B, C)
            pooled_list.append(pooled)
        
        # 堆叠得到 (B, num_groups, C)
        pooled = torch.stack(pooled_list, dim=1)
        
        return pooled
    
    def forward(self, x: List[Tensor]):
        """
        Args:
            x: 三个尺度的特征列表
        Returns:
            fused: 融合后的特征 (B, total_groups, C)
        """
        pooled_features = []
        
        for i, feat in enumerate(x):
            group_size = self.group_sizes[i]
            # 分组池化: (B, L, C) -> (B, num_groups, C)
            pooled = self._group_pool(feat, group_size)
            pooled_features.append(pooled)
        
        # 沿序列维度拼接: (B, total_groups, C)
        fused = torch.cat(pooled_features, dim=1)
        
        return fused