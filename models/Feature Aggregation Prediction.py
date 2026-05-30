import torch
import torch.nn as nn

class Feature_Aggregation_Prediction(nn.Module):
    """
    特征聚合预测模块（Feature Aggregation Prediction Module）
    Args:
        in_features: 输入特征维度 (C)
        hidden_features: MLP隐藏层维度
        out_features: 输出维度
        drop: Dropout率
    """
    def __init__(self, in_features, hidden_features, out_features, drop=0.):
        super().__init__()
        
        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_features, out_features)
        )

    def forward(self, x):
        """
        Args:
            x: 分组池化后的特征 (B, total_groups, C)
        Returns:
            out: 预测值 (B, out_features)
        """
        # 全局平均池化: (B, total_groups, C) -> (B, C, 1) -> (B, C)
        x = x.permute(0, 2, 1)  # (B, C, total_groups)
        x = self.gap(x).squeeze(-1)  # (B, C)
        
        # MLP: (B, C) -> (B, out_features)
        out = self.mlp(x)
        
        return out