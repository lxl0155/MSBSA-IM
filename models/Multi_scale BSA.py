import torch
import torch.nn.functional as F
from timm.layers import DropPath
from typing import List
from torch import nn, Tensor

from model_build import build_norm, build_act

class Mlp(nn.Module):
    def __init__(self, in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, dim,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, context=None):
        """
        Args:
            x (Tensor):: input features with shape of (num_windows*B, N, C)
        """
        B_, N, C = x.shape

        # 生成Q、K、V
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 每个形状: (B, num_heads, N, head_dim)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        
        attn = (k @ q.transpose(-2, -1))  
        attn = attn * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn) # [B_, num_heads, C//num_heads, C//num_heads]

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x 


class ChannelTransformerBlock(nn.Module):
    def __init__(self, dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.attn_C = ChannelAttention(
            dim, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            
        self.drop_path: nn.Module = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):

        x = x + self.drop_path(self.attn_C(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class ChannelTransformerLayer_BSA(nn.Module):
    def __init__(self, 
                 dim,
                 depth,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 downsample=None):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads

        if isinstance(norm_layer, str):
            norm_layer = build_norm(norm_layer)
        if isinstance(act_layer, str):
            act_layer = build_act(act_layer)

        # build blocks(修改)
        self.multi_blocks = nn.ModuleList([
            nn.ModuleList([
                ChannelTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, 
                    qk_scale=qk_scale,
                    drop=drop_rate, 
                    attn_drop=attn_drop,
                    drop_path=drop_path[j] if isinstance(drop_path, list) else drop_path,
                    act_layer=act_layer,
                    norm_layer=norm_layer
                ) for j in range(depth)
            ]) for _ in range(3)  # 多个独立的blocks
        ])

    def forward(self, x: List[Tensor]): 
        # x[0]——[B, 328, 512],x[1]——[B, 164, 512],x[2]——[B, 82, 512]
        
        for i in range(3):
            for blk in self.multi_blocks[i]:
                x[i] = blk(x[i])
   
        return x
