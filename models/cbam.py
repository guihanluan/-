# CBAM 注意力模块：通道注意力 + 空间注意力，轻量即插即用
# 位置：models/cbam.py，后续队友检测和优化只看这一个文件即可
# 参考：CBAM: Convolutional Block Attention Module (ECCV 2018)
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """通道注意力：让模型关注更重要的通道。"""

    def __init__(self, channels: int, ratio: int = 16):
        super().__init__()
        # 通道数较小时自动缩小压缩比，避免除零或通道为0
        if channels < ratio:
            ratio = max(1, channels // 2)
        hidden = max(1, channels // ratio)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        # 两层全连接，压缩再恢复，参数量小
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 平均池化与最大池化互补，相加后生成通道权重
        w = self.fc(self.avg(x)) + self.fc(self.max(x))
        return x * self.sigmoid(w)


class SpatialAttention(nn.Module):
    """空间注意力：让模型关注病斑所在位置，抑制背景。"""

    def __init__(self, k: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通道维度的平均与最大，拼接后学空间权重
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        w = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * w


class CBAM(nn.Module):
    """CBAM 整体：先通道后空间，插入 Backbone 尾部即可。"""

    def __init__(self, channels: int, ratio: int = 16, k: int = 7):
        super().__init__()
        self.ca = ChannelAttention(channels, ratio)
        self.sa = SpatialAttention(k)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x
