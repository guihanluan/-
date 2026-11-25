# ECA 注意力模块：无降维通道注意力，参数极少，作极轻量对照组
# 参考：ECA-Net: Efficient Channel Attention (CVPR 2020)
import math

import torch.nn as nn


class ECA(nn.Module):
    """ECA 模块：一维卷积学生成通道权重，核大小按通道数自适应，不降维。"""

    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        # 自适应核大小，通道越多看的邻域越大
        t = int(abs((math.log2(max(channels, 2)) + b) / gamma))
        k = t if t % 2 else t + 1
        k = max(k, 3)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # B,C,1,1 -> B,1,C -> 一维卷积 -> B,1,C -> B,C,1,1
        y = self.avg(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)
