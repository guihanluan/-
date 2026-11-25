# SE 注意力模块：只做通道加权，不做空间加权，比 CBAM 更轻，专救小目标被压问题
# 参考：Squeeze-and-Excitation Networks (CVPR 2018)
import torch.nn as nn


class SE(nn.Module):
    """SE 模块：全局池化 -> 压缩 -> 恢复 -> 通道加权，不改变形状。"""

    def __init__(self, channels: int, ratio: int = 16):
        super().__init__()
        # 小通道保护，避免压缩到 0
        if channels < ratio:
            ratio = max(1, channels // 2)
        hidden = max(1, channels // ratio)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # 只学通道权重，不碰空间，浅层小目标特征不被空间压缩吃掉
        return x * self.fc(self.avg(x))
