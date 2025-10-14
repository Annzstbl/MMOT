import math

import numpy as np
import torch
import torch.nn as nn


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class ConvMSI(nn.Module):
    default_act = nn.ReLU()  # default activation

    def __init__(self, c1, c2, c3=8, k=(3, 7, 7), s=(1, 2, 2), p=(1,3,3), groups=None, use_bn_3d=True, use_gn_3d=False, final_bn=True, final_act=True):
        """
        Arguments:
            c1 (int): 输入通道数, 一般为 1
            c2 (int): 输出通道数（如 64)
            c3 (int): 光谱数
            k  (tuple): 3D 卷积核尺寸 (t, h, w)
            s  (tuple): 3D 卷积步长 (t_stride, h_stride, w_stride)
            groups (int): fuse depth-wise conv 的分组数，默认等于 c2
        """
        super().__init__()
        assert c1==1 and c3>1, 'c1 must be 1 and c3 > 1'
        # --- 首个 3D 卷积 + BN3d + SiLU ---
        self.conv3d = nn.Conv3d(c1, c2, kernel_size=k, stride=s,
                                padding=p, bias=False)
        D_out = math.floor((c3 + 2*p[0] - 1*(k[0] - 1) - 1)/s[0]) + 1
        if use_bn_3d:
            self.bn3d = nn.BatchNorm3d(c2)
        elif use_gn_3d:
            self.gn3d = nn.GroupNorm(16, c2)
        # --- 深度方向 fusion conv (depth-wise) ---
        self.fuse   = nn.Conv3d(c2, c2, kernel_size=(D_out,1,1),
                                groups=groups or c2, bias=False)
        # --- 后续 2D BN + 激活 ---
        if final_bn: 
            self.bn2d = nn.BatchNorm2d(c2)
        self.act = self.default_act

        self.final_act = final_act
        self.fianl_bn = final_bn
        self.use_bn_3d = use_bn_3d
        self.use_gn_3d = use_gn_3d
        # self.bn2d = nn.BatchNorm2d(c2)
        # self.act  = self.default_act

    def forward(self, x):
        """
            x: [B, c1, c3, H, W] or [B, c3, H, W] @ c1 = 1
        """
        if x.ndim == 4:
            x = x.unsqueeze(1)

        # 3D conv + BN + SiLU
        if self.use_bn_3d:
            x = self.act(self.bn3d(self.conv3d(x)))
        elif self.use_gn_3d:
            x = self.act(self.gn3d(self.conv3d(x)))
        else:
            x = self.act(self.conv3d(x))
        # depth-wise  fusion → [B, c2, 1, H', W']
        x = self.fuse(x)
        # squeeze 时序维度 → [B, c2, H', W']
        x = x.squeeze(2)
        # 最后 BN2d + SiLU
        if self.fianl_bn:
            x = self.bn2d(x)
            if self.final_act:
                x = self.act(x)
        return x

