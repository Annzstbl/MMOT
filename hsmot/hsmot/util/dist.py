


from typing import Callable, List, Optional, Tuple
import torch
# 不可微
from mmcv.ops import box_iou_rotated, diff_iou_rotated_2d

import math
from hsmot.datasets.pipelines.channel import version_index_to_str

Tensor = torch.Tensor

def l1_dist_rotate(x1: Tensor, x2: Tensor, aligned=False, cal_sum=True, angle_cycle=False):
    r"""
        计算 x1 和 x2 的旋转 L1 距离，角度使用周期形式计算。

        Args:
        x1 (Tensor): 输入张量，形状为 [N, 5]。
        x2 (Tensor): 输入张量，形状为 [M, 5]。

        角度取值范围[0,1]

        aligned (bool): True时要求x1 x2形状相同, False时返回所有组合结果
        
        Returns:
        Tensor: 每行的旋转 L1 距离，形状为 [N]。 或[N, M]
    """
    # 检查输入形状
    n, ch = x1.shape
    n2, ch2 = x2.shape
    assert ch == ch2 == 5, "输入的最后一维必须为5。"

    if aligned:
        # 当 aligned=True 时，要求 x1 和 x2 的形状相同
        assert x1.shape == x2.shape, "当 aligned=True 时，x1 和 x2 的形状必须一致。"
        x1 = x1.view(-1, 5)  # 展平为 [N, 5]
        x2 = x2.view(-1, 5)  # 展平为 [N, 5]
    else:
        # 如果未对齐，则需要生成所有可能的组合
        x1 = x1.unsqueeze(1)  # [1, N, 5]
        x2 = x2.unsqueeze(0)  # [M, 1, 5]

        x1 = x1.expand(-1, x2.shape[1], -1)  # [M, N, 5]
        x2 = x2.expand(x1.shape[0], -1, -1)  # [M, N, 5]
        x1 = x1.contiguous().view(-1, 5)  # 展平为 [M * N, 5]
        x2 = x2.contiguous().view(-1, 5)  # 展平为 [M * N, 5]

    # 计算前四列的绝对差
    diff_abs = torch.abs(x1[:, :4] - x2[:, :4])  # [N, 4] 或 [M * N, 4]

    # 计算第五列的旋转周期距离
    angle_diff = torch.abs(x1[:, 4] - x2[:, 4])  # [N] 或 [M * N]
    if angle_cycle:
        angle_diff_min = torch.min(angle_diff, 1 - angle_diff)  # 处理周期性角度差

        # 合并结果
        distance = torch.cat([diff_abs, angle_diff_min.unsqueeze(1)], dim=1)  # [N, 5] 或 [M * N, 5]
    else:
        # 合并结果
        distance = torch.cat([diff_abs, angle_diff.unsqueeze(1)], dim=1)

    if not cal_sum:
        if aligned:
            result = distance
        else:
            result = distance.view(n, n2, -1)
        return result
    # 对行求和
    row_sum = distance.sum(dim=1)  # [N] 或 [M * N]

    if aligned:
        # 当 aligned=True，还原为 [N]
        result = row_sum
    else:
        # 当 aligned=False，还原为 [M, N]
        result = row_sum.view(n, n2)

    return result


def box_iou_rotated_norm_bboxes1(bboxes1: torch.Tensor,
                                 bboxes2: torch.Tensor,
                                 img_shape: torch.Tensor,
                                 version ='le135',
                                 mode: str='iou',
                                 aligned: bool=False,
                                 clockwise: bool=True) -> torch.Tensor:
    '''
        计算iou
        bbox1是归一化的值
        bbox2是真实值
    '''
    # angle_range = 0.5 if version=='oc' else 1
    if type(version) != str:
        version = version_index_to_str(version)
    if version == 'oc':
        raise NotImplementedError
    elif version == 'le135':
        angle_range = 1
        angle_offset = -1/4
    elif version == 'le90':
        angle_range = 1
        angle_offset = -1/2
    angle_range *= math.pi
    angle_offset *= math.pi
    h, w = img_shape
    bboxes1 = bboxes1 * torch.as_tensor([w, h, w, h, angle_range],dtype=bboxes1.dtype, device=bboxes1.device) + torch.as_tensor([0,0,0,0,angle_offset],dtype=bboxes1.dtype, device=bboxes1.device)

    ious = box_iou_rotated(bboxes1, bboxes2, mode, aligned, clockwise)
    # ious = diff_iou_rotated_2d(bboxes1, bboxes2)
    return ious
