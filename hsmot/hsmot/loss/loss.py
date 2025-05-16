from typing import Callable, List, Optional, Tuple
import torch
Tensor = torch.Tensor
from hsmot.util.dist import l1_dist_rotate
import math
from hsmot.datasets.pipelines.channel import version_index_to_str
from mmcv.ops import diff_iou_rotated_2d

def l1_loss_rotate(
    input: Tensor,
    target: Tensor,
    angle_cycle: bool = False,
) -> Tensor:
    """
    计算输入张量和目标张量之间的旋转 L1 损失。

    Args:
        input (Tensor): 输入张量，形状为 [B, N, 5] 或 [N, 5]。
        target (Tensor): 目标张量，形状与 input 相同。

    Returns:
        Tensor: 旋转 L1 损失。
    """
    # 确保输入和目标的形状一致
    assert target.size() == input.size(), "input 和 target 的形状必须一致"
    
    # 检查张量是否包含批次维度
    if input.dim() == 3:  # [B, N, 5]
        batch_size = input.size(0)
        loss = l1_dist_rotate(input.flatten(0,1), target.flatten(0,1), aligned=True, cal_sum=False, angle_cycle=angle_cycle)  # 调用 l1_dist_rotate
        return loss.view(batch_size, -1)
    
    elif input.dim() == 2:  # [N, 5]
        loss = l1_dist_rotate(input, target, aligned=True, cal_sum=False, angle_cycle=angle_cycle)  # 扩展批次维度
        return loss
    
    else:
        raise ValueError("输入的张量必须是 2D 或 3D 张量")
    

def loss_rotated_iou_norm_bboxes1(bboxes1: torch.Tensor,
                                bboxes2: torch.Tensor,
                                img_shape: torch.Tensor,
                                version ='le135',) -> torch.Tensor:
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

    # bboxes1和bboxes2扩展一个Batch维度,并把结果的batch维度消掉
    ious = diff_iou_rotated_2d(bboxes1.unsqueeze(0), bboxes2.unsqueeze(0)).squeeze(0)
    # ious = diff_iou_rotated_2d(bboxes1, bboxes2)
    
    return ious
