from typing import Callable, List, Optional, Tuple
import torch
# 不可微
from mmcv.ops import box_iou_rotated, diff_iou_rotated_2d

import math
from hsmot.datasets.pipelines.channel import version_index_to_str



def loss_rotated_iou_norm_bboxes1(bboxes1: torch.Tensor,
                                bboxes2: torch.Tensor,
                                img_shape: torch.Tensor,
                                version ='le135',) -> torch.Tensor:
    assert True, "Should use hsmot.loss.loss_rotated_iou_norm_bboxes1 instead of hsmot.util.loss.loss_rotated_iou_norm_bboxes1"
    
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

    ious = diff_iou_rotated_2d(bboxes1, bboxes2)
    return ious
