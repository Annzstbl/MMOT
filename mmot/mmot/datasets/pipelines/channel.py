'''
    其它各种形式和 mmrotate 格式转换
'''
import torch
import math
import mmcv
import cv2
from mmot.mmlab.hs_mmrotate import poly2obb, obb2poly
import numpy as np

class MotrToMmrotate:
    
    def __init():
        pass

    def __call__(self, results):
        return results

    
class MotipToMmrotate:
    def __init():
        pass

    def __call__(self, results):
        return results


def rotate_norm_angles_to_angles(angles, version='le135'):
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
    angles = angles * angle_range + angle_offset
    return angles


def rotate_boxes_to_norm_boxes(boxes, img_shape, version='le135'):
    '''
        计算从真实坐标到归一化坐标的转换
    '''
    h, w = img_shape
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
    norm_boxes = (boxes - torch.as_tensor([0,0,0,0,angle_offset], dtype=boxes.dtype, device=boxes.device)) /  torch.as_tensor([w, h, w, h, angle_range],dtype=boxes.dtype, device=boxes.device)
    return norm_boxes

def rotate_norm_boxes_to_boxes(norm_boxes, img_shape, version='le135'):
    '''
        计算从归一化坐标到真实坐标的转换
    '''
    h, w = img_shape
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
    boxes = norm_boxes * torch.as_tensor([w, h, w, h, angle_range],dtype=norm_boxes.dtype, device=norm_boxes.device) + torch.as_tensor([0,0,0,0,angle_offset],dtype=norm_boxes.dtype, device=norm_boxes.device)
    return boxes


def version_str_to_index(version:str)->int:
    if version == 'oc':
        return 1
    elif version == 'le90':
        return 2
    elif version == 'le135':
        return 3
    else:
        raise ValueError
    
def version_index_to_str(version:int)->str:
    if version == 1:
        return 'oc'
    elif version == 2:
        return 'le90'
    elif version == 3:
        return 'le135'
    else:
        raise ValueError
    
class MmrotateToMotr:
    def __init__(self, version='le135'):
        self.version = version

    def __call__(self, results_list):
        images = []
        targets = []
        img_metas = []

        if self.version == 'oc':
            raise NotImplementedError
        elif self.version == 'le135':
            angle_range = 1
            angle_offset = -1/4
        elif self.version == 'le90':
            angle_range = 1
            angle_offset = -1/2
        angle_range *= math.pi
        angle_offset *= math.pi

        for results in results_list:
            images.append(results['img'].data)

            img_metas.append({
                'img_shape':torch.as_tensor(results['img'].data.shape[1:],device=results['img'].data.device), 
                'version':torch.as_tensor(version_str_to_index(self.version), dtype=torch.int, device=results['img'].data.device) })
            # gt_bboxes-> norm_gt_bboxes
            h, w = results['img'].data.shape[1:]
            gt_bboxes = results['gt_bboxes'].data
            # norm_gt_bboxes = rotate_boxes_to_norm_boxes(gt_bboxes, (h, w), self.version) 函数版，未测试
            norm_gt_bboxes = (gt_bboxes - torch.as_tensor([0,0,0,0,angle_offset], dtype=gt_bboxes.dtype, device=gt_bboxes.device)) /  torch.as_tensor([w, h, w, h, angle_range],dtype=gt_bboxes.dtype, device=gt_bboxes.device)
            targets.append({'boxes':results['gt_bboxes'].data, 'norm_boxes':norm_gt_bboxes, 'labels':results['gt_labels'].data, 'obj_ids':results['gt_trackids'].data,
                            })
        return images, targets, img_metas

class MmrotateToMotrv2:
    def __init__(self, version='le135'):
        self.version = version

    def __call__(self, results_list):
        images = []
        targets = []
        img_metas = []

        if self.version == 'oc':
            raise NotImplementedError
        elif self.version == 'le135':
            angle_range = 1
            angle_offset = -1/4
        elif self.version == 'le90':
            angle_range = 1
            angle_offset = -1/2
        angle_range *= math.pi
        angle_offset *= math.pi

        for results in results_list:
            images.append(results['img'].data)

            img_metas.append({
                'img_shape':torch.as_tensor(results['img'].data.shape[1:],device=results['img'].data.device), 
                'version':torch.as_tensor(version_str_to_index(self.version), dtype=torch.int, device=results['img'].data.device) })
            # gt_bboxes-> norm_gt_bboxes
            h, w = results['img'].data.shape[1:]
            gt_bboxes = results['gt_bboxes'].data
            norm_gt_bboxes = (gt_bboxes - torch.as_tensor([0,0,0,0,angle_offset], dtype=gt_bboxes.dtype, device=gt_bboxes.device)) /  torch.as_tensor([w, h, w, h, angle_range],dtype=gt_bboxes.dtype, device=gt_bboxes.device)

            # process proposals
            proposals = results['proposals'].data
            proposal_scores = results['proposal_scores'].data
            norm_proposals = (proposals - torch.as_tensor([0,0,0,0,angle_offset], dtype=proposals.dtype, device=proposals.device)) /  torch.as_tensor([w, h, w, h, angle_range],dtype=proposals.dtype, device=proposals.device)

            targets.append({
                'boxes':results['gt_bboxes'].data, 
                'norm_boxes':norm_gt_bboxes, 
                'labels':results['gt_labels'].data, 
                'obj_ids':results['gt_trackids'].data,
                'proposals':results['proposals'].data,
                'norm_proposals':norm_proposals,
                'proposal_scores':proposal_scores,})
            
        return images, targets, img_metas
    
class MmrotateToMotip:
    def __init__(self, version='le135'):
        self.version = version

    def __call__(self, results_list):
        images = []
        targets = []
        img_metas = []

        if self.version == 'oc':
            raise NotImplementedError
        elif self.version == 'le135':
            angle_range = 1
            angle_offset = -1/4
        elif self.version == 'le90':
            angle_range = 1
            angle_offset = -1/2
        angle_range *= math.pi
        angle_offset *= math.pi

        for results in results_list:
            images.append(results['img'].data)

            img_metas.append({
                'img_shape':torch.as_tensor(results['img'].data.shape[1:],device=results['img'].data.device), 
                'version':torch.as_tensor(version_str_to_index(self.version), dtype=torch.int, device=results['img'].data.device),
                'transform_metas':results['img_metas']
                })
            # gt_bboxes-> norm_gt_bboxes
            h, w = results['img'].data.shape[1:]
            gt_bboxes = results['gt_bboxes'].data
            norm_gt_bboxes = (gt_bboxes - torch.as_tensor([0,0,0,0,angle_offset], dtype=gt_bboxes.dtype, device=gt_bboxes.device)) /  torch.as_tensor([w, h, w, h, angle_range],dtype=gt_bboxes.dtype, device=gt_bboxes.device)

            targets.append({'boxes':results['gt_bboxes'].data, 'norm_boxes':norm_gt_bboxes, 'labels':results['gt_labels'].data, 'obj_ids':results['gt_trackids'].data})

        return images, targets, img_metas



