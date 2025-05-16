'''
    其它各种形式和 mmrotate 格式转换
'''
import torch
import math
import mmcv
import cv2
from hsmot.mmlab.hs_mmrotate import poly2obb, obb2poly
import numpy as np

class MotrToMmrotate:
    
    def __init():
        pass

    def __call__(self, results):
        return results
    # def prepare_solo(self, img, target):
    #     result = dict()

    #     # 保存原有
    #     result['motr_format'] = (img, target)

    #     ann = {}
    #     ann['bboxes'] = target['boxes']
    #     ann['labels'] = target['labels']
    #     # ann['polygons'] = targets['polygons'] 暂时不支持
    #     # result = 

    #     # load img
    #     result['filename'] = 'unsupport'
    #     result['img'] = img
    #     result['img_fields'] = ['img']
    #     result['img_shape'] = img.shape
    #     result['ori_shape'] = img.shape

    #     result['ann_info'] = ann
    #     return result

    # def __call__(self, imgs_targets):
    #     imgs, targets = imgs_targets
    #     results = []
    #     for img, target in zip(imgs, targets):
    #         results.append(self.prepare_solo(img, target))
    #     return results
    
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
            # norm_gt_bboxes = rotate_boxes_to_norm_boxes(gt_bboxes, (h, w), self.version) 函数版，未测试
            norm_gt_bboxes = (gt_bboxes - torch.as_tensor([0,0,0,0,angle_offset], dtype=gt_bboxes.dtype, device=gt_bboxes.device)) /  torch.as_tensor([w, h, w, h, angle_range],dtype=gt_bboxes.dtype, device=gt_bboxes.device)
            # 构造heatmap
            heatmap = HeatmapFromRotateGt.heatmap_from_rotate_gt_xywha(gt_bboxes, (h, w), self.version)

            targets.append({'boxes':results['gt_bboxes'].data, 'norm_boxes':norm_gt_bboxes, 'labels':results['gt_labels'].data, 'obj_ids':results['gt_trackids'].data, 'heatmap':heatmap})

        return images, targets, img_metas


# 处理生成heatmap
class HeatmapFromRotateGt:
    @staticmethod
    def heatmap_from_rotate_gt_xyxyxyxy(gt_xyxyxyxy, img_shape, version='le135'):
        '''
            gt_xyxyxyxy: [N, 8] (x, y, w, h, a)
            img_shape: (h, w)
        '''
        gt_xywha = poly2obb(gt_xyxyxyxy, version)
        return HeatmapFromRotateGt.heatmap_from_rotate_gt(gt_xywha, gt_xyxyxyxy, img_shape)

    @staticmethod
    def heatmap_from_rotate_gt_xywha(gt_bboxes, img_shape, version='le135'):
        '''
            gt_bboxes: [N, 5] (x, y, w, h, a)
            img_shape: (h, w)
        '''
        gt_xyxyxyxy = obb2poly(gt_bboxes, version)
        return HeatmapFromRotateGt.heatmap_from_rotate_gt(gt_bboxes, gt_xyxyxyxy, img_shape, version)

    def heatmap_from_rotate_gt(gt_xywha, gt_xyxyxyxy, img_shape, version='le135'):
        
        height, width = img_shape

        bound_size = 200
        heatmap = torch.zeros((height + bound_size* 2 , width + bound_size * 2), dtype=torch.float32)

        for xywha, xyxyxyxy in zip(gt_xywha, gt_xyxyxyxy):
 
            gaussian_2d, x_lt, y_lt = HeatmapFromRotateGt.generate_2d_gaussian(heatmap.shape, xywha, xyxyxyxy)
            w_start = y_lt+bound_size
            w_end = y_lt+bound_size+gaussian_2d.shape[0]
            h_start = x_lt + bound_size
            h_end = x_lt + bound_size + gaussian_2d.shape[1]
            assert w_start>0 and w_end<heatmap.shape[0] and h_start>0 and h_end<heatmap.shape[1], f'w_start:{w_start}, w_end:{w_end}, h_start:{h_start}, h_end:{h_end}, heatmap.shape:{heatmap.shape}, xywha:{xywha}, xyxyxyxy:{xyxyxyxy}'
            heatmap[w_start: w_end, h_start:h_end] += gaussian_2d

        return heatmap[bound_size:bound_size+height, bound_size:bound_size+width]

    @staticmethod 
    def generate_2d_gaussian(shape, box_xywha, box_xyxyxyxy, gaussian_max_one=False):


        # 计算外接矩形
        x1, y1, x2, y2, x3, y3, x4, y4 = box_xyxyxyxy
        x_lt = min(x1, x2, x3, x4)
        y_lt = min(y1, y2, y3, y4)
        x_rb = max(x1, x2, x3, x4)
        y_rb = max(y1, y2, y3, y4)
        
        rect_w = int(x_rb - x_lt)
        rect_h = int(y_rb - y_lt)

        obb_w = int(box_xywha[2])
        obb_h = int(box_xywha[3])

        sigma_w = obb_w / 6
        sigma_h = obb_h / 6

        gauss_w = cv2.getGaussianKernel(int(rect_w), sigma_w, cv2.CV_32F)
        gauss_h = cv2.getGaussianKernel(int(rect_h), sigma_h, cv2.CV_32F)
        gaussian = np.outer(gauss_h, gauss_w)

        # 旋转
        angle = box_xywha[4]
        M = cv2.getRotationMatrix2D((int(rect_w/2), int(rect_h/2)), -float(angle)*180/math.pi, 1.0)
        gaussian = cv2.warpAffine(gaussian, M, (int(rect_w), int(rect_h)))

        if gaussian_max_one:
            gaussian /= gaussian.max()

        return gaussian, int(x_lt), int(y_lt)


if __name__ == '__main__':
    import numpy as np
    import torch
    img_path = '/data/users/wangying01/lth/data/hsmot/rgb/data36-7/000009.png'
    mot_path = '/data/users/wangying01/lth/data/hsmot/mot/data36-7.txt'

    img = mmcv.imread(img_path)
    xyxyxyxy = []
    with open(mot_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split(',')
        frame_id, obj_id, x1, y1, x2, y2, x3, y3, x4, y4, _, cls, _ = line
        if int(frame_id) == 9:
            xyxyxyxy.append([int(x1), int(y1), int(x2), int(y2), int(x3), int(y3), int(x4), int(y4)])
    xyxyxyxy = np.array(xyxyxyxy)
    xyxyxyxy = torch.from_numpy(xyxyxyxy).float()
    xywha = poly2obb(xyxyxyxy, 'le135')

    # heatmap = HeatmapFromRotateGt.heatmap_from_rotate_gt_xyxyxyxy(xyxyxyxy, img.shape[:2], 'le135')
    heatmap = HeatmapFromRotateGt.heatmap_from_rotate_gt_xywha(xywha, img.shape[:2], 'le135')

    import matplotlib.pyplot as plt
    # 叠加显示Heatmap和img
    plt.imshow(heatmap, cmap='hot')
    plt.imshow(img, alpha=0.5)
    plt.savefig('/data/users/wangying01/lth/hsmot/MOTIP/heatmap_img.png')

    plt.close()

    plt.imshow(heatmap, cmap='hot')
    plt.savefig('/data/users/wangying01/lth/hsmot/MOTIP/heatmap.png')

    import time
    time_start = time.time()
    #  测试速度
    for i in range(100):
        heatmap = HeatmapFromRotateGt.heatmap_from_rotate_gt_xyxyxyxy(xyxyxyxy, img.shape[:2], 'le135')
    time_end = time.time()
    print('time cost', time_end-time_start, 's')



