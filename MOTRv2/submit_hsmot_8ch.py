# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from copy import deepcopy
import json

import os
import argparse
import torchvision.transforms.functional as F
import torch
import cv2
from tqdm import tqdm
from pathlib import Path
from models import build_model
from util.tool import load_model
from main import get_args_parser

from models.structures import Instances
from torch.utils.data import Dataset, DataLoader
from hsmot.mmlab.hs_mmrotate import poly2obb, obb2poly
import math
import numpy as np
import random
from hsmot.datasets.pipelines.channel import rotate_boxes_to_norm_boxes, rotate_norm_boxes_to_boxes


COLORS_10 = [(144, 238, 144), (178, 34, 34), (221, 160, 221), (0, 255, 0), (0, 128, 0), (210, 105, 30), (220, 20, 60),
             (192, 192, 192), (255, 228, 196), (50, 205, 50), (139, 0, 139), (100, 149, 237), (138, 43, 226),
             (238, 130, 238),
             (255, 0, 255), (0, 100, 0), (127, 255, 0), (255, 0, 255), (0, 0, 205), (255, 140, 0), (255, 239, 213),
             (199, 21, 133), (124, 252, 0), (147, 112, 219), (106, 90, 205), (176, 196, 222), (65, 105, 225),
             (173, 255, 47),
             (255, 20, 147), (219, 112, 147), (186, 85, 211), (199, 21, 133), (148, 0, 211), (255, 99, 71),
             (144, 238, 144),
             (255, 255, 0), (230, 230, 250), (0, 0, 255), (128, 128, 0), (189, 183, 107), (255, 255, 224),
             (128, 128, 128),
             (105, 105, 105), (64, 224, 208), (205, 133, 63), (0, 128, 128), (72, 209, 204), (139, 69, 19),
             (255, 245, 238),
             (250, 240, 230), (152, 251, 152), (0, 255, 255), (135, 206, 235), (0, 191, 255), (176, 224, 230),
             (0, 250, 154),
             (245, 255, 250), (240, 230, 140), (245, 222, 179), (0, 139, 139), (143, 188, 143), (255, 0, 0),
             (240, 128, 128),
             (102, 205, 170), (60, 179, 113), (46, 139, 87), (165, 42, 42), (178, 34, 34), (175, 238, 238),
             (255, 248, 220),
             (218, 165, 32), (255, 250, 240), (253, 245, 230), (244, 164, 96), (210, 105, 30)]


class ListImgDataset(Dataset):
    def __init__(self, mot_path, img_list, det_db, npy2rgb=False, version='le135') -> None:
        super().__init__()
        self.mot_path = mot_path
        self.img_list = img_list
        self.det_db = det_db
        self.version = version

        '''
        common settings
        '''
        self.img_height = 900
        self.img_width = 1200
        self.mean = [0.27358221, 0.28804452, 0.28133921, 0.26906377, 0.28309119, 0.26928305, 0.28372527, 0.27149373]
        self.std = [0.19756629, 0.17432339, 0.16413284, 0.17581682, 0.18366176, 0.1536845, 0.15964683, 0.16557951]
        self.npy2rgb = npy2rgb

    def load_img_from_file(self, f_path):
        cur_img = np.load(os.path.join(self.mot_path, f_path))
        proposals = []
        im_h, im_w = cur_img.shape[:2]
        det_key = os.path.join(*f_path.split(os.sep)[-2:]).replace('.png','.txt').replace('.jpg','.txt').replace('.npy','.txt')
        for line in self.det_db[det_key]:
            proposals.append(torch.as_tensor(list(map(float, line.split()))))
        if not proposals:
            return cur_img, torch.zeros((0, 10))
        proposals = np.stack(proposals, axis=0)# shape [n, 10]
        return cur_img, proposals

    def init_img(self, img, proposals):
        ori_img = img.copy()
        self.seq_h, self.seq_w = img.shape[:2]
        scale = self.img_height / min(self.seq_h, self.seq_w)
        if max(self.seq_h, self.seq_w) * scale > self.img_width:
            scale = self.img_width / max(self.seq_h, self.seq_w)
        target_h = int(self.seq_h * scale)
        target_w = int(self.seq_w * scale)
        img = cv2.resize(img, (target_w, target_h))
        img = F.normalize(F.to_tensor(img), self.mean, self.std)
        if self.npy2rgb:
            img = img[[1,2,4], ...]
        img = img.unsqueeze(0)

        proposals = torch.as_tensor(proposals, dtype=torch.float32, device=img.device)

        proposals_xywha = poly2obb(proposals[:, :8], version=self.version)
        proposals_xywha_norm = rotate_boxes_to_norm_boxes(proposals_xywha, (self.seq_h, self.seq_w), version=self.version)
        proposals_with_scores = torch.cat([proposals_xywha_norm, proposals[:, 9:10]], dim=-1)

        return img, ori_img, proposals_with_scores

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        img, proposals = self.load_img_from_file(self.img_list[index])
        return self.init_img(img, proposals)


class Detector(object):
    def __init__(self, args, model, vid):
        self.args = args
        self.detr = model

        self.vid = vid
        self.seq_num = os.path.basename(vid)
        img_list = os.listdir(os.path.join(self.args.mot_path, vid))
        img_list = [os.path.join(vid, i) for i in img_list if ('jpg' in i or 'png' in i or 'npy' in i)]

        self.img_list = sorted(img_list)
        self.img_len = len(self.img_list)

        self.predict_path = os.path.join(self.args.output_dir, args.exp_name)
        self.save_path = os.path.join(self.args.output_dir, 'results', self.seq_num)
        os.makedirs(self.predict_path, exist_ok=True)
        os.makedirs(self.save_path, exist_ok=True)

    @staticmethod
    def filter_dt_by_score(dt_instances: Instances, prob_threshold: float) -> Instances:
        keep = dt_instances.scores > prob_threshold
        keep &= dt_instances.obj_idxes >= 0
        return dt_instances[keep]

    def detect(self, prob_threshold=0.6, area_threshold=100, vis=False):
        total_dts = 0
        total_occlusion_dts = 0

        track_instances = None
        with open(self.args.det_db) as f:
            det_db = json.load(f)
        loader = DataLoader(ListImgDataset(self.args.mot_path, self.img_list, det_db, npy2rgb=self.args.npy2rgb), 1, num_workers=2)
        lines = []
        for i, data in enumerate(tqdm(loader, desc=self.vid)):
            # 这里是获得图像
            cur_img, ori_img, proposals = [d[0] for d in data]
            cur_img, proposals = cur_img.cuda(), proposals.cuda()

            # track_instances = None
            if track_instances is not None:
                track_instances.remove('boxes')
                track_instances.remove('labels')
            seq_h, seq_w, _ = ori_img.shape

            res = self.detr.inference_single_image(cur_img, (seq_h, seq_w), track_instances, proposals)
            track_instances = res['track_instances']

            dt_instances = deepcopy(track_instances)

            # filter det instances by score.
            dt_instances = self.filter_dt_by_score(dt_instances, prob_threshold)

            total_dts += len(dt_instances)

            bbox_xyxyxyxy = dt_instances.boxes.tolist()
            identities = dt_instances.obj_idxes.tolist()
            labels = dt_instances.labels.tolist()

            if vis:
                cur_vis_img_path = os.path.join(self.save_path, 'frame_{}.jpg'.format(i+1))
                self.visualize_img_with_bbox_clscolor(cur_vis_img_path, ori_img, dt_instances.to(torch.device('cpu')))
            # save_format = '{frame},{id},{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f},{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f},{cls}\n'
            for xyxyxyxy, track_id,label in zip(bbox_xyxyxyxy, identities,labels):
                if track_id < 0 or track_id is None:
                    continue
                save_line = f'{i+1},{track_id+1},{xyxyxyxy[0]:.2f},{xyxyxyxy[1]:.2f},{xyxyxyxy[2]:.2f},{xyxyxyxy[3]:.2f},{xyxyxyxy[4]:.2f},{xyxyxyxy[5]:.2f},{xyxyxyxy[6]:.2f},{xyxyxyxy[7]:.2f},-1,{label},-1\n'
                lines.append(save_line)
                # lines.append(save_format.format(frame=i + 1, id=track_id, x1=x1, y1=y1, w=w, h=h))
        with open(os.path.join(self.predict_path, f'{self.seq_num}.txt'), 'w') as f:
            f.writelines(lines)
        print("totally {} dts {} occlusion dts".format(total_dts, total_occlusion_dts))


    @staticmethod
    def visualize_img_with_bbox_clscolor(img_path, img, dt_instances: Instances, ref_pts=None, gt_boxes=None, all_ref_pts_aux=None):
        if img.shape[2] >3:
            img = img[:,:,:3]
            img = np.ascontiguousarray(img)
        else:    
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if dt_instances.has('scores'):
            img_show = draw_bboxes(img, np.concatenate([dt_instances.boxes, dt_instances.scores.reshape(-1, 1)], axis=-1), dt_instances.obj_idxes, labels = dt_instances.labels, color='label')
        else:
            img_show = draw_bboxes(img, dt_instances.boxes, dt_instances.obj_idxes, labels=dt_instances.labels, color='label')
        if ref_pts is not None:
            img_show = draw_points(img_show, ref_pts)#暂不管
        if gt_boxes is not None:
            img_show = draw_bboxes(img_show, gt_boxes, identities=np.ones((len(gt_boxes), )) * -1)
        if all_ref_pts_aux is not None:
            img_show = draw_ref_pts_aux(img_show, all_ref_pts_aux)#暂不管
        cv2.imwrite(img_path, img_show)


def draw_bboxes(ori_img, bbox, identities=None, offset=(0, 0), cvt_color=False, labels=None, color='id'):
    if cvt_color:
        ori_img = cv2.cvtColor(np.asarray(ori_img), cv2.COLOR_RGB2BGR)
    img = ori_img
    for i, box in enumerate(bbox):
        x1, y1, x2, y2, x3, y3, x4, y4 = [int(i) for i in box[:8]]
        x1 += offset[0]
        x2 += offset[0]
        x3 += offset[0]
        x4 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        y3 += offset[1]
        y4 += offset[1]
        if len(box) > 8:
            score = '{:.2f}'.format(box[8])
        else:
            score = None
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        id_label = '{:d}'.format(id)
        cls_label = int(labels[i])
        color = COLORS_10[id % len(COLORS_10)] if color=='id' else COLORS_10[cls_label * 5 % len(COLORS_10)]

        # t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        img = plot_one_box([x1, y1, x2, y2, x3, y3, x4, y4], img, color, id_label, score=score)
    return img


def plot_one_box(x, img, color=None, label=None, score=None, line_thickness=None):
    # Plots one bounding box on image img

    # tl = line_thickness or round(
    #     0.002 * max(img.shape[0:2])) + 1  # line thickness
    tl = 2
    color = color or [random.randint(0, 255) for _ in range(3)]
    if len(x) == 8:
        c1, c2, c3, c4 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3])), (int(x[4]), int(x[5])), (int(x[6]), int(x[7]))
        cv2.line(img, c1, c2, color , thickness=tl)
        cv2.line(img, c2, c3, color , thickness=tl)
        cv2.line(img, c3, c4, color , thickness=tl)
        cv2.line(img, c4, c1, color , thickness=tl)
    else:
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        # c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        # cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img,
                    label, (c1[0], c1[1] - 2),
                    0,
                    tl / 3, [225, 255, 255],
                    thickness=tf,
                    lineType=cv2.LINE_AA)
        if score is not None:
            cv2.putText(img, score, (c1[0], c1[1] + 30), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


class RuntimeTrackerBase(object):
    def __init__(self, score_thresh=0.6, filter_score_thresh=0.5, miss_tolerance=10):
        self.score_thresh = score_thresh
        self.filter_score_thresh = filter_score_thresh
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0

    def clear(self):
        self.max_obj_id = 0

    def update(self, track_instances: Instances):
        device = track_instances.obj_idxes.device

        track_instances.disappear_time[track_instances.scores >= self.score_thresh] = 0
        new_obj = (track_instances.obj_idxes == -1) & (track_instances.scores >= self.score_thresh)
        disappeared_obj = (track_instances.obj_idxes >= 0) & (track_instances.scores < self.filter_score_thresh)
        num_new_objs = new_obj.sum().item()

        track_instances.obj_idxes[new_obj] = self.max_obj_id + torch.arange(num_new_objs, device=device)
        self.max_obj_id += num_new_objs

        track_instances.disappear_time[disappeared_obj] += 1
        to_del = disappeared_obj & (track_instances.disappear_time >= self.miss_tolerance)
        track_instances.obj_idxes[to_del] = -1


if __name__ == '__main__':

    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    parser.add_argument('--score_threshold', default=0.5, type=float)
    parser.add_argument('--update_score_threshold', default=0.5, type=float)
    parser.add_argument('--miss_tolerance', default=20, type=int)
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # load model and weights
    detr, _, _ = build_model(args)
    detr.track_embed.score_thr = args.update_score_threshold
    detr.track_base = RuntimeTrackerBase(args.score_threshold, args.score_threshold, args.miss_tolerance)
    checkpoint = torch.load(args.resume, map_location='cpu')
    detr = load_model(detr, args.resume)
    detr.eval()
    detr = detr.cuda()

    # '''for HSMOT RGB submit''' 
    sub_dir = 'test/npy'
    seq_nums = os.listdir(os.path.join(args.mot_path, sub_dir))
    if 'seqmap' in seq_nums:
        seq_nums.remove('seqmap')
    vids = [os.path.join(sub_dir, seq) for seq in seq_nums]

    rank = int(os.environ.get('RLAUNCH_REPLICA', '0'))
    ws = int(os.environ.get('RLAUNCH_REPLICA_TOTAL', '1'))
    vids = vids[rank::ws]

    for vid in vids:
        det = Detector(args, model=detr, vid=vid)
        det.detect(args.score_threshold, args.vis)
