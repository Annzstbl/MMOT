from collections import defaultdict
import json
import os
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.utils.data
import os.path as osp
# from PIL import Image, ImageDraw
import copy
import datasets.transforms as T
from models.structures import Instances
from hsmot.mmlab.hs_mmrotate import poly2obb
from hsmot.datasets.pipelines.compose import MotCompose
from hsmot.datasets.pipelines.channel import MotrToMmrotate, MmrotateToMotr
from hsmot.datasets.pipelines.loading import MotLoadAnnotations, MotLoadImageFromFile, MotLoadMultichannelImageFromNpy
from hsmot.datasets.pipelines.transforms import MotRRsize, MotRRandomFlip, MotRRandomCrop, MotNormalize, MotPad
from hsmot.datasets.pipelines.formatting import MotCollect, MotDefaultFormatBundle, MotShow


from random import choice, randint


CLASSES = ['car', 'bike','pedestrian', 'van', 'truck', 'bus', 'tricycle', 'awning-bike']
class DetHSMOTDetection:
    def __init__(self, args, data_txt_path: str, seqs_folder, dataset2transform, block_cat=None, block_trunc=False, vid_white_list=None, version='le135'):

        self.args = args
        self.dataset2transform = dataset2transform
        self.num_frames_per_batch = max(args.sampler_lengths)
        self.sample_mode = args.sample_mode
        self.sample_interval = args.sample_interval
        self.video_dict = {}
        self.split_dir = os.path.join(args.mot_path, "train", "npy")
        self.labels_dir = os.path.join(args.mot_path, "train", "mot")
        self.version = version


        assert block_cat == None, f'不支持屏蔽类别'
        assert block_trunc == False, f'不支持屏蔽截断'

        vid_white_list = ['data46-4']

        self.labels_full = defaultdict(lambda: defaultdict(list))
        for vid in os.listdir(self.labels_dir):
            # 过滤视频序列
            if vid_white_list is not None and os.path.splitext(vid)[0] not in vid_white_list:
                print(f'skip vid {vid}')
                continue

            gt_path = os.path.join(self.labels_dir, vid)
            for l in open(gt_path):
                t, i, *x0y0x1y1x2y2x3y3, _, cls, trunc = l.strip().split(',')[:13] 
                t, i, cls = map(int, (t, i, cls))
                x0, y0, x1, y1, x2, y2, x3, y3 = map(float, (x0y0x1y1x2y2x3y3))
                self.labels_full[vid][t].append(np.array([x0, y0, x1, y1, x2, y2, x3, y3, i, cls], dtype=np.float32))

        vid_files = list(self.labels_full.keys())

        self.indices = []
        self.vid_tmax = {}
        for vid in vid_files:
            self.video_dict[vid] = len(self.video_dict)
            t_min = min(self.labels_full[vid].keys())
            t_max = max(self.labels_full[vid].keys()) + 1
            self.vid_tmax[vid] = t_max - 1
            for t in range(t_min, t_max - self.num_frames_per_batch):
                self.indices.append((vid, t))
        print(f"Found {len(vid_files)} videos, {len(self.indices)} frames")

        self.sampler_steps: list = args.sampler_steps
        self.lengths: list = args.sampler_lengths
        print("sampler_steps={} lenghts={}".format(self.sampler_steps, self.lengths))
        self.period_idx = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        if self.sampler_steps is None or len(self.sampler_steps) == 0:
            # fixed sampling length.
            return

        for i in range(len(self.sampler_steps)):
            if epoch >= self.sampler_steps[i]:
                self.period_idx = i + 1
        print("set epoch: epoch {} period_idx={}".format(epoch, self.period_idx))
        self.num_frames_per_batch = self.lengths[self.period_idx]

    def step_epoch(self):
        # one epoch finishes.
        print("Dataset: epoch {} finishes".format(self.current_epoch))
        self.set_epoch(self.current_epoch + 1)

    @staticmethod
    def _targets_to_instances(targets: dict, img_shape) -> Instances:
        gt_instances = Instances(tuple(img_shape))
        gt_instances.boxes = targets['boxes']
        gt_instances.labels = targets['labels']
        gt_instances.obj_ids = targets['obj_ids']
        gt_instances.norm_boxes = targets['norm_boxes']
        return gt_instances

    def _pre_single_frame(self, vid, idx: int):
        img_path = os.path.join(self.split_dir, osp.splitext(vid)[0], f'{idx:06d}.npy')
       
        data_info = {}
        data_info['filename'] = img_path
        data_info['ann'] = {}
        gt_bboxes = []
        gt_labels = []
        gt_ids = []
        gt_polygons = []
        obj_idx_offset = self.video_dict[vid] * 100000
        
        for *xyxyxyxy, id, cls in self.labels_full[vid][idx]:
            x, y, w, h, a = poly2obb_np(np.array(xyxyxyxy, dtype=np.float32), self.version)
            gt_bboxes.append([x, y, w, h, a])
            gt_labels.append(cls)
            gt_polygons.append(xyxyxyxy)
            gt_ids.append(id+obj_idx_offset)
        
        if gt_bboxes:
            data_info['ann']['bboxes'] = np.array(
                gt_bboxes, dtype=np.float32)
            data_info['ann']['labels'] = np.array(
                gt_labels, dtype=np.int64)
            data_info['ann']['polygons'] = np.array(
                gt_polygons, dtype=np.float32)
            data_info['ann']['trackids'] = np.array(gt_ids, dtype=np.int64)
        else:
            data_info['ann']['bboxes'] = np.zeros((0, 5),
                                                    dtype=np.float32)
            data_info['ann']['labels'] = np.array([], dtype=np.int64)
            data_info['ann']['polygons'] = np.zeros((0, 8),
                                                    dtype=np.float32)
            data_info['ann']['trackids'] = np.zeros((0), dtype=np.int64)
        
        img_info = data_info
        ann_info = data_info['ann']
        results = dict(img_info=img_info, ann_info=ann_info)

        # """Prepare results dict for pipeline."""
        results['img_prefix'] = None
        results['seg_prefix'] = None
        results['proposal_file'] = None
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        
        return results

    def pre_continuous_frames(self, vid, indices):
        # return zip(*[self._pre_single_frame(vid, i) for i in indices])
        return [self._pre_single_frame(vid, i) for i in indices]

    def sample_indices(self, vid, f_index):
        assert self.sample_mode == 'random_interval'
        rate = randint(1, self.sample_interval + 1)
        tmax = self.vid_tmax[vid]
        ids = [f_index + rate * i for i in range(self.num_frames_per_batch)]
        return [min(i, tmax) for i in ids]

    def __getitem__(self, idx):
        vid, f_index = self.indices[idx]
        indices = self.sample_indices(vid, f_index)
        # images, targets = self.pre_continuous_frames(vid, indices)
        data_info = self.pre_continuous_frames(vid, indices)
        transform = self.dataset2transform
        # if transform is not None:
        #     images, targets = transform(images, targets)
        if transform is not None:
            results = transform(data_info)
        assert type(results) == tuple
        images = results[0]
        targets = results[1]
        img_metas = results[2]

        gt_instances = []
        for img_i, targets_i in zip(images, targets):
            gt_instances_i = self._targets_to_instances(targets_i, img_i.size())
            gt_instances.append(gt_instances_i)
        return {
            'imgs': images,
            'gt_instances': gt_instances,
            'img_metas': img_metas
        }
        
        # return images, gt_instances
        # 重组成[(img1, tar1), (img2, tar2)...]的形式
        # return [(img, tar) for img, tar in zip(images, gt_instances)]


    def __len__(self):
        return len(self.indices)

def build_dataset2transform(args, image_set):
    hsmot_train = make_transforms_for_hsmot_rgb('train', args)
    hsmot_test = make_transforms_for_hsmot_rgb('val', args)

    # dataset2transform_train = {'HSMOT_RGB': hsmot_train}
    # dataset2transform_val = {'HSMOT_RGB': hsmot_test}
    if image_set == 'train':
        return hsmot_train
    elif image_set == 'val':
        return hsmot_test
    else:
        raise NotImplementedError()


def build(image_set, args):
    root = Path(args.mot_path)
    assert root.exists(), f'provided MOT path {root} does not exist'
    dataset2transform = build_dataset2transform(args, image_set)
    if image_set == 'train':
        data_txt_path = args.data_txt_path_train
        dataset = DetHSMOTDetection(args, data_txt_path=data_txt_path, seqs_folder=root, dataset2transform=dataset2transform, block_cat=args.block_cat, block_trunc=args.block_trunc, vid_white_list=args.vid_white_list)
    if image_set == 'val':
        data_txt_path = args.data_txt_path_val
        dataset = DetHSMOTDetection(args, data_txt_path=data_txt_path, seqs_folder=root, dataset2transform=dataset2transform, block_cat=args.block_cat, block_trunc=args.block_trunc, vid_white_list=args.vid_white_list)
    return dataset

def make_transforms_for_hsmot_rgb(image_set, args=None):


    scales_h = [608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 1024, 1056, 1088, 1120, 1152, 1184]
    scales_w = [ int(h/4*3) for h in scales_h ]
    scales = list(zip(scales_h, scales_w))
    mean = [0.27358221, 0.28804452, 0.28133921, 0.26906377, 0.28309119, 0.26928305, 0.28372527, 0.27149373]
    std = [0.19756629, 0.17432339, 0.16413284, 0.17581682, 0.18366176, 0.1536845, 0.15964683, 0.16557951]
    mean = [m*255 for m in mean]
    std = [s*255 for s in std]
    
    if image_set == 'train':
        return MotCompose([
            MotrToMmrotate(),
            MotLoadMultichannelImageFromNpy(),
            MotLoadAnnotations(poly2mask=False),
            # MotShow(save_path='/data/users/litianhao/hsmot_code/workdir/debug/MotShow',to_bgr=False, img_name_tail='ori'),
            MotRRandomFlip(direction=['horizontal', 'vertical'], flip_ratio=[0.25, 0.25], version='le135'),
            MotRRandomCrop(crop_size=(800,1200), crop_type='absolute_range', version='le135', allow_negative_crop=True, iof_thr=0.5),
            # MotShow(save_path='/data/users/litianhao/hsmot_code/workdir/debug/MotShow',to_bgr=False, img_name_tail='randomcrop'),
            MotRRsize(multiscale_mode='value', img_scale=scales, bbox_clip_border=False),
            MotNormalize(mean=mean, std=std, to_rgb=False),
            MotPad(size_divisor=32),
            # MotShow(save_path='/data/users/litianhao/hsmot_code/workdir/motr/debug', version='le135', mean=[125,125,125,125,125,125,125,125,], std=[125,125,125,125,125,125,125,125,], to_bgr=False, img_name_tail='afterpad'),
            MotDefaultFormatBundle(),
            MotCollect(keys=['img', 'gt_bboxes', 'gt_labels', 'gt_trackids']),
            MmrotateToMotr()
        ])   
    
    if image_set == 'val':
        return T.MotCompose([
            MotrToMmrotate(),
            MotLoadImageFromFile(),
            MotRRsize(multiscale_mode='value', img_scale=scales, bbox_clip_border=False),
            # MotNormalize(mean=[0.259, 0.274, 0.241], std=[0.143, 0.140, 0.137]),
            MotNormalize(mean=mean, std=std, to_rgb=False),
            MotDefaultFormatBundle(),
            MotCollect(keys=['img']),
            MmrotateToMotr()
        ])

    raise ValueError(f'unknown {image_set}')

