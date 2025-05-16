from collections import defaultdict
import json
import os
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.utils.data
import os.path as osp
from PIL import Image, ImageDraw
import copy
from models.structures import Instances
from hsmot.mmlab.hs_mmrotate import poly2obb, poly2obb_np



from random import choice, randint


from hsmot.datasets.pipelines.compose import MotCompose, MotRandomChoice
from hsmot.datasets.pipelines.channel import MotrToMmrotate, MmrotateToMotrv2
from hsmot.datasets.pipelines.loading import MotLoadAnnotations, MotLoadImageFromFile, MotRLoadProposals, MotLoadMultichannelImageFromNpy
from hsmot.datasets.pipelines.transforms import MotRRsize, MotRRandomFlip, MotRRandomCrop, MotNormalize, MotPad
from hsmot.datasets.pipelines.formatting import MotCollect, MotDefaultFormatBundle, MotShow



class DetHSMOTDetection:
    def __init__(self, args, transform, image_set='train', version='le135', npy2rgb=False):
        self.args = args
        self.transform = transform
        self.num_frames_per_batch = max(args.sampler_lengths)
        self.sample_mode = args.sample_mode
        self.sample_interval = args.sample_interval
        self.video_dict = {}
        self.split_dir = os.path.join(args.mot_path, image_set, "npy")
        self.labels_dir = os.path.join(args.mot_path, image_set, "mot")
        self.version = version
        self.yolo_db = args.det_db
        self.npy2rgb = npy2rgb
        vid_white_list = None

        self.labels_full = defaultdict(lambda: defaultdict(list))
        for vid in os.listdir(self.labels_dir):
            if vid_white_list is not None and os.path.splitext(vid)[0] not in vid_white_list:
                print(f'skip vid {vid}')
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


        # assert(args.append_crowd, f'HSMOT dataset doesnot support crowdhuman dataset')

        # 加载yolo
        if self.yolo_db:
            print(f'load yolo detection results from file f{self.yolo_db}')
            with open(args.det_db) as f:
                self.det_db = json.load(f)

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
        # n_gt = len(targets['labels'])
        gt_instances.boxes = targets['boxes']#[:n_gt]
        gt_instances.labels = targets['labels']
        gt_instances.obj_ids = targets['obj_ids']
        gt_instances.norm_boxes = targets['norm_boxes']

        proposals_instances = Instances(tuple(img_shape))
        proposals_instances.proposals = targets['proposals']
        proposals_instances.proposal_scores = targets['proposal_scores']
        proposals_instances.norm_proposals = targets['norm_proposals']

        return gt_instances, proposals_instances

    def _pre_single_frame(self, vid, idx: int):
        img_path = os.path.join(self.split_dir, osp.splitext(vid)[0], f'{idx:06d}.npy')

        data_info = {}
        data_info['filename'] = img_path
        data_info['ann'] = {}
        gt_bboxes = []
        gt_labels = []
        gt_ids = []
        gt_scores = []
        gt_polygons = []
        proposals= []# yolo检测结果
        proposals_scores = []
        obj_idx_offset = self.video_dict[vid] * 100000

        for *xyxyxyxy, id, cls in self.labels_full[vid][idx]:
            x, y, w, h, a = poly2obb_np(np.array(xyxyxyxy, dtype=np.float32), self.version)
            gt_bboxes.append([x, y, w, h, a])
            gt_labels.append(cls)
            gt_polygons.append(xyxyxyxy)
            gt_ids.append(id+obj_idx_offset)
            gt_scores.append(1)

        #读取yolo的proposal
        txt_key = os.path.join(osp.splitext(vid)[0], f'{idx:06d}.txt')
        for line in self.det_db[txt_key]:
            *box, label, s = map(float, line.split())
            x, y, w, h, a = poly2obb_np(np.array(box, dtype=np.float32), self.version)
            proposals.append([x, y, w, h, a])
            proposals_scores.append(s)

        if gt_bboxes:
            data_info['ann']['bboxes'] = np.array(
                gt_bboxes, dtype=np.float32)
            data_info['ann']['labels'] = np.array(
                gt_labels, dtype=np.int64)
            data_info['ann']['polygons'] = np.array(
                gt_polygons, dtype=np.float32)
            data_info['ann']['trackids'] = np.array(gt_ids, dtype=np.int64)
            # data_info['ann']['proposals'] = np.array(proposals, dtype=np.float32)
            # data_info['ann']['proposal_scores'] = np.array(proposals_scores, dtype=np.float32)

        else:
            data_info['ann']['bboxes'] = np.zeros((0, 5),
                                                    dtype=np.float32)
            data_info['ann']['labels'] = np.array([], dtype=np.int64)
            data_info['ann']['polygons'] = np.zeros((0, 8),
                                                    dtype=np.float32)
            data_info['ann']['trackids'] = np.zeros((0), dtype=np.int64)
            # data_info['ann']['proposals'] = np.zeros((0, 5),
            #                                         dtype=np.float32)
            # data_info['ann']['proposal_scores'] = np.array([], dtype=np.float3)


        
        # 最终
        img_info = data_info
        ann_info = data_info['ann']
        results = dict(img_info=img_info, ann_info=ann_info)

        # 加载proposals
        if proposals:
            results['proposals'] = np.array(proposals, dtype=np.float32)
            results['proposal_scores'] = np.array(proposals_scores, dtype=np.float32)
        else:
            results['proposals'] = np.zeros((0, 5), dtype=np.float32)
            results['proposal_scores'] = np.array([], dtype=np.float32)

        # """Prepare results dict for pipeline."""
        results['img_prefix'] = None
        results['seg_prefix'] = None
        results['proposal_file'] = None # TODO不知道具体作用
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
        assert idx < len(self.indices)
        vid, f_index = self.indices[idx]
        indices = self.sample_indices(vid, f_index)
        # images, targets = self.pre_continuous_frames(vid, indices)
        data_info = self.pre_continuous_frames(vid, indices)
        # dataset_name = targets[0]['dataset']
        # transform = self.dataset2transform[dataset_name]
        if self.transform is not None:
            results = self.transform(data_info)
        # images, targets = self.transform(images, targets)
        assert type(results) == tuple
        images = results[0]
        targets = results[1]
        img_metas = results[2]
        

        gt_instances, proposals_instances = [], []
        for img_i, targets_i in zip(images, targets):
            gt_instances_i, proposal_instances_i = self._targets_to_instances(targets_i, img_i.shape[1:3])
            gt_instances.append(gt_instances_i)
            proposals_instances.append(proposal_instances_i)

        if self.npy2rgb:
            images = [img[[1,2,4], :, :] for img in images]


        return {
            'imgs': images,
            'gt_instances': gt_instances,
            'proposals_instances': proposals_instances,
            'img_metas': img_metas
        }
        
        # return images, gt_instances
        # 重组成[(img1, tar1), (img2, tar2)...]的形式
        # return [(img, tar) for img, tar in zip(images, gt_instances)]


    def __len__(self):
        return len(self.indices)

def build_transform(args, image_set):
    hsmot_train = make_transforms_for_hsmot_rgb('train', args)
    hsmot_val = make_transforms_for_hsmot_rgb('val', args)

    if image_set == 'train':
        return hsmot_train
    elif image_set == 'val':
        return hsmot_val
    else:
        raise NotImplementedError()


def build(image_set, args):
    root = Path(args.mot_path)
    assert root.exists(), f'provided MOT path {root} does not exist'
    transform = build_transform(args, image_set)
    if image_set == 'train':
        dataset = DetHSMOTDetection(args, transform=transform, npy2rgb=args.npy2rgb)
    if image_set == 'val':
        dataset = DetHSMOTDetection(args, transform=transform, npy2rgb=args.npy2rgb)
    return dataset

def make_transforms_for_hsmot_rgb(image_set, args=None):
        
    scales_w = [608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 1024, 1056, 1088, 1120, 1152, 1184]
    scales_h = [ int(w/4*3) for w in scales_w ]
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
            MotRLoadProposals(),
            MotRRandomFlip(direction=['horizontal'], flip_ratio=[0.5], version='le135'),
            MotRandomChoice(transforms=[
                [
                    MotRRsize(multiscale_mode='value', img_scale=scales, bbox_clip_border=False),],
                [
                    MotRRandomCrop(crop_size=(800,1200), crop_type='absolute_range', allow_negative_crop=True, iof_thr=0.5),
                    MotRRsize(multiscale_mode='value', img_scale=scales, bbox_clip_border=False),]
                ]),     
            MotNormalize(mean=mean, std=std, to_rgb=False),
            MotPad(size_divisor=32),
            # MotShow(save_path='/data/users/litianhao/hsmot_code/workdir/motrv2/debug', version='le135', mean=mean, std=std, img_name_tail='final', show_proposals=True, to_bgr=False),
            MotDefaultFormatBundle(),
            MotCollect(keys=['img', 'gt_bboxes', 'gt_labels', 'gt_trackids', 'proposals', 'proposal_scores']),
            MmrotateToMotrv2()
        ])   
    
    #DEBUG
    # if image_set == 'train':
    #     return MotCompose([
    #         MotrToMmrotate(),
    #         MotLoadImageFromFile(),
    #         MotLoadAnnotations(poly2mask=False),
    #         MotRLoadProposals(),

    #         MotShow(save_path='/data/users/litianhao/hsmot_code/workdir/debug/MotShow',to_bgr=False, img_name_tail='ori', show_proposals=True),

    #         MotRRandomFlip(direction=['horizontal'], flip_ratio=[0.5], version='le135'),
    #         MotRandomChoice(transforms=[
    #             [
    #                 MotRRsize(multiscale_mode='value', img_scale=scales, bbox_clip_border=False),],
    #             [
    #                 MotRRandomCrop(crop_size=(800,1200), crop_type='absolute_range', allow_negative_crop=True, iof_thr=0.5),
    #                 MotRRsize(multiscale_mode='value', img_scale=scales, bbox_clip_border=False),]
    #             ]),     
    #         MotShow(save_path='/data/users/litianhao/hsmot_code/workdir/debug/MotShow',to_bgr=False, img_name_tail='randomcrop', show_proposals=True),     
             
    #         MotNormalize(mean=[66.04, 69.87, 61.45], std=[36.46, 35.70, 34.93]),
    #         MotPad(size_divisor=32),

    #         MotShow(save_path='/data/users/litianhao/hsmot_code/workdir/debug/MotShow', version='le135', mean=[66.04, 69.87, 61.45], std=[36.46, 35.70, 34.93], img_name_tail='final', show_proposals=True),
            
    #         MotDefaultFormatBundle(),
    #         MotCollect(keys=['img', 'gt_bboxes', 'gt_labels', 'gt_trackids', 'proposals', 'proposal_scores']),
    #         MmrotateToMotrv2()
    #     ])   


    #TODO 没有VAL
    if image_set == 'val':
        return None

    raise ValueError(f'unknown {image_set}')

