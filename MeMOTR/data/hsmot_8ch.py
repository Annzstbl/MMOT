# @Author       : Ruopeng Gao
# @Date         : 2022/8/30
import os
from math import floor
from random import randint

import torch
from PIL import Image
import data.transforms as T
# from typing import List
# from torch.utils.data import Dataset
from .mot import MOTDataset
from collections import defaultdict

import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
import numpy as np
from hsmot.mmlab.hs_mmrotate import poly2obb, poly2obb_np
import os.path as osp
from hsmot.datasets.pipelines.compose import MotCompose, MotRandomChoice
from hsmot.datasets.pipelines.channel import MotrToMmrotate, MmrotateToMotr, MmrotateToMotip, MotipToMmrotate
from hsmot.datasets.pipelines.loading import MotLoadAnnotations, MotLoadImageFromFile, MotLoadMultichannelImageFromNpy
from hsmot.datasets.pipelines.transforms import MotRRsize, MotRRandomFlip, MotRRandomCrop, MotNormalize, MotPad
from hsmot.datasets.pipelines.formatting import MotCollect, MotDefaultFormatBundle, MotShow


class hsmot_8ch(MOTDataset):
    def __init__(self, config: dict, split: str, transform, version='le135'):
        super(hsmot_8ch, self).__init__(config=config, split=split, transform=transform)

        self.config = config
        self.transform = transform
        self.dataset_name = config["DATASET"]
        self.dataset_name = self.dataset_name.replace("_8ch", "")
        assert split == "train" or split == "test", f"Split {split} is not supported!"
        self.version = version

        # Sampling setting.
        self.sample_steps: list = config["SAMPLE_STEPS"]
        self.sample_intervals: list = config["SAMPLE_INTERVALS"]
        self.sample_modes: list = config["SAMPLE_MODES"]
        self.sample_lengths: list = config["SAMPLE_LENGTHS"]
        self.sample_stage = None
        self.sample_begin_frames = None
        self.sample_length = None
        self.sample_mode = None
        self.sample_interval = None
        self.sample_vid_tmax = None

        self.npy2rgb = config["NPY2RGB"]

        self.gts = defaultdict(lambda: defaultdict(list))
        self.vid_idx = dict()
        self.idx_vid = dict()

        self.split_dir = os.path.join(config["DATA_ROOT"], self.dataset_name, split, "npy")
        assert os.path.exists(self.split_dir), f"Dir {self.split_dir} is not exist."
        self.labels_dir = os.path.join(config["DATA_ROOT"], self.dataset_name, split, "mot")

        vid_white_list = config["VID_WHITE_LIST"] if "VID_WHITE_LIST" in config else None

        self.labels_full = defaultdict(lambda: defaultdict(list))
        for vid in os.listdir(self.labels_dir):
            # 过滤视频序列
            if vid_white_list is not None and os.path.splitext(vid)[0] not in vid_white_list:
                print(f'skip vid {vid}')
                continue
            else:
                print(f'loading vid {vid}')

            gt_path = os.path.join(self.labels_dir, vid)
            for l in open(gt_path):
                t, i, *x0y0x1y1x2y2x3y3, _, cls, trunc = l.strip().split(',')[:13] 
                t, i, cls = map(int, (t, i, cls))
                x0, y0, x1, y1, x2, y2, x3, y3 = map(float, (x0y0x1y1x2y2x3y3))
                self.labels_full[vid][t].append(np.array([x0, y0, x1, y1, x2, y2, x3, y3, i, cls], dtype=np.float32))
        vid_files = list(self.labels_full.keys())

        for vid in vid_files:
            self.vid_idx[vid] = len(self.vid_idx)
            self.idx_vid[self.vid_idx[vid]] = vid

        # for vid in os.listdir(self.split_dir):
        #     gt_path = os.path.join(self.split_dir, vid, "gt", "gt.txt")
        #     for line in open(gt_path):
        #         # gt per line: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, 1, 1, 1
        #         # https://github.com/DanceTrack/DanceTrack
        #         t, i, *xywh, a, b, c = line.strip().split(",")[:9]
        #         t, i, a, b, c = map(int, (t, i, a, b, c))
        #         x, y, w, h = map(float, xywh)
        #         assert a == b == c == 1, f"Check Digit ERROR!"
        #         self.gts[vid][t].append([i, x, y, w, h])

        # vids = list(self.gts.keys())

        # for vid in vids:
        #     self.vid_idx[vid] = len(self.vid_idx)
        #     self.idx_vid[self.vid_idx[vid]] = vid

        self.set_epoch(0)

        return

    def __getitem__(self, item):
        vid, begin_frame = self.sample_begin_frames[item]
        frame_idxs = self.sample_frames_idx(vid=vid, begin_frame=begin_frame)
        data_info = self.get_multi_frames(vid=vid, idxs=frame_idxs)
        assert self.transform is not None
        results = self.transform(data_info)
        if self.npy2rgb:
            images = [img[[1,2,4],...] for img in results[0]]
        else:
            images = results[0]
        return{
            "images": images,
            "targets": results[1],
            "img_metas": results[2]
        }

    def __len__(self):
        assert self.sample_begin_frames is not None, "Please use set_epoch to init DanceTrack Dataset."
        return len(self.sample_begin_frames)

    def sample_frames_idx(self, vid: int, begin_frame: int) -> list[int]:
        if self.sample_mode == "random_interval":
            assert self.sample_length > 1, "Sample length is less than 2."
            remain_frames = self.sample_vid_tmax[vid] - begin_frame
            max_interval = floor(remain_frames / (self.sample_length - 1))
            interval = min(randint(1, self.sample_interval), max_interval)
            frame_idxs = [begin_frame + interval * i for i in range(self.sample_length)]
            return frame_idxs
        else:
            raise ValueError(f"Sample mode {self.sample_mode} is not supported.")

    def set_epoch(self, epoch: int):
        self.sample_begin_frames = list()
        self.sample_vid_tmax = dict()
        self.sample_stage = 0
        for step in self.sample_steps:
            if epoch >= step:
                self.sample_stage += 1
        assert self.sample_stage < len(self.sample_steps) + 1
        self.sample_length = self.sample_lengths[min(len(self.sample_lengths) - 1, self.sample_stage)]
        self.sample_mode = self.sample_modes[min(len(self.sample_modes) - 1, self.sample_stage)]
        self.sample_interval = self.sample_intervals[min(len(self.sample_intervals) - 1, self.sample_stage)]
        for vid in self.vid_idx.keys():
            t_min = min(self.labels_full[vid].keys())
            t_max = max(self.labels_full[vid].keys())
            self.sample_vid_tmax[vid] = t_max
            for t in range(t_min, t_max - (self.sample_length - 1) + 1):
                self.sample_begin_frames.append((vid, t))

        return

    def get_single_frame(self, vid: str, idx: int):
        #确认源代码的 frame_idx 现在看不需要
        '''
            info["boxes"] = list()
            info["ids"] = list()
            info["labels"] = list()
            info["areas"] = list()
            info["frame_idx"] = torch.as_tensor(idx)

        '''
        img_path = os.path.join(self.split_dir, osp.splitext(vid)[0], f'{idx:06d}.npy')
        data_info = {}
        data_info['filename'] = img_path
        data_info['ann'] = {}
        gt_bboxes = []
        gt_labels = []
        gt_ids = []
        gt_polygons = []
        obj_idx_offset = self.vid_idx[vid] * 100000
        
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

    def get_multi_frames(self, vid: str, idxs: list[int]):
        return [self.get_single_frame(vid=vid, idx=i) for i in idxs]


def transfroms_for_train(coco_size: bool = False, overflow_bbox: bool = False, reverse_clip: bool = False):
    mean = [0.27358221, 0.28804452, 0.28133921, 0.26906377, 0.28309119, 0.26928305, 0.28372527, 0.27149373]
    std = [0.19756629, 0.17432339, 0.16413284, 0.17581682, 0.18366176, 0.1536845, 0.15964683, 0.16557951]
    mean = [_*255 for _ in mean]
    std = [_*255 for _ in std]

    scales_w = [608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 1024, 1056, 1088, 1120, 1152, 1184]
    scales_h = [ int(w/4*3) for w in scales_w ]
    scales = list(zip(scales_h, scales_w))
    
    return MotCompose([
                MotipToMmrotate(),
                MotLoadMultichannelImageFromNpy(),
                MotLoadAnnotations(poly2mask=False),
                MotRRandomFlip(direction=['horizontal'], flip_ratio=[0.5], version='le135'),
                MotRandomChoice(transforms=[
                    [
                        MotRRsize(multiscale_mode='value', img_scale=scales, bbox_clip_border=False),
                        ],
                    [
                        MotRRandomCrop(crop_size=(800, 1200), crop_type='absolute_range', version='le135', allow_negative_crop=True, iof_thr=0.5),
                        MotRRsize(multiscale_mode='value', img_scale=scales, bbox_clip_border=False),
                        ]
                ]),                
                # 缺少一个颜色预训练
                MotNormalize(mean=mean, std=std, to_rgb=False),
                MotPad(size_divisor=32),
                MotDefaultFormatBundle(),
                MotCollect(keys=['img', 'gt_bboxes', 'gt_labels', 'gt_trackids']),
                MmrotateToMotip()
                #TODO 缺少一个reverse clip 但实际参数是0所以暂不实现
            ])


def transforms_for_eval():
    #TODO 推理transform
    return T.MultiCompose([
        T.MultiRandomResize(sizes=[800], max_size=1333),
        T.MultiCompose([
            T.MultiToTensor(),
            T.MultiNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    ])


def build(config: dict, split: str):
    if split == "train":
        return hsmot_8ch(
            config=config,
            split=split,
            transform=transfroms_for_train(
                coco_size=config["COCO_SIZE"],
                overflow_bbox=config["OVERFLOW_BBOX"],
                reverse_clip=config["REVERSE_CLIP"]
            )
        )
    elif split == "test":
        return hsmot_8ch(config=config, split=split, transform=transforms_for_eval())
    else:
        raise ValueError(f"Data split {split} is not supported for DanceTrack dataset.")
