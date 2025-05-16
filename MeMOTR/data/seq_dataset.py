# Copyright (c) Ruopeng Gao. All Rights Reserved.
import os
import cv2

import torchvision.transforms.functional as F

from torch.utils.data import Dataset
import numpy as np
import mmcv
from hsmot.mmlab.hs_mmdet import to_tensor


class SeqDataset(Dataset):
    def __init__(self, seq_dir: str, stride=32, npy2rgb=False):
        # a hack implementation for BDD100K and others:
        # if "BDD100K" in seq_dir:
        #     image_paths = sorted(os.listdir(os.path.join(seq_dir)))
        #     image_paths = [os.path.join(seq_dir, _) for _ in image_paths if ("jpg" in _) or ("png" in _)]
        # else:
        #     image_paths = sorted(os.listdir(os.path.join(seq_dir, "img1")))
        #     image_paths = [os.path.join(seq_dir, "img1", _) for _ in image_paths if ("jpg" in _) or ("png" in _)]
        image_paths = sorted(os.listdir(seq_dir))
        image_paths = [os.path.join(seq_dir, _) for _ in image_paths if ("npy" in _)]
        self.image_paths = image_paths
        self.image_height = 900
        self.image_width = 1200
        mean = [0.27358221, 0.28804452, 0.28133921, 0.26906377, 0.28309119, 0.26928305, 0.28372527, 0.27149373]
        std = [0.19756629, 0.17432339, 0.16413284, 0.17581682, 0.18366176, 0.1536845, 0.15964683, 0.16557951]
        mean = [_*255 for _ in mean]
        std = [_*255 for _ in std]
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

        self.stride=stride
        self.npy2rgb=npy2rgb

        return

    @staticmethod
    def load(path):
        image = np.load(path)
        assert image is not None

        return image

    def process_image(self, image):
        ori_image = image.copy()

        # 首先归一化  然后resize 然后padding
        image = mmcv.imnormalize(image, self.mean, self.std, to_rgb=False)
        # image = mmcv.imresize(image, (self.scale_size_w, self.scale_size_h))
        image = mmcv.impad_to_multiple(image, self.stride, pad_val=0)
        image = np.ascontiguousarray(image.transpose(2, 0, 1))
        image = to_tensor(image)
        if self.npy2rgb:
            image = image[[1,2,4], :, :]
        # image = image.unsqueeze(0)
        return image, ori_image

        # ori_image = image.copy()
        # h, w = image.shape[:2]
        # scale = self.image_height / min(h, w)
        # if max(h, w) * scale > self.image_width:
        #     scale = self.image_width / max(h, w)
        # target_h = int(h * scale)
        # target_w = int(w * scale)
        # image = cv2.resize(image, (target_w, target_h))
        # image = F.normalize(F.to_tensor(image), self.mean, self.std)
        # return image, ori_image

    def __getitem__(self, item):
        image = self.load(self.image_paths[item])
        info = self.image_paths[item]
        return self.process_image(image=image), info

    def __len__(self):
        return len(self.image_paths)
