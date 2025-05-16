# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from .dance import build as build_e2e_dance
from .joint import build as build_e2e_joint
from .hsmot_rgb import build as build_e2e_hsmot_rgb
from .hsmot_8ch import build as build_e2e_hsmot_8ch
from .hsmot_8ch_1seq import build as build_e2e_hsmot_8ch_1seq


def build_dataset(image_set, args):
    if args.dataset_file == 'e2e_joint':
        return build_e2e_joint(image_set, args)
    if args.dataset_file == 'e2e_dance':
        return build_e2e_dance(image_set, args)
    if args.dataset_file == 'e2e_hsmot_rgb':
        return build_e2e_hsmot_rgb(image_set, args)
    if args.dataset_file == 'e2e_hsmot_8ch':
        return build_e2e_hsmot_8ch(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
