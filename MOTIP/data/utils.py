# Copyright (c) Ruopeng Gao. All Rights Reserved.
from utils.nested_tensor import NestedTensor, tensor_list_to_nested_tensor
import torch
from hsmot.datasets.pipelines.channel import rotate_boxes_to_norm_boxes, version_index_to_str
import mmcv
import numpy

def collate_fn(batch):
    collated_batch = {
        "images": [],
        "infos": [],
        "img_metas": []
    }
    for data in batch:
        collated_batch["images"].append(data["images"])
        
    collated_batch["nested_tensors"] = tensor_list_to_nested_tensor([_ for seq in collated_batch["images"] for _ in seq])
    shape = collated_batch["nested_tensors"].tensors.shape
    b = len(batch)
    t = len(collated_batch["images"][0])
    collated_batch["nested_tensors"].tensors = collated_batch["nested_tensors"].tensors.reshape(
        b, t, shape[1], shape[2], shape[3]
    )
    collated_batch["nested_tensors"].mask = collated_batch["nested_tensors"].mask.reshape(
        b, t, shape[2], shape[3]
    )

    # 修正infos和img_metas
    final_h, final_w = shape[2], shape[3]
    for data in batch:
        for info, meta in zip(data['infos'], data['img_metas']):
            meta["img_shape_beforecollate"] = torch.tensor([meta["img_shape"][0], meta["img_shape"][1]])
            meta["img_shape"] = torch.tensor([final_h, final_w])
            info["norm_boxes"] = rotate_boxes_to_norm_boxes(info["boxes"], (final_h, final_w), version_index_to_str(meta["version"]))

            heat_map = torch.zeros((final_h, final_w), dtype=info["heatmap"].dtype, device=info["heatmap"].device)
            heat_map[:info["heatmap"].shape[0], :info["heatmap"].shape[1]] = info["heatmap"]
            info["heatmap"] = heat_map

    for data in batch:
        collated_batch["infos"].append(data["infos"])
        collated_batch["img_metas"].append(data["img_metas"])   

    # collated_batch[]
    return collated_batch
