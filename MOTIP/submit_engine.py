# Copyright (c) RuopengGao. All Rights Reserved.
# About:
import os
import json

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data import DataLoader
from data.seq_dataset import SeqDataset
from utils.nested_tensor import tensor_list_to_nested_tensor
from models.utils import get_model
from utils.box_ops import box_cxcywh_to_xyxy
from collections import deque
from structures.instances import Instances
from structures.ordered_set import OrderedSet
from log.logger import Logger
from utils.utils import yaml_to_dict, is_distributed, distributed_rank, distributed_world_size
from models import build_model
from models.utils import load_checkpoint
from hsmot.datasets.pipelines.channel import rotate_norm_boxes_to_boxes
from hsmot.mmlab.hs_mmrotate import obb2poly
from tqdm import tqdm


def submit(config: dict, logger: Logger):
    """
    Submit a model for a specific dataset.
    :param config:
    :param logger:
    :return:
    """
    if config["INFERENCE_CONFIG_PATH"] is None:
        model_config = config
    else:
        model_config = yaml_to_dict(path=config["INFERENCE_CONFIG_PATH"])
    model = build_model(config=model_config)
    load_checkpoint(model, path=config["INFERENCE_MODEL"])

    if is_distributed():
        model = DDP(model, device_ids=[distributed_rank()])

    if config["INFERENCE_GROUP"] is not None:
        submit_outputs_dir = os.path.join(config["OUTPUTS_DIR"], config["INFERENCE_GROUP"], config["MODE"], )
                                        #   config["INFERENCE_SPLIT"],
                                        #   f'{config["INFERENCE_MODEL"].split("/")[-1][:-4]}')
    else:
        submit_outputs_dir = os.path.join(config["OUTPUTS_DIR"], "default", config["MODE"])

    # 需要调度整个 submit 流程
    submit_one_epoch(
        config=config,
        model=model,
        logger=logger,
        dataset=config["INFERENCE_DATASET"],
        data_split=config["INFERENCE_SPLIT"],
        outputs_dir=submit_outputs_dir,
        only_detr=config["INFERENCE_ONLY_DETR"]
    )

    logger.print(log=f"Finish submit process for model '{config['INFERENCE_MODEL']}' on the {config['INFERENCE_DATASET']} {config['INFERENCE_SPLIT']} set, outputs are write to '{submit_outputs_dir}/.'")
    logger.save_log_to_file(
        log=f"Finish submit process for model '{config['INFERENCE_MODEL']}' on the {config['INFERENCE_DATASET']} {config['INFERENCE_SPLIT']} set, outputs are write to '{submit_outputs_dir}/.'",
        filename="log.txt",
        mode="a"
    )

    return


@torch.no_grad()
def submit_one_epoch(config: dict, model: nn.Module,
                     logger: Logger, dataset: str, data_split: str,
                     outputs_dir: str, only_detr: bool = False):
    model.eval()

    all_seq_names = get_seq_names(data_root=config["DATA_ROOT"], dataset=dataset, data_split=data_split)
    seq_names = [all_seq_names[_] for _ in range(len(all_seq_names))
                 if _ % distributed_world_size() == distributed_rank()]

    if len(seq_names) > 0:
        for seq in seq_names:
            submit_one_seq(
                model=model, dataset=dataset,
                seq_dir=os.path.join(config["DATA_ROOT"], dataset, data_split, 'npy', seq),
                only_detr=only_detr, 
                max_temporal_length=min(config["MAX_TEMPORAL_LENGTH"], max(config["SAMPLE_LENGTHS"])),
                outputs_dir=outputs_dir,
                det_thresh=config["DET_THRESH"],
                newborn_thresh=config["DET_THRESH"] if "NEWBORN_THRESH" not in config else config["NEWBORN_THRESH"],
                area_thresh=config["AREA_THRESH"], id_thresh=config["ID_THRESH"],
                image_max_size=config["INFERENCE_MAX_SIZE"] if "INFERENCE_MAX_SIZE" in config else 1333,
                inference_ensemble=config["INFERENCE_ENSEMBLE"] if "INFERENCE_ENSEMBLE" in config else 0,
            )
    else:   # fake submit, will not write any outputs.
        submit_one_seq(
            model=model, dataset=dataset,
            seq_dir=os.path.join(config["DATA_ROOT"], dataset, data_split, all_seq_names[0]),
            only_detr=only_detr, max_temporal_length=config["MAX_TEMPORAL_LENGTH"],
            outputs_dir=outputs_dir,
            det_thresh=config["DET_THRESH"],
            newborn_thresh=config["DET_THRESH"] if "NEWBORN_THRESH" not in config else config["NEWBORN_THRESH"],
            area_thresh=config["AREA_THRESH"], id_thresh=config["ID_THRESH"],
            image_max_size=config["INFERENCE_MAX_SIZE"] if "INFERENCE_MAX_SIZE" in config else 1333,
            fake_submit=True,
            inference_ensemble=config["INFERENCE_ENSEMBLE"] if "INFERENCE_ENSEMBLE" in config else 0,
        )

    if is_distributed():
        torch.distributed.barrier()

    return


@torch.no_grad()
def submit_one_seq(
            model: nn.Module, dataset: str, seq_dir: str, outputs_dir: str,
            only_detr: bool, max_temporal_length: int = 0,
            det_thresh: float = 0.5, newborn_thresh: float = 0.5, area_thresh: float = 100, id_thresh: float = 0.1,
            image_max_size: int = 1333,
            fake_submit: bool = False,
            inference_ensemble: int = 0,
            npy2rgb: bool = False,
        ):
    os.makedirs(outputs_dir, exist_ok=True)
    seq_dataset = SeqDataset(seq_dir=seq_dir, dataset=dataset, npy2rgb=npy2rgb,)
    seq_dataloader = DataLoader(seq_dataset, batch_size=1, num_workers=4, shuffle=False)
    # seq_name = seq_dir.split("/")[-1]
    seq_name = os.path.split(seq_dir)[-1]
    device = model.device
    current_id = 0
    ids_to_results = {}
    id_deque = OrderedSet()     # an ID deque for inference, the ID will be recycled if the dictionary is not enough.

    # Trajectory history:
    if only_detr:
        trajectory_history = None
    else:
        trajectory_history = deque(maxlen=max_temporal_length)

    if fake_submit:
        print(f"[Fake] Start >> Submit seq {seq_name.split('/')[-1]}, {len(seq_dataloader)} frames ......")
    else:
        print(f"Start >> Submit seq {seq_name.split('/')[-1]}, {len(seq_dataloader)} frames ......")

    txt_lines = []

    for i, (image, ori_image) in tqdm(enumerate(seq_dataloader), desc=f"Submit seq {seq_name.split('/')[-1]}"):
        ori_h, ori_w = ori_image.shape[1], ori_image.shape[2]
        img_h, img_w = image.shape[2], image.shape[3]
        frame = tensor_list_to_nested_tensor([image[0]]).to(device)
        frame_h, frame_w = frame.tensors.shape[2], frame.tensors.shape[3]
        assert frame_h==img_h, f"frame_h={frame_h} != img_h={img_h}"
        assert frame_w==img_w, f"frame_w={frame_w} != img_w={img_w}"

        detr_outputs = model(frames=frame)
        detr_logits = detr_outputs["pred_logits"]
        detr_scores = torch.max(detr_logits, dim=-1).values.sigmoid()
        detr_det_idxs = detr_scores > det_thresh        # filter by the detection threshold
        detr_det_logits = detr_logits[detr_det_idxs]
        detr_det_labels = torch.max(detr_det_logits, dim=-1).indices
        detr_det_boxes = detr_outputs["pred_boxes"][detr_det_idxs]
        detr_det_outputs = detr_outputs["outputs"][detr_det_idxs]   # detr output embeddings
        area_legal_idxs = (detr_det_boxes[:, 2] * ori_w * detr_det_boxes[:, 3] * ori_h) > area_thresh   # filter by area
        detr_det_outputs = detr_det_outputs[area_legal_idxs]
        detr_det_boxes = detr_det_boxes[area_legal_idxs]
        detr_det_logits = detr_det_logits[area_legal_idxs]
        detr_det_labels = detr_det_labels[area_legal_idxs]

        # De-normalize to target image size:
        # box_results = detr_det_boxes.cpu() * torch.tensor([ori_w, ori_h, ori_w, ori_h])
        # box_results = box_cxcywh_to_xyxy(boxes=box_results)
        box_results = rotate_norm_boxes_to_boxes(detr_det_boxes.cpu(), (img_h, img_w), version='le135')
        box_results = obb2poly(box_results)
        label_results = detr_det_labels.cpu()
        confs = torch.max(detr_det_logits, dim=-1).values.sigmoid().cpu()


        if only_detr is False:
            if len(box_results) > get_model(model).num_id_vocabulary:
                raise ValueError(f"[Carefully!] we only support {get_model(model).num_id_vocabulary} ids, "
                                    f"but get {len(box_results)} detections in seq {seq_name.split('/')[-1]} {i+1}th frame.")
                print(f"[Carefully!] we only support {get_model(model).num_id_vocabulary} ids, "
                      f"but get {len(box_results)} detections in seq {seq_name.split('/')[-1]} {i+1}th frame.")
                # 随机删除一些
                random_filter = torch.randperm(len(box_results))[:get_model(model).num_id_vocabulary]
                box_results = box_results[random_filter]
                detr_det_outputs = detr_det_outputs[random_filter]
                detr_det_logits = detr_det_logits[random_filter]
                detr_det_labels = detr_det_labels[random_filter]
                

        # Decoding the current objects' IDs
        if only_detr is False:
            assert max_temporal_length - 1 > 0, f"MOTIP need at least T=1 trajectory history, " \
                                                f"but get T={max_temporal_length - 1} history in Eval setting."
            current_tracks = Instances(image_size=(0, 0))
            current_tracks.boxes = detr_det_boxes
            current_tracks.outputs = detr_det_outputs
            current_tracks.ids = torch.tensor([get_model(model).num_id_vocabulary] * len(current_tracks),
                                              dtype=torch.long, device=current_tracks.outputs.device)
            current_tracks.confs = torch.max(detr_det_logits, dim=-1).values.sigmoid()
            trajectory_history.append(current_tracks)
            if len(trajectory_history) == 1:    # first frame, do not need decoding:
                newborn_filter = (trajectory_history[0].confs > newborn_thresh).reshape(-1, )   # filter by newborn
                trajectory_history[0] = trajectory_history[0][newborn_filter]

                box_results = box_results[newborn_filter.cpu()]
                label_results = label_results[newborn_filter.cpu()]
                confs = confs[newborn_filter.cpu()]

                ids = torch.tensor([current_id + _ for _ in range(len(trajectory_history[-1]))],
                                   dtype=torch.long, device=current_tracks.outputs.device)
                trajectory_history[-1].ids = ids
                for _ in ids:
                    ids_to_results[_.item()] = current_id
                    current_id += 1
                id_results = []
                for _ in ids:
                    id_results.append(ids_to_results[_.item()])
                    id_deque.add(_.item())
                id_results = torch.tensor(id_results, dtype=torch.long)
            else:
                ids, trajectory_history, ids_to_results, current_id, id_deque, boxes_keep = get_model(model).inference(
                    trajectory_history=trajectory_history,
                    num_id_vocabulary=get_model(model).num_id_vocabulary,
                    ids_to_results=ids_to_results,
                    current_id=current_id,
                    id_deque=id_deque,
                    id_thresh=id_thresh,
                    newborn_thresh=newborn_thresh,
                    inference_ensemble=inference_ensemble,
                )   # already update the trajectory history/ids_to_results/current_id/id_deque in this function
                id_results = []
                for _ in ids:
                    id_results.append(ids_to_results[_])
                id_results = torch.tensor(id_results, dtype=torch.long)
                if boxes_keep is not None:
                    box_results = box_results[boxes_keep.cpu()]
                    label_results = label_results[boxes_keep.cpu()]
                    confs = confs[boxes_keep.cpu()]

        else:   # only detr, ID is just +1 for each detection.
            id_results = torch.tensor([current_id + _ for _ in range(len(box_results))], dtype=torch.long)
            current_id += len(id_results)

        # Output to tracker file:
        if fake_submit is False:
            save_format = '{frame:6d},{id:6d},{x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f},{x3:.3f},{y3:.3f},{x4:.3f},{y4:.3f},{conf:.3f},{label:2d},-1\n'
            assert len(id_results) == len(box_results), f"Boxes and IDs should in the same length, " \
                                                        f"but get len(IDs)={len(id_results)} and " \
                                                        f"len(Boxes)={len(box_results)}"
            for obj_id, box , label, conf in zip(id_results, box_results, label_results, confs):
                obj_id = int(obj_id.item())
                x1, y1, x2, y2, x3, y3, x4, y4 = box.tolist()
                line = save_format.format(
                    frame=i + 1, id=obj_id, x1=x1, y1=y1, x2=x2, y2=y2, x3=x3, y3=y3, x4=x4, y4=y4, conf=conf, label=label
                )
                txt_lines.append(line)

    if fake_submit:
        print(f"[Fake] Finish >> Submit seq {seq_name.split('/')[-1]}. ")
    else:
        result_file_path = os.path.join(outputs_dir, f"{seq_name}.txt")
        with open(result_file_path, "w") as f:
            f.writelines(txt_lines)
        print(f"Finish >> Submit seq {seq_name.split('/')[-1]}. ")
    return


def get_seq_names(data_root: str, dataset: str, data_split: str):
    if dataset in ["DanceTrack", "SportsMOT", "MOT17", "MOT17_SPLIT"]:
        dataset_dir = os.path.join(data_root, dataset, data_split)
        return sorted(os.listdir(dataset_dir))
    elif dataset in ['hsmot_8ch']:
        dataset_dir = os.path.join(data_root, 'HSMOT', data_split, 'npy')
        return sorted(os.listdir(dataset_dir))
    else:
        raise NotImplementedError(f"Do not support dataset '{dataset}' for eval dataset.")
