# Copyright (c) Ruopeng Gao. All Rights Reserved.
# ------------------------------------------------------------------------
import os

import torch
import torch.distributed
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel as DDP

from models import build_model
from models.utils import load_checkpoint
from log.logger import Logger, ProgressLogger
from log.log import Metrics
from utils.utils import is_distributed, distributed_rank, yaml_to_dict, \
    distributed_world_size, is_main_process, distributed_world_rank
from submit_engine import submit_one_seq, get_seq_names
import sys
from hsmot.eval.validator import PredictValidator, val_folder



def evaluate(config: dict, logger: Logger):
    """
    Evaluate a model.

    Args:
        config:
        logger:
    Returns:

    """
    model_config = yaml_to_dict(path=config["INFERENCE_CONFIG_PATH"])
    model = build_model(config=model_config)
    load_checkpoint(model, path=config["INFERENCE_MODEL"])

    # If DDP:
    if is_distributed():
        model = DDP(model, device_ids=[distributed_rank()])

    if config["INFERENCE_GROUP"] is not None:
        eval_outputs_dir = os.path.join(config["OUTPUTS_DIR"], config["MODE"], config["INFERENCE_GROUP"],
                                        config["INFERENCE_SPLIT"],
                                        f'{config["INFERENCE_MODEL"].split("/")[-1][:-4]}')
    else:
        eval_outputs_dir = os.path.join(config["OUTPUTS_DIR"], config["MODE"], "default", config["INFERENCE_SPLIT"],
                                        f'{config["INFERENCE_MODEL"].split("/")[-1][:-4]}')
    eval_metrics = evaluate_one_epoch(
        config=config,
        model=model,
        logger=logger,
        dataset=config["INFERENCE_DATASET"],
        data_split=config["INFERENCE_SPLIT"],
        outputs_dir=eval_outputs_dir,
        only_detr=config["INFERENCE_ONLY_DETR"]
    )
    eval_metrics.sync()
    logger.save_metrics(
        metrics=eval_metrics,
        prompt=f"[Eval Checkpoint '{config['INFERENCE_MODEL']}'] ",
        fmt="{global_average:.4f}",
        statistic=None
    )

    return


@torch.no_grad()
def evaluate_one_epoch(config: dict, model: nn.Module,
                       logger: Logger, dataset: str, data_split: str,
                       outputs_dir: str, only_detr: bool = False):
    model.eval()
    metrics = Metrics()
    device = config["DEVICE"]

    all_seq_names = get_seq_names(data_root=config["DATA_ROOT"], dataset=dataset, data_split=data_split)
    seq_names = [all_seq_names[_] for _ in range(len(all_seq_names))
                 if _ % distributed_world_size() == distributed_rank()]

    if len(seq_names) > 0:
        for seq in seq_names:
            submit_one_seq(
                model=model, dataset=dataset,
                seq_dir=os.path.join(config["DATA_ROOT"], dataset, data_split, 'npy', seq),
                only_detr=only_detr, max_temporal_length=config["MAX_TEMPORAL_LENGTH"],
                outputs_dir=outputs_dir,
                det_thresh=config["DET_THRESH"],
                newborn_thresh=config["DET_THRESH"] if "NEWBORN_THRESH" not in config else config["NEWBORN_THRESH"],
                area_thresh=config["AREA_THRESH"], id_thresh=config["ID_THRESH"],
                image_max_size=config["INFERENCE_MAX_SIZE"] if "INFERENCE_MAX_SIZE" in config else 1333,
                inference_ensemble=config["INFERENCE_ENSEMBLE"] if "INFERENCE_ENSEMBLE" in config else 0,
                npy2rgb=config["NPY2RGB"]
            )
    else:
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
            npy2rgb=config["NPY2RGB"]
        )

    tracker_dir = os.path.join(outputs_dir, '..', '..')
    trackers_name = outputs_dir.split("/")[-2]
    trackers_subfolder = outputs_dir.split("/")[-1]


    dataset_dir = os.path.join(config["DATA_ROOT"], dataset.replace('_8ch',''))
    gt_dir = os.path.join(dataset_dir, data_split, 'mot')
    img_dir = os.path.join(dataset_dir, data_split, 'npy')

    if is_distributed():
        torch.distributed.barrier()

    if is_main_process():
        if only_detr:
            val_lines = val_folder(gt_folder=gt_dir, pred_folder=os.path.join(tracker_dir, trackers_name, trackers_subfolder))
            logger.save_log_to_file('\n'.join(val_lines))

        else:
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            os_flag = os.system(
                f"{sys.executable} {current_file_dir}/../TrackEval/scripts/run_hsmot_8ch.py " 
                f"--USE_PARALLEL False "
                f"--METRICS HOTA CLEAR Identity " 
                f"--GT_FOLDER {gt_dir} "
                f"--TRACKERS_FOLDER {tracker_dir} "
                f"--TRACKERS_TO_EVAL {trackers_name} "
                f"--TRACKER_SUB_FOLDER {trackers_subfolder} "
                f"--IMG_FOLDER {img_dir} "
            )
            assert os_flag == 0, "TrackEval failed to run."

    if is_distributed():
        torch.distributed.barrier()
    if not only_detr:
        # Get eval Metrics:
        eval_metric_path = os.path.join(tracker_dir, trackers_name, 'eval', "cls_comb_det_av_summary.txt")
        eval_metrics_dict = get_eval_metrics_dict(metric_path=eval_metric_path)
        metrics["HOTA"].update(eval_metrics_dict["HOTA"])
        metrics["DetA"].update(eval_metrics_dict["DetA"])
        metrics["AssA"].update(eval_metrics_dict["AssA"])
        metrics["DetPr"].update(eval_metrics_dict["DetPr"])
        metrics["DetRe"].update(eval_metrics_dict["DetRe"])
        metrics["AssPr"].update(eval_metrics_dict["AssPr"])
        metrics["AssRe"].update(eval_metrics_dict["AssRe"])
        metrics["MOTA"].update(eval_metrics_dict["MOTA"])
        metrics["IDF1"].update(eval_metrics_dict["IDF1"])

    return metrics


def get_eval_metrics_dict(metric_path: str):
    with open(metric_path) as f:
        metric_names = f.readline()[:-1].split(" ")
        metric_values = f.readline()[:-1].split(" ")
    metrics = {
        n: float(v) for n, v in zip(metric_names, metric_values)
    }
    return metrics
