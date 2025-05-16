from yolox.tracker.byte_tracker_rotate import BYTETracker
from yolox.sort_tracker.sort_rotated import Sort
from trackers.ocsort_tracker.ocsort_rotate import OCSort
from trackers.bot_sort_tracker.bot_sort_rotate import BoTSORT
import argparse
import os
import numpy as np
from mmcv.ops import box_iou_rotated
from tqdm import tqdm


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "--local_rank", default=0, type=int, help="local rank for dist training"
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    parser.add_argument(
        "--test",
        dest="test",
        default=False,
        action="store_true",
        help="Evaluating on test-dev set.",
    )
    parser.add_argument(
        "--speed",
        dest="speed",
        default=False,
        action="store_true",
        help="speed test only.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    # # det args
    # parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    # parser.add_argument("--conf", default=0.01, type=float, help="test conf")
    # parser.add_argument("--nms", default=0.7, type=float, help="test nms threshold")
    # parser.add_argument("--tsize", default=None, type=int, help="test img size")
    # parser.add_argument("--seed", default=None, type=int, help="eval seed")

    # tracking args


    parser.add_argument("--tracker", type=str, help="bytetrack, sort")
    # ocsort
    parser.add_argument("--use_byte", default=False, action="store_true")
    parser.add_argument('--oc_track_thresh', type=float, default=0.6)

    ## bytetrack
    parser.add_argument("--track_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
    parser.add_argument("--min-box-area", type=float, default=0, help='filter out tiny boxes') 
    # parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")

    ## sort
    parser.add_argument("--det_thresh", type=float, default=0.4)
    parser.add_argument("--max_age", type=int, default=30)
    parser.add_argument("--min_hits", type=int, default=3)
    parser.add_argument("--iou_threshold", type=float, default=0.3)

    # file path
    parser.add_argument("--img_root", type=str, help="path to images")
    parser.add_argument("--det_root", type=str, help="path to detections")

    parser.add_argument("--workdir", type=str, help="workdir relative path to det_root")
    parser.add_argument("--vid_white_list", type=str, default=None, help="video white list")

    # botsrot
    parser.add_argument("--bot_track_high_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--bot_track_low_thresh", default=0.1, type=float, help="lowest detection threshold valid for tracks")
    parser.add_argument("--bot_new_track_thresh", default=0.7, type=float, help="new track thresh")
    parser.add_argument("--bot_track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--bot_match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--bot_with_reid", default=False, action="store_true", help="use reid for tracking")
    parser.add_argument("--bot_cmc_method", default="sparseOptFlow", type=str, help="method for cmc")
    parser.add_argument("--bot_name", default = None, type=str, help="seq name")
    parser.add_argument("--bot_ablation", default = None, type=str, help="seq name")



    return parser


INFO_IMGS = [900, 1200]
IMG_SIZE = [900, 1200]

def load_imgs_and_dets(img_path, det_path):
    '''
        返回图片列表和检测结果列表
        保证image_list 和 dets_list的长度一致
        检测结果格式：[[x1, y1, x2, y2, x3, y3, x4, y4, cls, score], ...]
    '''
    # Load images
    assert os.path.exists(img_path), f"file not found: {img_path}"
    img_list = os.listdir(img_path)
    img_list.sort()

    # Load dets
    assert os.path.exists(det_path), f"file not found: {det_path}"
    det_list = os.listdir(det_path)
    det_list.sort()

    assert len(img_list) == len(det_list), "The number of images and detections should be the same."

    dets = []
    for det_file in det_list:
        with open(os.path.join(det_path, det_file), "r") as f:
            lines = f.readlines()
            det = []
            for line in lines:
                det.append(list(map(float, line.strip().split(" "))))
            dets.append(det) 
    return img_list, dets


def result2str(frame, track_id, xyxyxyxy, score, cls):
    '''
        frame, track_id, xyxyxyxy, score, cls
    '''

    # xyxyxyxy的每个值保留3位小数
    ret = f"{int(frame)} {int(track_id)} {xyxyxyxy[0]:.3f} {xyxyxyxy[1]:.3f} {xyxyxyxy[2]:.3f} {xyxyxyxy[3]:.3f} {xyxyxyxy[4]:.3f} {xyxyxyxy[5]:.3f} {xyxyxyxy[6]:.3f} {xyxyxyxy[7]:.3f} {score:.3f} {int(cls)} -1"
    return ret.replace(" ", ",")


def track_one_video_bytetrack(args, img_path, det_path, result_txt, vid_name):

    BYTETracker._count = 0
    
    # Load tracker
    tracker = BYTETracker(args)

    # Load images and detections
    image_list, dets_list = load_imgs_and_dets(img_path, det_path)

    txt = []
    # Track
    for index, dets in enumerate(tqdm(dets_list, desc=vid_name)):
        dets_np = np.array(dets)
        if len(dets_np) == 0:
            dets_np = np.zeros((0, 10))

        online_targets = tracker.update(dets_np, INFO_IMGS, IMG_SIZE)
        # print(online_targets)
        for targets in online_targets:
            xyxyxyxy = targets.xyxyxyxy
            frame = index +1# 从1开始
            score = targets.score
            cls = targets.cls
            track_id = targets.track_id
            txt.append(result2str(frame, track_id, xyxyxyxy, score, cls))
    
    with open(result_txt, "w") as f:
        f.write("\n".join(txt))
    print("Results saved to ", result_txt)


def track_one_video_sort(args, img_path, det_path, result_txt, vid_name):

    # Load tracker
    tracker = Sort(args.det_thresh, args.max_age, args.min_hits, args.iou_threshold)

    # Load images and detections
    image_list, dets_list = load_imgs_and_dets(img_path, det_path)

    txt = []
    # Track
    for index, dets in enumerate(tqdm(dets_list, desc=vid_name)):
        dets_np = np.array(dets)
        if len(dets_np) == 0:
            dets_np = np.zeros((0, 10))

        online_targets = tracker.update(dets_np, INFO_IMGS, IMG_SIZE)
        # print(online_targets)
        for targets in online_targets:
            xyxyxyxy = targets[:8]
            frame = index+1
            cls = targets[8]
            score = targets[9]
            track_id = targets[10]
            txt.append(result2str(frame, track_id, xyxyxyxy, score, cls))
    
    with open(result_txt, "w") as f:
        f.write("\n".join(txt))
    print("Results saved to ", result_txt)

def track_one_video_oc_sort(args, img_path, det_path, result_txt, vid_name):

    # Load tracker
    tracker = OCSort(args.oc_track_thresh, args.max_age, args.min_hits, args.iou_threshold, use_byte=args.use_byte)

    # Load images and detections
    image_list, dets_list = load_imgs_and_dets(img_path, det_path)

    txt = []
    # Track
    for index, dets in enumerate(tqdm(dets_list, desc=vid_name)):
        dets_np = np.array(dets)
        if len(dets_np) == 0:
            dets_np = np.zeros((0, 10))

        online_targets = tracker.update(dets_np, INFO_IMGS, IMG_SIZE)
        # print(online_targets)
        for targets in online_targets:
            xyxyxyxy = targets[:8]
            frame = index+1
            cls = targets[8]
            score = targets[9]
            track_id = targets[10]
            txt.append(result2str(frame, track_id, xyxyxyxy, score, cls))
    
    with open(result_txt, "w") as f:
        f.write("\n".join(txt))
    print("Results saved to ", result_txt)

def track_one_video_bot_sort(args, img_path, det_path, result_txt, vid_name):

    # Load images and detections
    image_list, dets_list = load_imgs_and_dets(img_path, det_path)

    # find keys in args which start with bot_, and construct a new args
    args_bot = {}
    for key in args.__dict__:
        if key.startswith("bot_"):
            args_bot[key[4:]] = args.__dict__[key]
    # turn to namespace
    args_bot = argparse.Namespace(**args_bot)


    # Load tracker
    tracker = BoTSORT(args_bot)

    txt = []
    # Track
    for index, dets in enumerate(tqdm(dets_list, desc=vid_name)):

        img = np.load(os.path.join(img_path, image_list[index])) #shape [H, W, 8]
        img = img[:,:, [1, 2, 4]]
        img = np.ascontiguousarray(img)

        dets_np = np.array(dets)
        if len(dets_np) == 0:
            dets_np = np.zeros((0, 10))

        online_targets = tracker.update(dets_np, img)
        # print(online_targets)
        for targets in online_targets:
            xyxyxyxy = targets.xyxyxyxy
            frame = index +1# 从1开始
            score = targets.score
            cls = targets.cls
            track_id = targets.track_id
            txt.append(result2str(frame, track_id, xyxyxyxy, score, cls))
    
    with open(result_txt, "w") as f:
        f.write("\n".join(txt))
    print("Results saved to ", result_txt)

if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()

    img_root = args.img_root
    det_root = args.det_root

    workdir = os.path.join(args.workdir, args.tracker, args.experiment_name)
    track_dir = os.path.join(workdir, 'track')
    os.makedirs(track_dir, exist_ok=True)

    tracker = args.tracker.lower()

    if args.vid_white_list is not None:
        vid_white_list = args.vid_white_list.split(",")
    else:
        vid_white_list = []

    for vid in os.listdir(img_root):
        if len(vid_white_list) > 0 and not any([v in vid for v in vid_white_list]):
            print(f'VID_WHITE_LIST logging: skip {vid}')
            continue

        img_path = os.path.join(img_root, vid)
        det_path = os.path.join(det_root, vid)
        result_txt = os.path.join(track_dir, vid + ".txt")

        if tracker == 'bytetrack':
            track_one_video_bytetrack(args, img_path, det_path, result_txt, vid)
        elif tracker == 'sort':
            track_one_video_sort(args, img_path, det_path, result_txt, vid)
        elif tracker == 'ocsort':
            track_one_video_oc_sort(args, img_path, det_path, result_txt, vid)
        elif tracker == 'botsort':
            track_one_video_bot_sort(args, img_path, det_path, result_txt, vid)
        else:
            raise NotImplementedError(f"Tracker {tracker} not implemented.")







