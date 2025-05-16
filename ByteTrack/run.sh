#!/bin/bash

# 激活 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate yolo

# 设置路径
cd $ROOT_PATH/ByteTrack
DET_ROOT=/path/to/yolo_exp/
IMG_ROOT=/path/to/dataset/test/npy
WORK_DIR=/path/to/output
EVAL_SCRIPT=$ROOT_PATH/TrackEval/eval.sh

# 所有的 tracker 名称和对应的 expn 名
declare -A trackers
trackers=( 
  ["sort"]="sort_yolo1gpu3ch_detth01" 
  ["bytetrack"]="bytetrack_yolo1gpu3ch_detth01"
  ["ocsort"]="ocsort_yolo1gpu3ch_detth01"
  ["botsort"]="botsort_yolo1gpu3ch_detth01"
)
# 每个 tracker 对应的额外参数
declare -A tracker_args
tracker_args=(
  ["sort"]="--det_thresh 0.1"
  ["bytetrack"]="--track_thresh 0.1"
  ["ocsort"]="--oc_track_thresh 0.1"
  ["botsort"]="--bot_new_track_thresh 0.6"
)

# 执行每个 tracker
for tracker in "${!trackers[@]}"; do
  expn="${trackers[$tracker]}"
  args="${tracker_args[$tracker]}"
  echo ">>> Running tracker: $tracker, experiment: $expn"
  
  python track_by_dets.py \
    -expn "$expn" \
    --tracker "$tracker" \
    --img_root "$IMG_ROOT" \
    --det_root "$DET_ROOT" \
    --workdir "$WORK_DIR" \
    $args

  echo ">>> Evaluating $tracker..."
  sh "$EVAL_SCRIPT" "$tracker/$expn" track
done