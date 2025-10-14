#!/bin/bash

# 激活 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mmot1

ROOT_PATH=/data/users/litianhao/MMOT

# 设置路径
cd $ROOT_PATH/TrackByDetection/association
DET_ROOT=$ROOT_PATH/exp/yolo/train_3ch10/predict_test
IMG_ROOT=$ROOT_PATH/data/test/npy
WORK_DIR=$ROOT_PATH/exp
EVAL_SCRIPT=$ROOT_PATH/TrackEval/eval.sh

# 所有的 tracker 名称和对应的 expn 名
declare -A trackers
trackers=( 
  ["sort"]="sort" 
  ["bytetrack"]="bytetrack"
  ["ocsort"]="ocsort"
  ["botsort"]="botsort"
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