# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
GPU=$1
PWD=$(cd `dirname $0` && pwd)
cd $PWD/../

PRETRAIN=/path/top/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth
EXP_DIR=/exp_dir/train_3ch
DET_DB=/path/to/train.json

mkdir -p ${EXP_DIR}
touch ${EXP_DIR}/output.log
cp $0 ${EXP_DIR}/



CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --nproc_per_node=2 \
    --master_port 29901 \
    --use_env main.py \
    --meta_arch motr \
    --dataset_file e2e_hsmot_8ch \
    --epoch 20 \
    --with_box_refine \
    --lr_drop 10 \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
    --pretrained ${PRETRAIN} \
    --batch_size 1 \
    --sample_mode random_interval \
    --sample_interval 3 \
    --sampler_lengths 5 \
    --merger_dropout 0 \
    --dropout 0 \
    --random_drop 0.1 \
    --fp_ratio 0.3 \
    --query_interaction_layer QIMv2 \
    --query_denoise 0.05 \
    --num_queries 10 \
    --det_db ${DET_DB} \
    --mot_path /path/to/dataset \
    --output_dir ${EXP_DIR} \
    --input_channels 3 \
    --npy2rgb \
    --num_workers 2 \
    | tee ${EXP_DIR}/output.log
