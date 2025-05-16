# 参数1是EXP_DIR
# 参数2是CUDA_VISIBLE_DIVECES
EXP_DIR=$1
GPU=$2

PWD=$(cd `dirname $0` && pwd)
cd $PWD/../
RESUME=${EXP_DIR}/checkpoint.pth

echo ${EXP_DIR} >> ${EXP_DIR}/predict.log
echo ${GPU} >> ${EXP_DIR}/predict.log

cp $0 ${EXP_DIR}/

CUDA_VISIBLE_DEVICES=${GPU} python3 submit_hsmot_8ch.py \
    --meta_arch motr \
    --dataset_file e2e_hsmot_8ch \
    --epoch 5 \
    --with_box_refine \
    --lr_drop 4 \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
    --resume ${RESUME} \
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
    --det_db /path/to/test.json \
    --use_checkpoint \
    --mot_path /path/to/dataset \
    --output_dir ${EXP_DIR} \
    --input_channels 8 \
    | tee -a ${EXP_DIR}/predict.log
