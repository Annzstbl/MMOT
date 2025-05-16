# 参数1是EXP_DIR
# 参数2是CUDA_VISIBLE_DIVECES
# 用法 sh predict_hsmot_8ch_normal.sh /data/users/litianhao/hsmot_code/workdir/motr/motr_r50_train_hsmot_8ch_4gpu 0
EXP_DIR=$1
GPU=$2

PWD=$(cd `dirname $0` && pwd)
cd $PWD/../

PRETRAIN=${EXP_DIR}/checkpoint0019.pth
touch ${EXP_DIR}/predict.log

echo ${EXP_DIR} >> ${EXP_DIR}/predict.log
echo ${GPU} >> ${EXP_DIR}/predict.log

cp $0 ${EXP_DIR}/

CUDA_VISIBLE_DEVICES=${GPU} python3 eval_hsmot_8ch.py \
    --pretrained ${PRETRAIN} \
    --output_dir ${EXP_DIR} \
    --mot_path /data/users/litianhao/data/HSMOT \
    --resume ${PRETRAIN} \
    --meta_arch motr \
    --use_checkpoint \
    --dataset_file e2e_hsmot_8ch \
    --epoch 20 \
    --with_box_refine \
    --lr_drop 10 \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
    --batch_size 1 \
    --sample_mode 'random_interval' \
    --sample_interval 3 \
    --sampler_steps 5 9 15 \
    --sampler_lengths 2 3 4 5 \
    --update_query_pos \
    --merger_dropout 0 \
    --dropout 0 \
    --random_drop 0.1 \
    --fp_ratio 0.3 \
    --query_interaction_layer 'QIM' \
    --extra_track_attn \
    --num_workers 0 \
    | tee -a ${EXP_DIR}/predict.log
