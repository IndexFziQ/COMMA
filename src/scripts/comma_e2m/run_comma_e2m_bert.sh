#!/usr/bin/env bash
PER_GPU_BATCH_SIZE=8
BATCH_SIZE=32
MAX_LEN=200
GC_STEP=4
GPU_NUM=1

lr_iter=(2)
epoch_iter=(3)
# seed_iter=(42 100 512 1024 2019)
SEED=42

TASK_NAME='mber.e2m'
NET_NAME='basic'
M_NAME='bert'

BASE_DIR=/your_file_path/comma
DATA_DIR=/your_data_path/ea_data
MODEL_PATH=/your_model_path/bert-large-uncased
TK_PATH=/your_model_path/bert-large-uncased

OUTPUT_BASE=/your_checkpoint_path/checkpoints_comma_bert_e2m


for LR_RATE in ${lr_iter[@]}; do
    for EPOCH_N in ${epoch_iter[@]}; do

        EXP_PRE="seed${SEED}"

        OUTPUT_DIR=${OUTPUT_BASE}/${TASK_NAME}_${NET_NAME}_${M_NAME}_gn${GPU_NUM}_bp${PER_GPU_BATCH_SIZE}_gc${GC_STEP}_lr${LR_RATE}_l${MAX_LEN}_e${EPOCH_N}_${EXP_PRE}
        mkdir -p ${OUTPUT_DIR}

        CUDA_VISIBLE_DEVICES=0 \
        python -u /your_file_path/COMMA/src/run_bert_comma_e2m.py \
        --task_name ${TASK_NAME} \
        --data_dir ${DATA_DIR} \
        --overwrite_output_dir \
        --overwrite_cache \
        --num_choices 5 \
        --model_type bert \
        --network_name ${NET_NAME} \
        --model_name_or_path ${MODEL_PATH} \
        --tokenizer_name_or_path ${TK_PATH} \
        --output_dir ${OUTPUT_DIR} \
        --log_file ${BASE_DIR}/log.${TASK_NAME}.${NET_NAME}.${M_NAME}.gn${GPU_NUM}.bp${PER_GPU_BATCH_SIZE}.gc${GC_STEP}.lr${LR_RATE}.l${MAX_LEN}.e${EPOCH_N}.${EXP_PRE}.out \
        --result_eval_file result.eval.${TASK_NAME}.${NET_NAME}.${M_NAME}.gn${GPU_NUM}.bp${PER_GPU_BATCH_SIZE}.gc${GC_STEP}.lr${LR_RATE}.l${MAX_LEN}.e${EPOCH_N}.${EXP_PRE}.txt \
        --tfboard_log_dir ${OUTPUT_DIR}/tfboard.event.out \
        --do_train \
        --do_eval \
        --do_test \
        --have_test_label \
        --do_lower_case \
        --max_seq_length ${MAX_LEN} \
        --train_batch_size ${BATCH_SIZE} \
        --per_gpu_train_batch_size ${PER_GPU_BATCH_SIZE} \
        --per_gpu_eval_batch_size 4 \
        --num_train_epochs ${EPOCH_N} \
        --max_steps -1 \
        --learning_rate ${LR_RATE}e-5 \
        --gradient_accumulation_steps ${GC_STEP} \
        --max_grad_norm 0.0 \
        --seed ${SEED}

    done
done
