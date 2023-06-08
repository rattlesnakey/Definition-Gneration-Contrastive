#!/bin/bash
set -ve

TRAIN_BATCH_SIZE=16
TEST_BATCH_SIZE=8
MAX_LEN=120
BEAM_SIZE=15

MODEL=pretrained_models/t5
DATASET=wordnet

LR=3e-4
DATA_DIR=data/$DATASET
WARMUPS=2000
CONTRASTIVE_RATIO=0.8
POOLING_METHOD=max #! average、max、None
SHUFFLE=yes
CUDA=1

VAL_CHECK_INTER=10

LOG_DIR=logs

mkdir -p $T5_OUTPUT_DIR
mkdir -p $LOG_DIR

export CUDA_VISIBLE_DEVICES=$CUDA


train_pipeline(){
    python -u src/finetune.py --model_name_or_path $MODEL \
        --use_warmup 1 \
        --data_dir $DATA_DIR --learning_rate $LR \
        --early_stopping_patience $1 --train_batch_size $TRAIN_BATCH_SIZE \
        --eval_batch_size $TEST_BATCH_SIZE --output_dir $2 \
        --max_source_length $MAX_LEN --max_target_length $MAX_LEN \
        --num_train_epochs $3 --gpus 1 --do_train --do_predict \
        --task $4 \
        --pooling_method $POOLING_METHOD \
        --weight_decay 0.001 \
        --warmup_steps $WARMUPS \
        --num_workers 10 \
        --shuffle $SHUFFLE \
        --contrastive_ratio $5 \
        --test_dataset test 2>&1 | tee $LOG_DIR/$6-training.log
}

#! first stage
TASK=def-gen
NUM_EPOCHS_STAGE_ONE=50
EARLY_STOP_STAGE_ONE=10
T5_OUTPUT_DIR_STAGE_ONE=output/$DATASET-def-gen
CONTRASTIVE_RATIO_STAGE_ONE=0.0
STAGE=stage-one

train_pipeline "$EARLY_STOP_STAGE_ONE" "$T5_OUTPUT_DIR_STAGE_ONE" "$NUM_EPOCHS_STAGE_ONE" "$TASK" "$CONTRASTIVE_RATIO_STAGE_ONE" "$STAGE"

#! second stage
TASK=def-gen-with-contras
NUM_EPOCHS_STAGE_TWO=50
EARLY_STOP_STAGE_TWO=10
CKPT_PATH=$T5_OUTPUT_DIR_STAGE_ONE
T5_OUTPUT_DIR_STAGE_TWO=output/$DATASET-def-gen-with-contras
CONTRASTIVE_RATIO_STAGE_ONE=0.8
STAGE=stage-two

train_pipeline "$EARLY_STOP_STAGE_TWO" "$T5_OUTPUT_DIR_STAGE_TWO" "$NUM_EPOCHS_STAGE_TWO" "$TASK" "$CONTRASTIVE_RATIO_STAGE_TWO" "$STAGE"



