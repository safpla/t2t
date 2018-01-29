#!/bin/bash
#PROBLEM=translate_ende_wmt32k
PROBLEM=word2ner_msra
MODEL=transformer
#HPARAMS=transformer_base_single_gpu
HPARAMS=word2ner_msra_hparams

DATA_DIR=$HOME/GitHub/t2t/t2t_data
TMP_DIR=$HOME/GitHub/t2t/t2t_datagen_msra
TRAIN_DIR=$HOME/GitHub/t2t/t2t_train/$PROBLEM/$MODEL-$HPARAMS
USR_DIR=$HOME/GitHub/t2t/t2t_usr
mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

export CUDA_VISIBLE_DEVICES=2
echo $CUDA_VISIBLE_DEVICES

# Generate data
t2t-datagen \
    --data_dir=$DATA_DIR \
    --tmp_dir=$TMP_DIR \
    --t2t_usr_dir=$USR_DIR \
    --problem=$PROBLEM
#exit 0
# Train
t2t-trainer \
    --data_dir=$DATA_DIR \
    --problems=$PROBLEM \
    --model=$MODEL \
    --hparams_set=$HPARAMS \
    --output_dir=$TRAIN_DIR \
    --eval_steps=200 \
    --save_checkpoints_secs=300 \
    --local_eval_frequency=600 \
    --t2t_usr_dir=$USR_DIR
#exit 0
    #--early_stopping=true \
# Decode

BEAM_SIZE=4
ALPHA=1
DECODE_FILE=$TMP_DIR/word_test
t2t-decoder \
    --data_dir=$DATA_DIR \
    --problems=$PROBLEM \
    --model=$MODEL \
    --output_dir=$TRAIN_DIR \
    --hparams_set=$HPARAMS \
    --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA,use_last_position_only=True" \
    --decode_from_file=$DECODE_FILE \
    --t2t_usr_dir=$USR_DIR
    #--decode_interactive \
exit 0
DECODE_FILE=$DATA_DIR/decode_this_event_part.txt
t2t-decoder \
    --data_dir=$DATA_DIR \
    --problems=$PROBLEM \
    --model=$MODEL \
    --output_dir=$TRAIN_DIR \
    --hparams_set=$HPARAMS \
    --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA,use_last_position_only=True" \
    --decode_from_file=$DECODE_FILE \
    --t2t_usr_dir=$USR_DIR

