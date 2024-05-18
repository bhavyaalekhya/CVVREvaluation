#!/bin/bash

DIR="ckpt/timechat"
MODEL_DIR=${DIR}/timechat_7b.pth

TASK='captaincook'
ANNO_DIR='./captain_cook_dvc.json'
VIDEO_DIR='/data/rohith/captain_cook/videos/gopro/resolution_360p/8_16_360p.mp4'
DATASET='captaincook'
SPLIT='test'
PROMPT_FILE="./captain_cook_description_zeroshot.txt"
GT_FILE="${ANNO_DIR}/${SPLIT}.caption_coco_format.json"
#ASR_DIR='data/YouCook2-BB/YouCook2_asr_denseCap/whisper_outputs_with_time/small.en.cleaned/'

NUM_FRAME=30
OUTPUT_DIR=${DIR}/${TASK}

python cvvr_evaluation_suite/evaluate_cc.py --anno_path ${ANNO_DIR} --video_path ${VIDEO_DIR} \
--task ${TASK} --dataset ${DATASET} --output_dir ${OUTPUT_DIR} --split ${SPLIT} --num_frames ${NUM_FRAME} --batch_size 16 \
--prompt_file ${PROMPT_FILE} --timechat_model_path ${MODEL_DIR} \
#--asr --asr_path ${ASR_DIR}
#--debug
