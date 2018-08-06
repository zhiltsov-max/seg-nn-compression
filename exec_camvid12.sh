#!/bin/sh

PATH=$HOME/install/bin:$PATH
LD_LIBRARY_PATH=$HOME/install/cudnn-5.0/cuda-8.0/lib64:$LD_LIBRARY_PATH

WORKING_DIR=$1

MODELS_DIR="models"
MODEL="cnn1_seg.lua"

cp "${MODELS_DIR}/${MODEL}" ${WORKING_DIR}

SNAPSHOT_DIR="${WORKING_DIR}/models"
INFERENCE_DIR="${WORKING_DIR}/inference"

DEVICE_IDS=0 # This way Torch won't allocate extra memory on GPUs

CMD="th main.lua \
  --train \
  --test \
  --dataset=camvid12 \
  --model=${MODEL} \
  --save="${SNAPSHOT_DIR}" \
  --inferencePath="${INFERENCE_DIR}" \
  --imWidth=480 --imHeight=360 \
  --learningRate=1 --batchSize=10 \
  --testStep=100 --weightDecay=5e-4 \
  --epochs=400 --snapshotStep=200 \
  --lrDecay=0.015 \
  --manualSeed=42 --deterministic \
  --optnet \
"

export CUDA_VISIBLE_DEVICES=$DEVICE_IDS
${CMD}