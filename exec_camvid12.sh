#!/bin/sh

PATH=$HOME/install/bin:$HOME/install/cmake/bin:$PATH
LD_LIBRARY_PATH=$HOME/install/cudnn-5.0/cuda-8.0/lib64:$LD_LIBRARY_PATH

WORKING_DIR=$1

MODELS_DIR="models"
MODEL="resnet50_seg.lua"

cp "${MODELS_DIR}/${MODEL}" ${WORKING_DIR}

SNAPSHOT_DIR="${WORKING_DIR}/models"
mkdir -p "${SNAPSHOT_DIR}"

INFERENCE_DIR="${WORKING_DIR}/inference"
mkdir -p "${INFERENCE_DIR}"

CMD="th main.lua \
  --train --test \
  --dataset=camvid12 --datapath=datasets/camvid12 --class_count=12 \
  --model=${MODEL} \
  --save="${SNAPSHOT_DIR}" \
  --inferencePath="${INFERENCE_DIR}" \
  --imWidth=480 --imHeight=384 \
  --learningRate=0.1 --batchSize=8 \
  --testStep=200 --weightDecay=5e-1 \
  --epochs=600 --snapshotStep=200 \
  --lrDecay=0.015 \
  --i=3 --optnet \
  "

${CMD}