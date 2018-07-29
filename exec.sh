#!/bin/sh

PATH=$HOME/install/bin:$HOME/install/cmake/bin:$PATH
LD_LIBRARY_PATH=$HOME/install/cudnn-5.0/cuda-8.0/lib64:$LD_LIBRARY_PATH

WORKING_DIR=$1

MODELS_DIR="models"
MODEL="resnet18conc_seg.lua"

cp "${MODELS_DIR}/${MODEL}" ${WORKING_DIR}

SNAPSHOT_DIR="${WORKING_DIR}/models"
mkdir -p "${SNAPSHOT_DIR}"

INFERENCE_DIR="${WORKING_DIR}/inference"
mkdir -p "${INFERENCE_DIR}"

CMD="th main.lua \
  --test \
  --dataset=camvid \
  --model=${MODEL} \
  --save="${SNAPSHOT_DIR}" \
  --inferencePath="${INFERENCE_DIR}" \
  --learningRate=1 --batchSize=1 \
  --testStep=100 --weightDecay=5e-4 \
  --epochs=2 --snapshotStep=100 \
  --lrDecay=0.02 \
  --optnet \
  "

${CMD}