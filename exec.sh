#!/bin/sh

PATH=$HOME/install/bin:$HOME/install/cmake/bin:$PATH
LD_LIBRARY_PATH=$HOME/install/cudnn-5.0/cuda-8.0/lib64:$LD_LIBRARY_PATH

WORKING_DIR=$1

MODELS_DIR="models"
MODEL="cnn2_seg.lua"

cp "${MODELS_DIR}/${MODEL}" ${WORKING_DIR}

SNAPSHOT_DIR="${WORKING_DIR}/models"
mkdir -p "${SNAPSHOT_DIR}"

INFERENCE_DIR="${WORKING_DIR}/inference"
mkdir -p "${INFERENCE_DIR}"

CMD="th main.lua \
  --train --test \
  --dataset=camvid --datapath=datasets/camvid \
  --model=${MODEL} \
  --save="${SNAPSHOT_DIR}" \
  --inferencePath="${INFERENCE_DIR}" \
  --imWidth=512 --imHeight=512 \
  --learningRate=4 --batchSize=12 \
  --testStep=0 --weightDecay=5e-4 \
  --epochs=1000 --snapshotStep=0 \
  --lrDecay=2.5 --lrDecayStep=33 \
  --i=1
  "

${CMD}