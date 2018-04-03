#!/bin/sh

PATH=$HOME/install/bin:$HOME/install/cmake/bin:$PATH
LD_LIBRARY_PATH=$HOME/install/cudnn-5.0/cuda-8.0/lib64:$LD_LIBRARY_PATH

WORKING_DIR=$1

# CMD="th main.lua --train --test --dataset=camvid --datapath=datasets/camvid --model=models/resnet18_seg.lua --learningRate=1 --batchSize=4 --testStep=20 --weightDecay=1e-5 --outputpath=results --imWidth=512 --imHeight=512 --labelWidth=512 --labelHeight=512 --maxIterations=100 --snapshotStep=20 --lrDecay=0.1 --lrDecayStep=20"

MODEL="models/resnet18closure4_seg.lua"

cp ${MODEL} ${WORKING_DIR}

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
  --imWidth=512 --imHeight=512 --labelWidth=512 --labelHeight=512 \
  --learningRate=0.1 --batchSize=4 \
  --testStep=20 --weightDecay=1e-5 \
  --epochs=200 --snapshotStep=20 \
  --lrDecay=0.5 --lrDecayStep=20"

${CMD}