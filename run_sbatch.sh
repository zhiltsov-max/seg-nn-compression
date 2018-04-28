#!/bin/sh
module load cuda/cuda-8.0 mkl2017

TIME=$(date -u +%s | sed "s/ /_/g")
OUTPUT_DIR=results/exec_${TIME}
OUTPUT_FILE=${OUTPUT_DIR}/log_${TIME}.txt
TARGET=exec.sh

if [ ! -d ${OUTPUT_DIR} ]; then
    mkdir -p ${OUTPUT_DIR}
fi

cp "${TARGET}" "${OUTPUT_DIR}/exec.sh"

sbatch --time=12:00:00 -p gpu -N 1 -o ${OUTPUT_FILE} ${TARGET} "${OUTPUT_DIR}"