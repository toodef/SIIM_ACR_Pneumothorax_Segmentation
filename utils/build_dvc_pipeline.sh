#!/usr/bin/env bash

if [[ "$#" -ne 1 ]]; then
    echo "Usage: ./utils/build_dvc_pipeline.sh <experiment dir name>"
    echo "   For example: ./utils/build_dvc_pipeline.sh exp1"
    exit 0
fi

EXP_DIR="$1"

dvc run -d prepare_dataset.py \
  -o data/indices/train.npy \
  -o data/indices/val.npy \
  --no-exec python prepare_dataset.py

git add train.npy.dvc

dvc run -d train.py \
  -o experiments/$EXP_DIR/resnet18 \
  -d data/indices/train.npy \
  -d data/indices/val.npy \
  --no-exec python train.py -m resnet18

git add resnet18.dvc

dvc run -d train.py \
  -o experiments/$EXP_DIR/resnet34 \
  -d data/indices/train.npy \
  -d data/indices/val.npy \
  --no-exec python train.py -m resnet34

git add resnet34.dvc

dvc run -d predict.py \
  -d experiments/$EXP_DIR \
  -d data/indices/test.npy \
  --no-exec python predict.py

git add Dvcfile