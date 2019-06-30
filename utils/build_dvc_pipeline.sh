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

dvc run -d predict_model.py \
  -d experiments/$EXP_DIR/resnet18 \
  -o out/resnet18_out.csv \
  --no-exec python predict_model.py -m resnet18 -o out/resnet18_out.csv

git add resnet18_out.csv.dvc

dvc run -d predict_model.py \
  -d experiments/$EXP_DIR/resnet34 \
  -o out/resnet34_out.csv \
  --no-exec python predict_model.py -m resnet34 -o out/resnet34_out.csv

git add resnet34_out.csv.dvc
