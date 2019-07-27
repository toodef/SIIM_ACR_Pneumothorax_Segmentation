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

dvc run -d train.py \
  -o experiments/$EXP_DIR/seg/resnet18 -f resnet18seg.dvc \
  -d data/indices/train.npy \
  -d data/indices/val.npy \
  --no-exec python train.py -m resnet18

dvc run -d train.py \
  -o experiments/$EXP_DIR/seg/resnet34 -f resnet34seg.dvc \
  -d data/indices/train.npy \
  -d data/indices/val.npy \
  --no-exec python train.py -m resnet34

dvc run -d train.py \
  -o experiments/$EXP_DIR/class/resnet18 -f resnet18class.dvc \
  -d data/indices/train.npy \
  -d data/indices/val.npy \
  --no-exec python train.py -m resnet18

dvc run -d train.py \
  -o experiments/$EXP_DIR/class/resnet34 -f resnet34class.dvc \
  -d data/indices/train.npy \
  -d data/indices/val.npy \
  --no-exec python train.py -m resnet34

dvc run -d predict_model.py \
  -d experiments/$EXP_DIR/seg/resnet18 \
  -o out/resnet18_out.csv \
  --no-exec python predict_model.py -m resnet18 -o out/resnet18_out.csv

dvc run -d predict_model.py \
  -d experiments/$EXP_DIR/seg/resnet34 \
  -o out/resnet34_out.csv \
  --no-exec python predict_model.py -m resnet34 -o out/resnet34_out.csv

git add *.dvc
