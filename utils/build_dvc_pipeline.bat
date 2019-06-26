if [%1]==[] goto usage

set EXP_DIR=%1

dvc run -d prepare_dataset.py ^
  -o data/indices/train_indices.npy ^
  -o data/indices/val_indices.npy ^
  --no-exec python prepare_dataset.py

dvc run -d train.py -m experiments/$EXP_DIR/monitors/metrics_log/metrics.json ^
  -o experiments/%EXP_DIR%/monitors/metrics_log/metrics_log.json ^
  -o experiments/%EXP_DIR%/checkpoints/last/last_checkpoint.zip ^
  -o experiments/%EXP_DIR%/checkpoints/best/best_checkpoint.zip ^
  -d data/indices/train_indices.npy ^
  -d data/indices/val_indices.npy ^
  --no-exec python src/train.py

git add last_checkpoint.zip.dvc

dvc run -d predict.py ^
  -d experiments/%EXP_DIR%/checkpoints/last/last_checkpoint.zip ^
  -d experiments/%EXP_DIR%/checkpoints/best/best_checkpoint.zip ^
  --no-exec python predict.py

git add Dvcfile

:usage
@echo "Usage: ./utils/build_dvc_pipeline.sh <experiment dir name>"
@echo "   For example: ./utils/build_dvc_pipeline.sh exp1"
exit /B 1