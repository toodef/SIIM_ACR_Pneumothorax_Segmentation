if [%1]==[] goto usage

set EXP_DIR=%1

dvc run -d prepare_dataset.py ^
  -o data/indices/train_seg.npy -o data/indices/train_class.npy ^
  -o data/indices/val_seg.npy -o data/indices/val_class.npy ^
  -o data/indices/test_seg.npy -o data/indices/test_class.npy ^
  --no-exec python prepare_dataset.py

dvc run -d train.py ^
  -o experiments/%EXP_DIR%/class/resnet18 -f resnet18class.dvc ^
  -d data/indices/train.npy ^
  -d data/indices/val.npy ^
  --no-exec python train.py -m resnet18

dvc run -d train.py ^
  -o experiments/%EXP_DIR%/class/resnet34 -f resnet34class.dvc ^
  -d data/indices/train.npy ^
  -d data/indices/val.npy ^
  --no-exec python train.py -m resnet34

dvc run -d predict_model_class.py ^
  -d experiments/%EXP_DIR%/class/resnet18 ^
  -o out/class/resnet18_class_out.csv ^
  --no-exec python predict_model_class.py -m resnet18 -o out/class/resnet18_class_out.csv

dvc run -d predict_model_class.py ^
  -d experiments/%EXP_DIR%/class/resnet34 ^
  -o out/class/resnet34_class_out.csv ^
  --no-exec python predict_model_class.py -m resnet34 -o out/class/resnet34_class_out.csv

dvc run -d get_class_best_predict.py ^
  -o out/resnet18_class_out.csv ^
  -o out/resnet34_class_out.csv ^
  -d data/indices/test.npy ^
  -o out/class/class_best_predict.json ^
  --no-exec python get_class_best_predict.py

dvc run -d class_eval.py ^
  -d out/class/class_best_predict.json ^
  -o out/class/class_predict.csv ^
  --no-exec python class_eval.py

dvc run -d train.py ^
  -o experiments/%EXP_DIR%/seg/resnet18 -f resnet18seg.dvc ^
  -d data/indices/train.npy ^
  -d data/indices/val.npy ^
  --no-exec python train.py -m resnet18

dvc run -d train.py ^
  -o experiments/%EXP_DIR%/seg/resnet34 -f resnet34seg.dvc ^
  -d data/indices/train.npy ^
  -d data/indices/val.npy ^
  --no-exec python train.py -m resnet34

dvc run -d predict_model.py ^
  -d experiments/%EXP_DIR%/seg/resnet18 ^
  -d out/resnet18_class_out.csv ^
  -o out/resnet18_out.csv ^
  --no-exec python predict_model.py -m resnet18 -o out/resnet18_out.csv

dvc run -d predict_model.py ^
  -d experiments/%EXP_DIR%/seg/resnet34 ^
  -o out/resnet34_out.csv ^
  --no-exec python predict_model.py -m resnet34 -o out/resnet34_out.csv

git add *.dvc

:usage
@echo "Usage: ./utils/build_dvc_pipeline.sh <experiment dir name>"
@echo "   For example: ./utils/build_dvc_pipeline.sh exp1"
exit /B 1