import argparse
import os
import sys

import cv2
import numpy as np
import torch
from cv_utils.utils import mask2rle
from neural_pipeline import Predictor, FileStructManager
from tqdm import tqdm

from train_config.dataset import create_dataset
from train_config.train_config import BaseSegmentationTrainConfig, ResNet18SegmentationTrainConfig,\
    ResNet34SegmentationTrainConfig


def predict(config_types: [type(BaseSegmentationTrainConfig)], output_file: str, class_predicts: {}):
    dataset = create_dataset(is_test=True, include_negatives=False)

    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir) and not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    predictors = []
    for config_type in config_types:
        fsm = FileStructManager(base_dir=config_type.experiment_dir, is_continue=True)
        predictors.append(Predictor(config_type.create_model().cuda(), fsm=fsm))

    with open(output_file, 'w') as out_file:
        out_file.write("ImageId,EncodedPixels\n")
        out_file.flush()

        images_paths = dataset.get_items()
        for i, data in enumerate(tqdm(dataset)):
            cur_img_path = os.path.splitext(os.path.basename(images_paths[i]))[0]
            data = cv2.resize(data, (512, 512))
            img_tensor = torch.from_numpy(np.expand_dims(np.expand_dims(data.astype(np.float32), 0) / 128 - 1, 0)).cuda()

            res = []
            for predictor in predictors:
                res.append(np.squeeze(predictor.predict({'data': img_tensor}).data.cpu().numpy()))
            res = np.median(res, axis=0)
            res[res < 0.7] = 0

            if not class_predicts[cur_img_path]:
                rle = '-1'
            else:
                res = (res * 255).astype(np.uint8)
                res = cv2.resize(res, (1024, 1024))

                res[res > 0] = 255
                rle = mask2rle(res)

                if len(rle) < 1:
                    rle = '-1'

            out_file.write("{},{}\n".format(cur_img_path, rle))
            out_file.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict')
    parser.add_argument('-m', '--models', help='Models to predict', required=True, nargs='+',
                        choices=['resnet18', 'resnet34'])
    parser.add_argument('-o', '--out', type=str, help='Output file path', required=True)
    parser.add_argument('-c', '--class_predict', type=str, help='Classification file predict path', required=True)

    if len(sys.argv) < 3:
        print('Bad arguments passed', file=sys.stderr)
        parser.print_help(file=sys.stderr)
        exit(2)
    args = parser.parse_args()

    with open(args.class_predict, 'r') as class_predict_file:
        class_predicts = np.loadtxt(class_predict_file, delimiter=',', dtype=str)[1:]
    class_predicts = {v[0]: bool(int(float(v[1]))) for v in class_predicts}

    model_configs = {'resnet18': ResNet18SegmentationTrainConfig,
                     'resnet34': ResNet34SegmentationTrainConfig}

    configs = []
    for model in args.models:
        if model not in model_configs:
            raise Exception("Train pipeline doesn't implemented for model '{}'".format(args.model))
        configs.append(model_configs[model])

    predict(configs, args.out, class_predicts)
