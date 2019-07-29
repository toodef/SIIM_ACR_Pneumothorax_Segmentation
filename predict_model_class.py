import argparse
import os
import sys

import cv2
import numpy as np
import torch
from neural_pipeline import Predictor, FileStructManager
from tqdm import tqdm

from train_config.dataset import create_dataset
from train_config.train_config import BaseClassificationTrainConfig, ResNet18ClassificationTrainConfig, \
    ResNet34ClassificationTrainConfig

__all__ = ['predict']


def predict(config_type: type(BaseClassificationTrainConfig), output_file: str):
    dataset = create_dataset(is_test=True, for_segmentation=False, include_negatives=True, indices_path='data/indices/test_class.npy')

    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir) and not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    fsm = FileStructManager(base_dir=config_type.experiment_dir, is_continue=True)
    predictor = Predictor(config_type.create_model().cuda(), fsm=fsm)

    with open(output_file, 'w') as out_file:
        out_file.write("ImageId,EncodedPixels\n")
        out_file.flush()

        images_paths = dataset.get_items()
        for i, data in enumerate(tqdm(dataset)):
            data = cv2.resize(data, (512, 512))
            img_tensor = torch.from_numpy(np.expand_dims(np.expand_dims(data.astype(np.float32), 0) / 128 - 1, 0)).cuda()
            res = np.squeeze(predictor.predict({'data': img_tensor}).data.cpu().numpy())

            out_file.write("{},{}\n".format(os.path.splitext(os.path.basename(images_paths[i]))[0], float(res)))
            out_file.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict')
    parser.add_argument('-m', '--model', type=str, help='Model to predict', required=True, choices=['resnet18', 'resnet34'])
    parser.add_argument('-o', '--out', type=str, help='Output file path', required=True)

    if len(sys.argv) < 3:
        print('Bad arguments passed', file=sys.stderr)
        parser.print_help(file=sys.stderr)
        exit(2)
    args = parser.parse_args()

    if args.model == 'resnet18':
        predict(ResNet18ClassificationTrainConfig, args.out)
    elif args.model == 'resnet34':
        predict(ResNet34ClassificationTrainConfig, args.out)
    else:
        raise Exception("Train pipeline doesn't implemented for model '{}'".format(args.model))
