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
from train_config.train_config import BaseTrainConfig, ResNet18TrainConfig, ResNet34TrainConfig


def predict(config_type: BaseTrainConfig, output_file: str):
    # dataset = create_dataset(is_test=False, indices_path='data/indices/train.npy')
    dataset = create_dataset(is_test=True)

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
            # img = data['data'].copy()
            # target = data['target'].copy()
            data = cv2.resize(data, (512, 512))
            img_tensor = torch.from_numpy(np.expand_dims(np.expand_dims(data.astype(np.float32), 0) / 128 - 1, 0)).cuda()
            res = np.squeeze(predictor.predict({'data': img_tensor}).data.cpu().numpy())
            res[res < 0.7] = 0

            if res[res > 0].size < 101:
                rle = -1
            else:
                res = (res * 255).astype(np.uint8)
                res = cv2.resize(res, (1024, 1024))

                # res_cntrs, _ = cv2.findContours(res, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                # target_cntrs, _ = cv2.findContours((target * 255).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                #
                # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                # img = cv2.drawContours(img, res_cntrs, -1, (0, 0, 255))
                # img = cv2.drawContours(img, target_cntrs, -1, (0, 255, 0))

                # cv2.imshow("img", img)
                # cv2.waitKey(0)

                # res = cv2.flip(res, 1)
                # res = cv2.rotate(res, cv2.ROTATE_90_COUNTERCLOCKWISE)

                res[res > 0] = 255
                rle = mask2rle(res)

            out_file.write("{},{}\n".format(os.path.splitext(os.path.basename(images_paths[i]))[0], rle))
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
        predict(ResNet18TrainConfig, args.out)
    elif args.model == 'resnet34':
        predict(ResNet34TrainConfig, args.out)
    else:
        raise Exception("Train pipeline doesn't implemented for model '{}'".format(args.model))
