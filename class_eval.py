import argparse
import itertools
import json
import os
import sys

import cv2
import numpy as np
import torch
from neural_pipeline import Predictor, FileStructManager
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from train_config.dataset import create_dataset
from train_config.train_config import BaseClassificationTrainConfig, ResNet18ClassificationTrainConfig, \
    ResNet34ClassificationTrainConfig


def predict_on_test_set(config_type: type(BaseClassificationTrainConfig)):
    dataset = create_dataset(is_test=False, include_negatives=True, indices_path='data/indices/test_class.npy')

    fsm = FileStructManager(base_dir=config_type.experiment_dir, is_continue=True)
    predictor = Predictor(config_type.create_model().cuda(), fsm=fsm)

    predicts, targets = [], []

    for i, data in enumerate(tqdm(dataset)):
        targets.append(np.squeeze(data['target'].data.cpu().numpy()))

        data = cv2.resize(data['data'], (512, 512))
        img_tensor = torch.from_numpy(np.expand_dims(np.expand_dims(data.astype(np.float32), 0) / 128 - 1, 0)).cuda()
        res = np.squeeze(predictor.predict({'data': img_tensor}).data.cpu().numpy())
        predicts.append(res)

    return np.array(predicts), np.array(targets)


def predict_results(config_types: [type(BaseClassificationTrainConfig)], output_file: str, threshold: float):
    dataset = create_dataset(is_test=True, include_negatives=True, indices_path='data/indices/test_class.npy')

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
            data = cv2.resize(data['data'], (512, 512))
            img_tensor = torch.from_numpy(np.expand_dims(np.expand_dims(data.astype(np.float32), 0) / 128 - 1, 0)).cuda()

            res = []
            for predictor in predictors:
                res.append(np.squeeze(predictor.predict({'data': img_tensor}).data.cpu().numpy()))
            res = np.median(res)
            res = np.where(res > threshold, 1, 0).astype(np.int)

            out_file.write("{},{}\n".format(os.path.splitext(os.path.basename(images_paths[i]))[0], float(res)))
            out_file.flush()


def merge_predicts(predicts: []) -> {}:
    res = predicts[0].copy()

    for pred in predicts[1:]:
        if type(res['predicts']) is not list:
            res['predicts'] = [res['predicts']]
        res['predicts'] += [pred['predicts']]
        res['models'].extend(res['models'])

    res['predicts'] = np.median(res['predicts'], axis=0)
    return res


def calc_metric(predict: {}, threshold: float):
    inner_predict = np.where(predict['predicts'][0] > threshold, 1, 0).astype(np.int)
    return roc_auc_score(predict['predicts'][1], inner_predict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict')
    parser.add_argument('-m', '--models', help='Models to predict', required=True, nargs='+',
                        choices=['resnet18', 'resnet34'])
    parser.add_argument('-o', '--out', type=str, help='Output file path', required=True)

    if len(sys.argv) < 3:
        print('Bad arguments passed', file=sys.stderr)
        parser.print_help(file=sys.stderr)
        exit(2)
    args = parser.parse_args()

    model_configs = {'resnet18': ResNet18ClassificationTrainConfig,
                     'resnet34': ResNet34ClassificationTrainConfig}
    all_predicts = {}
    for model in args.models:
        if model not in model_configs:
            raise Exception("Train pipeline doesn't implemented for model '{}'".format(args.model))

        all_predicts[model] = {'models': [model], 'predicts': predict_on_test_set(model_configs[model])}

    for cmb_len in tqdm(range(2, len(all_predicts))):
        for cmb in itertools.combinations(all_predicts, cmb_len):
            all_predicts.append(merge_predicts(cmb))

    best_predict, best_metric = None, 0
    for predict in all_predicts:
        for thresh in np.linspace(0.1, 0.99, num=int((0.99 - 0.1) / 0.01)):
            cur_metric = calc_metric(predict, threshold=thresh)

            if cur_metric > best_metric:
                best_predict = {'thresh': thresh, 'models': predict['models']}
                best_metric = cur_metric

    with open('best_class_config.json', 'w') as out:
        json.dump({'models': best_predict['models'], 'thresh': best_predict['thresh']}, out)

    predict_results([model_configs[model] for model in best_predict['models']], args.out,
                    threshold=best_predict['thresh'])
