import os
from itertools import combinations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from train_config.dataset import create_dataset, Dataset


def calc_metric(predicts_dict: {}, dataset: Dataset) -> float:
    predicts, targets = [], []
    images_paths = dataset.get_items()
    for i, data in enumerate(dataset):
        targets.append(data['target'])
        predicts.append(predicts_dict[os.path.splitext(os.path.basename(images_paths[i]))[0]])

    predicts = np.array(predicts)
    targets = np.array(targets)
    metric = roc_auc_score(predicts, targets)
    return float(np.mean(metric))


def get_metric_by_threshold(predicts_dict: {}, dataset: Dataset):
    best_pred, best_metric = None, 0
    for pred in all_predicts:
        cur_metric = calc_metric(pred, dataset)
        if cur_metric > best_metric:
            best_metric = cur_metric
            best_pred = pred


def unite_predicts(predicts: [{}]) -> {}:
    res = {k: [v] for k, v in predicts[0].items()}
    for pred in predicts[1:]:
        for k, v in pred.items():
            res[k].append(v)

    for k, v in res.items():
        res[k] = np.mean(res[k])

    return res


if __name__ == '__main__':
    predicts_path = r'out/class'
    all_predicts = []
    for f in os.listdir(predicts_path):
        predicts_from_file = np.loadtxt(os.path.join(predicts_path, f), delimiter=',')[1:]
        predicts_from_file = {v[0]: v[1] for v in predicts_from_file}

        all_predicts.append(predicts_from_file)

    for num in range(2, len(all_predicts)):
        for cmb in list(combinations(all_predicts, num)):
            all_predicts.append(unite_predicts([all_predicts[i] for i in cmb]))

    dataset = create_dataset(is_test=False, indices_path='data/out/test_class.npy')

    best_pred, best_metric = None, 0
    for pred in all_predicts:
        cur_metric = calc_metric(pred, dataset)
        if cur_metric > best_metric:
            best_metric = cur_metric
            best_pred = pred

    with open(os.path.join(predicts_path, 'class_best_predict.csv'), 'w') as out:
        out.write("ImageId,EncodedPixels\n")
        out.flush()

        for name, pred in best_pred.items():
            out.write("{},{}\n".format(name, float(pred)))
            out.flush()
