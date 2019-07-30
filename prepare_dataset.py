import os
from random import randint

from cv_utils.datasets.common import BasicDataset
import numpy as np
from train_config.dataset import Dataset


def fill_hist(target_hist: [], indices: {}):
    def pick(d):
        idx = randint(0, len(indices[d]) - 1)
        res = indices[d][idx]
        del indices[d][idx]
        return res

    res = {}
    for idx, d in enumerate(target_hist):
        idxes = []
        for _ in range(d):
            idxes.append(pick(idx))
        res[idx] = idxes
    return res


def calc_hist(dataset: Dataset):
    indices = {}
    for i, d in enumerate(dataset.get_items()):
        indices[d[0]] = 0 if d[1] == -1 else len(d[1])

    hist = [[] for i in range(max(indices.values()))]
    for data, idxes in indices.items():
        hist[idxes - 1].append(data)
    return np.array([len(v) for v in hist]), hist


def stratificate_dataset(hist: np.ndarray, indices: {}, parts: [float]) -> []:
    res = []
    for part in parts:
        target_hist = (hist.copy() * part).astype(np.uint32)
        res.append([target_hist, fill_hist(target_hist, indices)])
    return res


def check_indices_for_intersection(indices: []):
    for i in range(len(indices)):
        for index in indices[i]:
            for other_indices in indices[i + 1:]:
                if index in other_indices:
                    raise Exception('Indices intersects')


def balance_classes(hist: np.ndarray, indices: {}) -> tuple:
    target_hist = hist.copy()
    target_hist[np.argmax(target_hist)] = np.sum(target_hist[target_hist != target_hist.max()])
    return target_hist, fill_hist(target_hist, indices)


def generate_indices(dataset, for_segmentation) -> None:
    hist, indices = calc_hist(dataset)
    # if not for_segmentation:
    #     hist, indices = balance_classes(hist, indices)

    train_indices, val_indices, test_indices = stratificate_dataset(hist, indices, [0.7, 0.2, 0.1])

    dir = os.path.join('data', 'indices')
    if not os.path.exists(dir) and not os.path.isdir(dir):
        os.makedirs(dir)

    indices = {d[0]: i for i, d in enumerate(dataset.get_items())}
    train_indices = [indices[it] for bin in train_indices[1].values() for it in bin]
    val_indices = [indices[it] for bin in val_indices[1].values() for it in bin]
    test_indices = [indices[it] for bin in test_indices[1].values() for it in bin]

    check_indices_for_intersection([train_indices, val_indices, test_indices])

    train_path = os.path.join(dir, 'train_{}.npy'.format('seg' if for_segmentation else 'class'))
    val_path = os.path.join(dir, 'val_{}.npy'.format('seg' if for_segmentation else 'class'))
    test_path = os.path.join(dir, 'test_{}.npy'.format('seg' if for_segmentation else 'class'))

    Dataset(**dataset_args).set_indices(train_indices).flush_indices(train_path)
    Dataset(**dataset_args).set_indices(val_indices).flush_indices(val_path)
    Dataset(**dataset_args).set_indices(test_indices).flush_indices(test_path)

    Dataset(**dataset_args).load_indices(train_path, remove_unused=True)
    Dataset(**dataset_args).load_indices(val_path, remove_unused=True)
    Dataset(**dataset_args).load_indices(test_path, remove_unused=True)


if __name__ == '__main__':
    dataset_args = {'is_test': False, 'include_negatives': True}
    generate_indices(Dataset(**dataset_args), for_segmentation=False)

    dataset_args = {'is_test': False, 'include_negatives': False}
    generate_indices(Dataset(**dataset_args), for_segmentation=True)
