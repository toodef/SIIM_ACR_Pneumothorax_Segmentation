import os

from cv_utils.datasets.common import BasicDataset

from train_config.dataset import Dataset


def stratificate_dataset(dataset: BasicDataset, parts: []):
    indices_num = len(dataset)
    all_indices = list(range(indices_num))
    
    res = []
    
    for p in parts:
        res.append(all_indices[:int(indices_num * p)])
        del all_indices[:int(indices_num * p)]
    return res


def check_indices_for_intersection(indices: []):
    for i in range(len(indices)):
        for index in indices[i]:
            for other_indices in indices[i + 1:]:
                if index in other_indices:
                    raise Exception('Indices intersects')


if __name__ == '__main__':
    dataset = Dataset(is_test=False)

    train_indices, val_indices = stratificate_dataset(dataset, [0.8, 0.2])

    dir = os.path.join('data', 'indices')
    if not os.path.exists(dir) and not os.path.isdir(dir):
        os.makedirs(dir)

    check_indices_for_intersection([train_indices, val_indices])

    train_path = os.path.join(dir, 'train.npy')
    val_path = os.path.join(dir, 'val.npy')

    Dataset(is_test=False).set_indices(train_indices).flush_indices(train_path)
    Dataset(is_test=False).set_indices(val_indices).flush_indices(val_path)

    Dataset(is_test=False).load_indices(train_path, remove_unused=True)
    Dataset(is_test=False).load_indices(val_path, remove_unused=True)
