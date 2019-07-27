import os

from cv_utils.datasets.common import BasicDataset

from train_config.dataset import Dataset


def stratificate_dataset(dataset: BasicDataset, parts: [], for_segmentation: bool) -> []:
    if for_segmentation:
        all_indices = [i for i, d in enumerate(dataset) if d['target'] > 0]
    else:
        all_indices = list(range(len(dataset)))

    indices_num = len(all_indices)

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


def generate_indices(dataset, for_segmentation) -> None:
    train_indices, val_indices, test_indices = stratificate_dataset(dataset, [0.7, 0.2, 0.1], for_segmentation)

    dir = os.path.join('data', 'indices')
    if not os.path.exists(dir) and not os.path.isdir(dir):
        os.makedirs(dir)

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
    dataset_args = {'is_test': False, 'for_segmentation': False}
    dataset = Dataset(**dataset_args)

    generate_indices(dataset, for_segmentation=False)
    generate_indices(dataset, for_segmentation=True)
