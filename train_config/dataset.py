import os

import cv2
import pydicom
from cv_utils.utils import rle2mask
from pydicom.data import get_testdata_files

import numpy as np
import torch
from albumentations import Compose, HorizontalFlip, Resize, Rotate, OneOf, RandomContrast, RandomGamma, RandomBrightness, \
    ElasticTransform, GridDistortion, OpticalDistortion, RandomSizedCrop, RandomCrop, RandomBrightnessContrast, GaussNoise
from cv_utils.datasets.common import BasicDataset

__all__ = ['Dataset', 'AugmentedDataset', 'create_dataset', 'create_augmented_dataset_for_seg',
           'create_augmented_dataset_for_class']

DATA_HEIGHT, DATA_WIDTH = 512, 512


class Dataset(BasicDataset):
    env_name = 'SIIM_ACR_PNEUMOTORAX_SEGMENTATION_DATASET'  # name of environment with path to dataset

    def __init__(self, is_test: bool, include_negatives: bool):
        if self.env_name not in os.environ:
            raise Exception("Cant find dataset root path. Please set {} environment variable".format(self.env_name))

        self._is_test = is_test
        self._include_negatives = include_negatives

        root = os.environ[self.env_name]

        labels = np.genfromtxt(os.path.join(root, 'train-rle.csv'), delimiter=',', dtype=str)[1:]
        images_root = os.path.join(root, 'dicom-images-test' if is_test else 'dicom-images-train')
        items = self._parse_test_items(images_root) if is_test else self._parse_train_items(images_root, labels)

        super().__init__(items)

    def _interpret_item(self, it) -> any:
        if self._is_test:
            return pydicom.dcmread(it).pixel_array

        ds = pydicom.dcmread(it[0])
        img = ds.pixel_array
        rles = it[1]

        if type(rles) is list:
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
            for rle in rles:
                mask += rle2mask(rle, img.shape[0], img.shape[1]).astype(np.float32) / 255
            mask = mask.T
        else:
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
        return {'data': img, 'target': mask}

    def _parse_train_items(self, images_root: str, labels: np.ndarray) -> []:
        images = self._parse_images(images_root)

        pairs = {}
        for label in labels:
            ident = label[0]
            if ident in images:
                rle = str(label[1]).strip()
                if ident in pairs:
                    pairs[ident].append(rle)
                else:
                    pairs[ident] = [rle]
            else:
                raise Exception("Not all targets have paired images")

        res = []
        for ident, rle in pairs.items():
            if rle == ['-1']:
                if self._include_negatives:
                    res.append([images[ident], -1])
            else:
                rle = [[int(v) for v in r.split(' ')] for r in rle]
                res.append([images[ident], rle])

        return res

    @staticmethod
    def _parse_test_items(images_root: str) -> []:
        images = []
        for cur_root, dirs, files in os.walk(images_root):
            for file in files:
                if os.path.splitext(file)[1] == ".dcm":
                    images.append(os.path.join(cur_root, file))
        return images

    @staticmethod
    def _parse_images(images_root: str) -> []:
        images = {}
        for cur_root, dirs, files in os.walk(images_root):
            for file in files:
                if os.path.splitext(file)[1] == ".dcm":
                    identifier = os.path.splitext(file)[0]
                    if identifier in images:
                        raise RuntimeError('Images names are duplicated [{} and {}]'.format(file, images[identifier]))
                    images[identifier] = os.path.join(cur_root, file)
        return images


class AugmentedDataset:
    def __init__(self, dataset):
        self._dataset = dataset
        self._augs = {}
        self._augs_for_whole = []

    def add_aug(self, aug: callable, identificator=None) -> 'AugmentedDataset':
        if identificator is None:
            self._augs_for_whole.append(aug)
        else:
            self._augs[identificator] = aug
        return self

    def __getitem__(self, item):
        res = self._dataset[item]
        for k, v in res.items() if isinstance(res, dict) else enumerate(res):
            if k in self._augs:
                res[k] = self._augs[k](v)
        for aug in self._augs_for_whole:
            res = aug(res)
        return res

    def __len__(self):
        return len(self._dataset)


class SegmentationAugmentations:
    def __init__(self, is_train: bool, to_pytorch: bool):
        preprocess = OneOf([Resize(height=DATA_HEIGHT, width=DATA_WIDTH),
                            Compose([Resize(height=int(DATA_HEIGHT * 1.2), width=int(DATA_WIDTH * 1.2)),
                                     RandomCrop(height=DATA_HEIGHT, width=DATA_WIDTH)])], p=1)

        if is_train:
            self._aug = Compose([preprocess,
                                 HorizontalFlip(p=0.5),
                                 GaussNoise(p=0.3),
                                 # OneOf([
                                 #     RandomBrightnessContrast(),
                                 #     RandomGamma(),
                                 # ], p=0.3),
                                 # OneOf([
                                 #     ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                                 #     GridDistortion(),
                                 #     OpticalDistortion(distort_limit=2, shift_limit=0.5),
                                 # ], p=0.3),
                                 Rotate(limit=20, border_mode=cv2.BORDER_REPLICATE),
                                 ], p=1)
        else:
            self._aug = preprocess

        self._need_to_pytorch = to_pytorch

    def augmentate(self, data: {}):
        augmented = self._aug(image=data['data'], mask=data['target'])
        clahe = cv2.createCLAHE(clipLimit=2.0)
        augmented['image'] = clahe.apply(augmented['image'])

        if self._need_to_pytorch:
            img = np.expand_dims(augmented['image'], axis=0)
            image = img.astype(np.float32) / 128 - 1
            target = np.expand_dims(augmented['mask'], 0)
            return {'data': torch.from_numpy(image), 'target': torch.from_numpy(target)}
        else:
            return {'data': augmented['image'], 'target': augmented['mask']}


class ClassificationAugmentations:
    def __init__(self, is_train: bool, to_pytorch: bool):
        preprocess = OneOf([Resize(height=DATA_HEIGHT, width=DATA_WIDTH),
                            Compose([Resize(height=int(DATA_HEIGHT * 1.2), width=int(DATA_WIDTH * 1.2)),
                                     RandomCrop(height=DATA_HEIGHT, width=DATA_WIDTH)])], p=1)

        if is_train:
            self._aug = Compose([preprocess,
                                 HorizontalFlip(p=0.5),
                                 GaussNoise(p=0.3),
                                 # OneOf([
                                 #     RandomBrightnessContrast(),
                                 #     RandomGamma(),
                                 # ], p=0.3),
                                 # OneOf([
                                 #     ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                                 #     GridDistortion(),
                                 #     OpticalDistortion(distort_limit=2, shift_limit=0.5),
                                 # ], p=0.3),
                                 Rotate(limit=20),
                                 ], p=1)
        else:
            self._aug = preprocess

        self._need_to_pytorch = to_pytorch

    def augmentate(self, data: {}):
        augmented = self._aug(image=data['data'])
        target = np.array([self.mask2class(data['target'])], dtype=np.float32)

        clahe = cv2.createCLAHE(clipLimit=2.0)
        augmented['image'] = clahe.apply(augmented['image'])

        if self._need_to_pytorch:
            img = np.expand_dims(augmented['image'], axis=0)
            image = img.astype(np.float32) / 128 - 1
            return {'data': torch.from_numpy(image), 'target': torch.from_numpy(target)}
        else:
            return {'data': augmented['image'], 'target': target}

    @staticmethod
    def mask2class(mask: np.ndarray) -> int:
        return int(np.count_nonzero(mask) > 5)


def create_dataset(is_test: bool, include_negatives: bool, indices_path: str = None) -> 'Dataset':
    dataset = Dataset(is_test, include_negatives=include_negatives)
    if indices_path is not None:
        dataset.load_indices(indices_path, remove_unused=True)
    return dataset


def create_augmented_dataset_for_seg(is_train: bool, is_test: bool, include_negatives: bool, to_pytorch: bool = True,
                                     indices_path: str = None) -> 'AugmentedDataset':
    dataset = create_dataset(is_test, include_negatives, indices_path)
    augs = SegmentationAugmentations(is_train, to_pytorch)

    return AugmentedDataset(dataset).add_aug(augs.augmentate)


def create_augmented_dataset_for_class(is_train: bool, is_test: bool, to_pytorch: bool = True,
                                       indices_path: str = None) -> 'AugmentedDataset':
    dataset = create_dataset(is_test, True, indices_path)
    augs = ClassificationAugmentations(is_train, to_pytorch)

    return AugmentedDataset(dataset).add_aug(augs.augmentate)
