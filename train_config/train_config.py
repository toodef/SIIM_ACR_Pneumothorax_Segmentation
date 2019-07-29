import os
from abc import ABCMeta, abstractmethod

import torch
from cv_utils.losses.common import Reduction
from cv_utils.losses.segmentation import BCEDiceLoss
from cv_utils.metrics.torch.classification import ClassificationMetricsProcessor
from cv_utils.metrics.torch.segmentation import SegmentationMetricsProcessor
from cv_utils.models import ResNet18, ModelsWeightsStorage, ModelWithActivation, ResNet34, ClassificationModel
from cv_utils.models.decoders.unet import UNetDecoder
from neural_pipeline import TrainConfig, DataProducer, TrainStage, ValidationStage
from torch import nn
from torch.optim import Adam
from torch.nn import Module, BCEWithLogitsLoss, BCELoss

from train_config.dataset import create_augmented_dataset_for_seg, create_augmented_dataset_for_class

__all__ = ['BaseSegmentationTrainConfig', 'ResNet18SegmentationTrainConfig', 'ResNet34SegmentationTrainConfig',
           'BaseClassificationTrainConfig', 'ResNet18ClassificationTrainConfig', 'ResNet34ClassificationTrainConfig']


class BaseSegmentationTrainConfig(TrainConfig, metaclass=ABCMeta):
    experiment_name = 'exp1'
    experiment_dir = os.path.join('experiments', experiment_name)
    batch_size = 2

    def __init__(self, fold_indices: {}):
        model = self.create_model().cuda()

        dir = os.path.join('data', 'indices')

        train_dts = []
        for indices in fold_indices['train']:
            train_dts.append(create_augmented_dataset_for_seg(is_train=True, is_test=False,
                                                              indices_path=os.path.join(dir, indices)))

        val_dts = create_augmented_dataset_for_seg(is_train=False, is_test=False,
                                                   indices_path=os.path.join(dir, fold_indices['val']))

        self._train_data_producer = DataProducer(train_dts, batch_size=self.batch_size, num_workers=6). \
            global_shuffle(True).pin_memory(True)
        self._val_data_producer = DataProducer([val_dts], batch_size=self.batch_size, num_workers=6). \
            global_shuffle(True).pin_memory(True)

        self.train_stage = TrainStage(self._train_data_producer, SegmentationMetricsProcessor('train'))
        self.val_stage = ValidationStage(self._val_data_producer, SegmentationMetricsProcessor('validation'))

        loss = BCEDiceLoss(0.5, 0.5, reduction=Reduction('mean')).cuda()
        optimizer = Adam(params=model.parameters(), lr=1e-4)

        super().__init__(model, [self.train_stage, self.val_stage], loss, optimizer)

    @staticmethod
    @abstractmethod
    def create_model() -> Module:
        pass


class ResNet18SegmentationTrainConfig(BaseSegmentationTrainConfig):
    experiment_dir = os.path.join(BaseSegmentationTrainConfig.experiment_dir, 'seg', 'resnet18')

    @staticmethod
    def create_model() -> Module:
        """
        It is better to init model by separated method
        :return:
        """
        enc = ResNet18(in_channels=3)
        ModelsWeightsStorage().load(enc, 'imagenet')
        model = UNetDecoder(enc, classes_num=1)
        return ModelWithActivation(model, activation='sigmoid')


class ResNet34SegmentationTrainConfig(BaseSegmentationTrainConfig):
    experiment_dir = os.path.join(BaseSegmentationTrainConfig.experiment_dir, 'seg', 'resnet34')

    @staticmethod
    def create_model() -> Module:
        """
        It is better to init model by separated method
        :return:
        """
        enc = ResNet34(in_channels=3)
        ModelsWeightsStorage().load(enc, 'imagenet')
        model = UNetDecoder(enc, classes_num=1)
        return ModelWithActivation(model, activation='sigmoid')


class BaseClassificationTrainConfig(TrainConfig, metaclass=ABCMeta):
    experiment_name = 'exp2'
    experiment_dir = os.path.join('experiments', experiment_name)
    batch_size = 2

    def __init__(self, fold_indices: {}):
        model = self.create_model().cuda()

        dir = os.path.join('data', 'indices')

        train_dts = []
        for indices in fold_indices['train']:
            train_dts.append(create_augmented_dataset_for_class(is_train=True, is_test=False,
                                                                indices_path=os.path.join(dir, indices)))

        val_dts = create_augmented_dataset_for_class(is_train=False, is_test=False,
                                                     indices_path=os.path.join(dir, fold_indices['val']))

        self._train_data_producer = DataProducer(train_dts, batch_size=self.batch_size, num_workers=6). \
            global_shuffle(True).pin_memory(True).drop_last(True)
        self._val_data_producer = DataProducer([val_dts], batch_size=self.batch_size, num_workers=6). \
            global_shuffle(True).pin_memory(True).drop_last(True)

        self.train_stage = TrainStage(self._train_data_producer,
                                      ClassificationMetricsProcessor('train', [0.4, 0.6, 0.8]))
        self.val_stage = ValidationStage(self._val_data_producer,
                                         ClassificationMetricsProcessor('validation', [0.4, 0.6, 0.8]))

        loss = BCELoss(weight=torch.Tensor([[2]] * self.batch_size), reduction='sum').cuda()
        optimizer = Adam(params=model.parameters(), lr=1e-4)

        super().__init__(model, [self.train_stage, self.val_stage], loss, optimizer)

    @staticmethod
    @abstractmethod
    def create_model() -> Module:
        pass


class ResNet18ClassificationTrainConfig(BaseClassificationTrainConfig):
    experiment_dir = os.path.join(BaseClassificationTrainConfig.experiment_dir, 'class', 'resnet18')

    @staticmethod
    def create_model() -> Module:
        """
        It is better to init model by separated method
        :return:
        """
        enc = ResNet18(in_channels=3)
        ModelsWeightsStorage().load(enc, 'imagenet')
        model = ClassificationModel(enc, in_features=115200, classes_num=1, pool=nn.AdaptiveAvgPool2d(15))
        return ModelWithActivation(model, activation='sigmoid')


class ResNet34ClassificationTrainConfig(BaseClassificationTrainConfig):
    experiment_dir = os.path.join(BaseClassificationTrainConfig.experiment_dir, 'class', 'resnet34')

    @staticmethod
    def create_model() -> Module:
        """
        It is better to init model by separated method
        :return:
        """
        enc = ResNet34(in_channels=3)
        ModelsWeightsStorage().load(enc, 'imagenet')
        model = ClassificationModel(enc, in_features=115200, classes_num=1, pool=nn.AdaptiveAvgPool2d(15))
        return ModelWithActivation(model, activation='sigmoid')
