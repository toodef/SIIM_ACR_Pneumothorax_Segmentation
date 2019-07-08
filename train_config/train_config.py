import os
from abc import ABCMeta, abstractmethod

import torch
from cv_utils.losses.common import Reduction
from cv_utils.losses.segmentation import BCEDiceLoss
from cv_utils.metrics.torch.segmentation import SegmentationMetricsProcessor
from cv_utils.models import ResNet18, ModelsWeightsStorage, ModelWithActivation, ResNet34
from cv_utils.models.decoders.unet import UNetDecoder
from neural_pipeline import TrainConfig, DataProducer, TrainStage, ValidationStage
from torch.optim import Adam
from torch.nn import Module

from train_config.dataset import create_augmented_dataset

__all__ = ['BaseTrainConfig', 'ResNet18TrainConfig', 'ResNet34TrainConfig']


class BaseTrainConfig(TrainConfig, metaclass=ABCMeta):
    experiment_name = 'exp1'
    experiment_dir = os.path.join('experiments', experiment_name)
    batch_size = 6

    def __init__(self, fold_indices: {}):
        model = self.create_model().cuda()

        dir = os.path.join('data', 'indices')

        train_dts = []
        for indices in fold_indices['train']:
            train_dts.append(create_augmented_dataset(is_train=True, is_test=False, indices_path=os.path.join(dir, indices)))

        val_dts = create_augmented_dataset(is_train=False, is_test=False, indices_path=os.path.join(dir, fold_indices['val']))

        self._train_data_producer = DataProducer(train_dts, batch_size=self.batch_size, num_workers=12).\
            global_shuffle(True).pin_memory(True)
        self._val_data_producer = DataProducer([val_dts], batch_size=self.batch_size, num_workers=12).\
            global_shuffle(True).pin_memory(True)

        self.train_stage = TrainStage(self._train_data_producer, SegmentationMetricsProcessor('train'))
        self.val_stage = ValidationStage(self._val_data_producer, SegmentationMetricsProcessor('validation'))

        loss = BCEDiceLoss(0.5, 0.5, reduction=Reduction('mean'), class_weights=[0.3, 0.7]).cuda()
        optimizer = Adam(params=model.parameters(), lr=1e-4)

        super().__init__(model, [self.train_stage, self.val_stage], loss, optimizer)

    @staticmethod
    @abstractmethod
    def create_model() -> Module:
        pass


class ResNet18TrainConfig(BaseTrainConfig):
    experiment_dir = os.path.join(BaseTrainConfig.experiment_dir, 'resnet18')

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


class ResNet34TrainConfig(BaseTrainConfig):
    experiment_dir = os.path.join(BaseTrainConfig.experiment_dir, 'resnet34')

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
