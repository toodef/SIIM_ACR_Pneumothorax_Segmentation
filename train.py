import argparse
import sys

import torch
import numpy as np

from neural_pipeline import Trainer, FileStructManager
from neural_pipeline.builtin.monitors.tensorboard import TensorboardMonitor

from train_config.train_config import ResNet18TrainConfig, ResNet34TrainConfig, BaseTrainConfig


def train(config_type: BaseTrainConfig):
    fsm = FileStructManager(base_dir=config_type.experiment_dir, is_continue=False)

    config = config_type({'train': ['train.npy'], 'val': 'val.npy'})

    trainer = Trainer(config, fsm, device=torch.device('cuda'))
    tensorboard = TensorboardMonitor(fsm, is_continue=False)
    trainer.monitor_hub.add_monitor(tensorboard)

    trainer.set_epoch_num(300)
    trainer.enable_lr_decaying(coeff=0.5, patience=10, target_val_clbk=lambda: np.mean(config.val_stage.get_losses()))
    trainer.add_on_epoch_end_callback(lambda: tensorboard.update_scalar('params/lr', trainer.data_processor().get_lr()))
    trainer.enable_best_states_saving(lambda: np.mean(config.val_stage.get_losses()))
    trainer.add_stop_rule(lambda: trainer.data_processor().get_lr() < 1e-6)

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('-m', '--model', type=str, help='Train one model', required=True,
                        choices=['resnet18', 'resnet34'])

    if len(sys.argv) < 2:
        print('Bad arguments passed', file=sys.stderr)
        parser.print_help(file=sys.stderr)
        exit(2)
    args = parser.parse_args()

    if args.model == 'resnet18':
        train(ResNet18TrainConfig)
    elif args.model == 'resnet34':
        train(ResNet34TrainConfig)
    else:
        raise Exception("Train pipeline doesn't implemented for model '{}'".format(args.model))
