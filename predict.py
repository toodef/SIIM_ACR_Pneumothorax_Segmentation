import os

import numpy as np
from neural_pipeline import Predictor, FileStructManager

from train_config.dataset import create_augmented_dataset
from train_config.metrics import MyMetric
from train_config.train_config import MyTrainConfig, create_model

if __name__ == '__main__':
    dataset = create_augmented_dataset(is_train=False, indices_path='test_indices.npy')

    dir = os.path.join('predicts')
    if not os.path.exists(dir) and not os.path.isdir(dir):
        os.makedirs(dir)

    metric = MyMetric()

    fsm = FileStructManager(base_dir=MyTrainConfig.experiment_dir, is_continue=True)
    predictor = Predictor(create_model(), fsm=fsm)

    metrics_vals = []
    for i, data in enumerate(dataset):
        res = predictor.predict({'data': data}).data.cpu().numpy()

        metrics_vals.append(metric.calc(data, res))

    metrics_vals = np.array(metrics_vals)
    print('Metric score: ', np.mean(metrics_vals), np.std(metrics_vals))
