import cv2
import numpy as np
from cv_utils.viz import ColormapVisualizer

from train_config.dataset import create_augmented_dataset_for_seg


if __name__ == '__main__':
    dataset = create_augmented_dataset_for_seg(is_train=True, is_test=False, to_pytorch=False, include_negatives=False)
    vis = ColormapVisualizer([0.5, 0.5])
    for img_idx, d in enumerate(dataset):
        if d['target'].max() > 0:
            img = vis.process_img(cv2.cvtColor(d['data'], cv2.COLOR_GRAY2BGR), (np.expand_dims(d['target'], axis=2) * 255).astype(np.uint8))
        else:
            img = d['data']
        cv2.imshow('img', img)
        cv2.imshow('mask', d['target'].astype(np.float32))
        cv2.waitKey()
