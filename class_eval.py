import json
import numpy as np
import os

from predict_model_class import predict

if __name__ == '__main__':
    with open(r'out/class/class_best_predict.json') as class_config_pred_file:
        config = json.load(class_config_pred_file)

    thresh, net = config['thresh'], config['net']
    nets = net.split(',')

    # predicts_path = r'out/class'
    # for net in nets:
    #     predict
