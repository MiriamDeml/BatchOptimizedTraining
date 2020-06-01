import os
import buildModels.model_loader

import torch

# Code by Li et al. (2018): https://github.com/tomgoldstein/loss-landscape/blob/master/model_loader.py

def load(dataset, model_name, model_file, data_parallel=False):
    if dataset == 'cifar10':
        net = buildModels.model_loader.load(model_name, model_file, data_parallel)
    return net
