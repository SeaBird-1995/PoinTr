'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-05-04 10:36:14
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

import sys
sys.path.append("./")
sys.path.append("../")

import os
import os.path as osp
from glob import glob
from tqdm import tqdm
import time
import numpy as np
import torchvision
import cv2
from PIL import Image
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from utils.config import cfg_from_yaml_file
from tools import builder

def test_PoinTr():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config_file = "./cfgs/ShapeNet34_models/PoinTr.yaml"

    config = cfg_from_yaml_file(config_file)
    base_model = builder.model_builder(config.model)
    base_model = base_model.to(device)


    input = torch.randn(4, 2048, 3).to(device)
    output = base_model(input)
    print(output[0].shape, output[1].shape)




if __name__ == "__main__":
    test_PoinTr()