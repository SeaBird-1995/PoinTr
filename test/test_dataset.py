'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-04-11 23:15:41
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
from easydict import EasyDict
import torch

import utils.misc as misc

def test_seprate_point_cloud():
    file_path = "/home/zhanghm/Research/Completion/PoinTr/data/ShapeNet55-34/shapenet_pc/02691156-1a04e3eab45ca15dd86060f189eb133.npy"
    gt = np.load(file_path).astype(np.float32)

    gt = torch.from_numpy(gt[None]).float().cuda()
    npoints = 8192
    partial, _ = misc.seprate_point_cloud(gt, npoints, [int(npoints * 1/4) , int(npoints * 3/4)], fixed_points = None)
    print(partial.shape)


def test_ShapeNetViPCOrigin():
    from datasets.ShapeNetViPC import ShapeNetViPCOrigin
    
    config = EasyDict(filepath="./data/ShapeNetViPC-Dataset/train_list2.txt",
                      data_path="data/ShapeNetViPC-Dataset",
                      subset="train",
                      category="car",
                      pc_input_num=3500)
    dataset = ShapeNetViPCOrigin(config)
    print(len(dataset))
    
    entry = dataset[0]
    for key, value in entry.items():
        try:
            print(key, value.shape)
        except:
            print(key, value)

    pc_gt = entry['pc_gt']
    pc_partial = entry['pc_partial']
    np.savetxt("pc_gt2.xyz", pc_gt)
    np.savetxt("pc_partial2.xyz", pc_partial)


def test_KITTIDataset():
    from datasets.KITTIDataset import KITTI
    
    config = EasyDict(CATEGORY_FILE_PATH="data/KITTI/KITTI.json",
                      subset="test",
                      N_POINTS=16384,
                      CLOUD_PATH="data/KITTI/cars/%s.pcd",
                      BBOX_PATH="data/KITTI/bboxes/%s.txt")
    dataset = KITTI(config)
    print(len(dataset))
    
    entry = dataset[0]
    taxonomy_id, model_id, partial_cloud = entry
    print(taxonomy_id, model_id, partial_cloud.shape)

    np.savetxt(f"{model_id}.xyz", partial_cloud)


if __name__ == "__main__":
    test_KITTIDataset()