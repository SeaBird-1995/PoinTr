'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-05-04 12:09:31
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''
import sys
sys.path.append("./")
sys.path.append("../")

from PIL import Image
import torch
import numpy as np

from utils import misc, dist_utils


def test_get_ptcloud_img():
    input_pc = np.random.random((2048, 3))
    input_pc = misc.get_ptcloud_img(input_pc)
    print(type(input_pc), input_pc.shape)

    img = Image.fromarray(input_pc)
    img.save("dfsdf.png")


def test_ChamferDistance():
    from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2, L2_ChamferEval_Split

    pred_pc = "/home/zhanghm/Research/Completion/XMFnet/experiments/NIPS/Trans2CapDistillStudentOnly/03636649/1a5ebc8575a4e5edcc901650bbbbb0b5_pred_1.76193.xyz"
    gt_pc = "/home/zhanghm/Research/Completion/XMFnet/experiments/NIPS/Trans2CapDistillStudentOnly/03636649/1a5ebc8575a4e5edcc901650bbbbb0b5_gt.xyz"

    pred_pc = np.loadtxt(pred_pc).astype(np.float32)
    gt_pc = np.loadtxt(gt_pc).astype(np.float32)

    pred_pc = torch.from_numpy(pred_pc)[None].cuda()
    gt_pc = torch.from_numpy(gt_pc)[None].cuda()

    cd_l2_metric = ChamferDistanceL2()

    cd_l2 = cd_l2_metric(pred_pc, gt_pc)
    print(type(cd_l2))
    print(cd_l2.item())

    cd_l2_metric_split = L2_ChamferEval_Split()
    dist = cd_l2_metric_split(pred_pc, gt_pc)
    print(dist.shape)
    # print(cd_l2.item().cpu())


if __name__ == "__main__":
    test_ChamferDistance()

