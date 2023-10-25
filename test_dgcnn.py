'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-12-08 10:30:06
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

import torch
from models.dgcnn_group import DGCNN_Grouper


def test_DGCNN():
    input = torch.randn(4, 3, 2048).cuda()
    input = torch.randn(4, 3, 2048).cuda()

    grouper = DGCNN_Grouper().cuda()

    coor, f = grouper(input)
    print(coor.shape, f.shape)


def test_FoldNet():
    from models.PoinTr import Fold

    num_pred = 14336
    num_query = 224

    fold_step = int(pow(num_pred//num_query, 0.5) + 0.5)
    print(fold_step)

    foldingnet = Fold(384, step=fold_step, hidden_dim=256)

    B = 5
    M = 224

    input_feature = torch.randn(B*M, 384)
    relative_xyz = foldingnet(input_feature).reshape(B, M, 3, -1)
    print(relative_xyz.shape)

    coarse_point_cloud = torch.randn(B, M, 3)

    rebuild_points = (relative_xyz + coarse_point_cloud.unsqueeze(-1)).transpose(2,3).reshape(B, -1, 3)  # B N 3
    print(rebuild_points.shape)


def test_FoldNet2():
    from models.PoinTr import Fold

    num_pred = 8192
    num_query = 128

    fold_step = int(pow(num_pred//num_query, 0.5) + 0.5)
    print(fold_step)

    foldingnet = Fold(256, step=fold_step, hidden_dim=256)

    B = 5
    M = 128

    input_feature = torch.randn(B*M, 256)
    relative_xyz = foldingnet(input_feature).reshape(B, M, 3, -1)
    print(relative_xyz.shape)

    coarse_point_cloud = torch.randn(B, M, 3)

    rebuild_points = (relative_xyz + coarse_point_cloud.unsqueeze(-1)).transpose(2,3).reshape(B, -1, 3)  # B N 3
    print(rebuild_points.shape)


if __name__ == "__main__":
    test_FoldNet2()