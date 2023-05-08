'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-05-04 11:05:03
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

import os
from torchvision import transforms
import os.path
from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import pickle
import random
import math 
from tqdm import tqdm

from .build import DATASETS

SYNSET_MAP = {
    'airplane':'02691156',
    'bench': '02828884', 
    'cabinet':'02933112', 
    'car':'02958343',
    'chair':'03001627',
    'monitor': '03211117',
    'lamp':'03636649',
    'speaker': '03691459', 
    'firearm': '04090263', 
    'couch':'04256520',
    'table':'04379243',
    'cellphone': '04401088', 
    'watercraft':'04530566'
}


@DATASETS.register_module()
class ShapeNetViPCOrigin(Dataset):
    """The original ShapeNet-ViPC dataset without view images information.

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, config):
        super(ShapeNetViPCOrigin, self).__init__()

        filepath = config.filepath
        data_path = config.data_path
        status = config.subset
        pc_input_num = config.pc_input_num
        category = config.category

        self.pc_input_num = pc_input_num
        self.status = status
        self.filelist = []
        self.cat = []
        self.key = []

        self.imcomplete_path = os.path.join(data_path,'ShapeNetViPC-Partial')
        self.gt_path = os.path.join(data_path,'ShapeNetViPC-GT')
        self.rendering_path = os.path.join(data_path,'ShapeNetViPC-View')

        # load the split file
        lines = []
        with open(filepath, 'r') as rf:
            for line in tqdm(rf.readlines()):
                line = line.strip()

                if category != 'all':
                    if line.split(';')[0] != SYNSET_MAP[category]:
                        continue
                lines.append(line)
        rf.close()

        self.key = lines
        print(f'{status} data num: {len(self.key)}')

    def __getitem__(self, idx):
        key = self.key[idx]
        category_id, object_id, view_id = key.strip().split(';')
       
        pc_part_path = os.path.join(self.imcomplete_path, category_id, object_id, f'{view_id}.dat')
        pc_path = os.path.join(self.gt_path, category_id, object_id, f'{view_id}.dat')

        if not os.path.exists(pc_path):
            # print(f"[WARNING] {pc_path} is not exist....")
            return None
        
        # load gt
        with open(pc_path,'rb') as f:
            pc = pickle.load(f).astype(np.float32)
        # load partial points
        with open(pc_part_path,'rb') as f:
            pc_part = pickle.load(f).astype(np.float32)
        # incase some item point number less than 3500 
        if pc_part.shape[0]<self.pc_input_num:
            pc_part = np.repeat(pc_part,(self.pc_input_num//pc_part.shape[0])+1,axis=0)[0:self.pc_input_num]

        # normalize partial point cloud and GT to the same scale
        gt_mean = pc.mean(axis=0) 
        pc = pc -gt_mean
        pc_L_max = np.max(np.sqrt(np.sum(abs(pc ** 2), axis=-1)))
        pc = pc/pc_L_max

        pc_part = pc_part-gt_mean
        pc_part = pc_part/pc_L_max

        return {'taxonomy_id': category_id, 
                'model_id': object_id,
                'pc_gt': torch.from_numpy(pc), 
                'pc_partial': torch.from_numpy(pc_part)}  # pc: (N, 3), pc_part: (N, 3)

    def __len__(self):
        return len(self.key)
    

@DATASETS.register_module()
class ShapeNetViPC(Dataset):
    """The ViPC dataset using the new split borrowed from PoinTr, it totally sperate the objects
    in the training and testing set.

    Args:
        ViPCDataLoader (_type_): _description_
    """
    
    def __init__(self, config):
        super().__init__()

        filepath = config.filepath
        data_path = config.data_path
        status = config.subset
        pc_input_num = config.pc_input_num
        category = config.category
        total_views = config.total_views

        self.total_views = total_views
        self.pc_input_num = pc_input_num
        self.status = status

        self.imcomplete_path = os.path.join(data_path,'ShapeNetViPC-Partial')
        self.gt_path = os.path.join(data_path,'ShapeNetViPC-GT')
        self.rendering_path = os.path.join(data_path,'ShapeNetViPC-View')
        self.cat_map = {
            'airplane':'02691156',
            'bench': '02828884', 
            'cabinet':'02933112', 
            'car':'02958343',
            'chair':'03001627',
            'monitor': '03211117',
            'lamp':'03636649',
            'speaker': '03691459', 
            'firearm': '04090263', 
            'couch':'04256520',
            'table':'04379243',
            'cellphone': '04401088', 
            'watercraft':'04530566'
        }

        # NOTE: here we set the default test_skip to 4
        self.initialize(status, filepath, category, total_views, test_skip=4)
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor()
        ])

        print(f"[{status}]: Current category {category} with length {len(self.key)}")

    def initialize(self, status, file_path, category, total_views, test_skip=1):
        """_summary_

        Args:
            status (_type_): _description_
            file_path (_type_): _description_
            category (_type_): _description_
            total_views (_type_): _description_
            test_skip (int, optional): You can set this value if you don't want to test all views in
                the testing set. Defaults to 1.

        Raises:
            ValueError: _description_
        """
        with open(file_path) as f:
            lines = [line.rstrip() for line in f.readlines()]
        if category != "all":
            lines = list(filter(lambda x: x.split(" ")[0] == self.cat_map[category], lines))

        if status == "train":
            self.key = lines
        elif status == "test":
            view_id_list = [f"{i:02d}" for i in range(total_views)]
            view_id_list = view_id_list[::test_skip]

            data_arr = np.repeat(lines, len(view_id_list))

            view_id_arr = np.tile(view_id_list, len(lines))
            res = np.concatenate([data_arr[:, None], view_id_arr[:, None]], axis=-1)
            self.key = [" ".join(line) for line in res]
        else:
            raise ValueError
    
    def __len__(self):
        return len(self.key)

    def __getitem__(self, idx):
        
        key = self.key[idx]
        if self.status == "train":
            category_id, object_id = key.strip().split(' ')
            pc_part_dir = os.path.join(self.imcomplete_path, category_id, object_id)
            pc_gt_dir = os.path.join(self.gt_path, category_id, object_id)

            view_id = random.randint(0, 23)
            pc_gt_path = os.path.join(pc_gt_dir, f'{view_id:02d}.dat')

            while not os.path.exists(pc_gt_path):
                print(f"[Error] {pc_gt_path} not exist!")
                view_id = random.randint(0, 23)
                pc_gt_path = os.path.join(pc_gt_dir, f'{view_id:02d}.dat')
            
            pc_part_path = os.path.join(pc_part_dir, f'{view_id:02d}.dat')
        else:
            category_id, object_id, view_id = key.strip().split()
            pc_part_path = os.path.join(self.imcomplete_path, category_id, object_id, f'{view_id}.dat')
            pc_gt_path = os.path.join(self.gt_path, category_id, object_id, f'{view_id}.dat')
        
        # load partial points
        with open(pc_part_path,'rb') as f:
            pc_part = pickle.load(f).astype(np.float32)
        
        # load gt points
        with open(pc_gt_path,'rb') as f:
            pc = pickle.load(f).astype(np.float32)
        
        # incase some item point number less than 3500 
        pc_part = self.upsample(pc_part, self.pc_input_num)

        # normalize partial point cloud and GT to the same scale
        gt_mean = pc.mean(axis=0) 
        pc = pc - gt_mean
        pc_L_max = np.max(np.sqrt(np.sum(abs(pc ** 2), axis=-1)))
        pc = pc/pc_L_max

        pc_part = pc_part-gt_mean
        pc_part = pc_part/pc_L_max

        return {'taxonomy_id': category_id, 
                'model_id': object_id,
                'pc_gt': torch.from_numpy(pc), 
                'pc_partial': torch.from_numpy(pc_part)}  # pc: (N, 3), pc_part: (N, 3)
        
    def upsample(self, ptcloud, n_points):
        curr = ptcloud.shape[0]
        need = n_points - curr

        if need < 0:
            return ptcloud[np.random.permutation(n_points)]

        while curr <= need:
            ptcloud = np.tile(ptcloud, (2, 1))
            # ptcloud = np.concatenate([ptcloud,np.zeros_like(ptcloud)],dim=0)
            need -= curr
            curr *= 2

        choice = np.random.permutation(need)
        ptcloud = np.concatenate((ptcloud, ptcloud[choice]))

        return ptcloud


@DATASETS.register_module()
class ViPCTestDataset(Dataset):
    def __init__(self, config):
        """For testing the ViPC dataset.

        Args:
            filepath (_type_): _description_
            data_path (_type_): _description_
            status (_type_): _description_
            num_views_limit (_type_, optional): if not None, we set the maximum number of testing view 
                for each object. Defaults to None.
            pc_input_num (int, optional): _description_. Defaults to 3500.
            view_align (bool, optional): _description_. Defaults to False.
            category (str, optional): _description_. Defaults to 'all'.
        """
        
        super(ViPCTestDataset, self).__init__()
        filepath = config.filepath
        data_path = config.data_path
        status = config.subset
        pc_input_num = config.pc_input_num
        category = config.category
        total_views = config.total_views
        num_views_limit = config.num_views_limit

        self.pc_input_num = pc_input_num
        self.status = status
        self.category = category

        self.imcomplete_path = os.path.join(data_path,'ShapeNetViPC-Partial')
        self.gt_path = os.path.join(data_path,'ShapeNetViPC-GT')
        self.rendering_path = os.path.join(data_path,'ShapeNetViPC-View')

        # load the split file
        lines = []
        with open(filepath, 'r') as rf:
            for line in tqdm(rf.readlines()):
                line = line.strip()

                if category != 'all':
                    if line.split(';')[0] != SYNSET_MAP[category]:
                        continue
                lines.append(line)
        rf.close()

        ## choose the testing objects in fact
        if num_views_limit is not None:
            needed_idx = []
            tmp_dict = {}
            # we choose the objects with num_views_limit views at most
            for idx, line in enumerate(lines):
                object_str = line.split(';')[0] + ';' + line.split(';')[1]
                if object_str not in tmp_dict:
                    tmp_dict[object_str] = 1
                else:
                    tmp_dict[object_str] += 1
                if tmp_dict[object_str] <= num_views_limit:
                    needed_idx.append(idx)
            lines = np.array(lines)[needed_idx].tolist()
        
        self.key = lines

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor()
        ])

        print(f'{status} {category} data num: {len(self.key)}')

    def __getitem__(self, idx):
        category_id, object_id, view_id = self.key[idx].split(';')
       
        pc_part_dir = os.path.join(self.imcomplete_path, category_id, object_id)
        pc_gt_dir = os.path.join(self.gt_path, category_id, object_id)

        view_dir = os.path.join(self.rendering_path, category_id, object_id, 'rendering')

        # Read the point cloud
        pc_part_path = os.path.join(pc_part_dir, f'{view_id}.dat')
        pc_gt_path = os.path.join(pc_gt_dir, f'{view_id}.dat')
        view_path = os.path.join(view_dir, f'{view_id}.png')

        ## Load partial points
        with open(pc_part_path,'rb') as f:
            pc_part = pickle.load(f).astype(np.float32)
        ## Load gt
        with open(pc_gt_path,'rb') as f:
            pc = pickle.load(f).astype(np.float32)
        # incase some item point number less than 3500 
        if pc_part.shape[0]<self.pc_input_num:
            pc_part = np.repeat(pc_part,(self.pc_input_num//pc_part.shape[0])+1,axis=0)[0:self.pc_input_num]

        # normalize partial point cloud and GT to the same scale
        gt_mean = pc.mean(axis=0) 
        pc = pc - gt_mean
        pc_L_max = np.max(np.sqrt(np.sum(abs(pc ** 2), axis=-1)))
        pc = pc/pc_L_max

        pc_part = pc_part-gt_mean
        pc_part = pc_part/pc_L_max
        
        ## Load view image
        view = self.transform(Image.open(view_path))[:3, ...]
        
        data_dict = {
            'pc_partial': pc_part,
            'pc_gt': pc,
            'views': view,
            'category_id': category_id,
            'object_id': object_id,
            'view_id': view_id
        }

        return data_dict
    
    def __len__(self):
        return len(self.key)