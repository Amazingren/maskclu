#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 6/26/2020 4:35 PM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : modelnet.py
# @Software: PyCharm
import glob
import os
import pickle
import sys

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from datasets.transform import farthest_point_sample, pc_normalize_np
from tools.logger import print_log

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../data_utils'))

modelnet10_label = np.array([2, 3, 9, 13, 15, 23, 24, 31, 34, 36]) - 1


def load_scanobjectnn(root, partition):
    data_dir = os.path.join(root, 'ScanObjectNN', 'main_split')
    h5_name = os.path.join(data_dir, f'{partition}.h5')

    with h5py.File(h5_name, 'r') as f:
        data = f['datasets'][:].astype('float32')  # type: ignore
        label = f['label'][:].astype('int64')  # type: ignore

    return data, label


def load_modelnet_data(data_dir, partition):
    all_data = []
    all_label = []
    pattern = os.path.join(data_dir, 'modelnet40_ply_hdf5_2048', f'ply_data_{partition}*.h5')

    for h5_name in glob.glob(pattern):
        with h5py.File(h5_name, 'r') as f:
            data = f['datasets'][:].astype('float32')  # type: ignore
            label = f['label'][:].astype('int64')  # type: ignore
            all_data.append(data)
            all_label.append(label)

    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)

    return all_data, all_label


class ModelNetFewShot(Dataset):
    def __init__(self, config):
        self.root = config.DATA_PATH
        self.npoints = config.N_POINTS
        self.use_normals = config.USE_NORMALS
        self.num_category = config.NUM_CATEGORY
        self.process_data = True
        self.uniform = True
        split = config.subset
        self.subset = config.subset

        self.way = config.way
        self.shot = config.shot
        self.fold = config.fold
        if self.way == -1 or self.shot == -1 or self.fold == -1:
            raise RuntimeError()

        self.pickle_path = os.path.join(self.root, f'{self.way}way_{self.shot}shot', f'{self.fold}.pkl')

        print_log('Load processed data from %s...' % self.pickle_path, logger='ModelNetFewShot')

        with open(self.pickle_path, 'rb') as f:
            self.dataset = pickle.load(f)[self.subset]

        print_log('The size of %s data is %d' % (split, len(self.dataset)), logger='ModelNetFewShot')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        points, label, _ = self.dataset[index]
        points[:, 0:3] = pc_normalize_np(points[:, 0:3])
        if not self.use_normals:
            points = points[:, 0:3]

        pt_idxs = np.arange(0, points.shape[0])  # 2048
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)
        current_points = points[pt_idxs].copy()
        current_points = torch.from_numpy(current_points).float()
        return 'ModelNet', 'sample', (current_points, label)


class ModelNet(Dataset):
    def __init__(self, config):
        self.root = config.DATA_PATH
        self.npoints = config.N_POINTS
        self.use_normals = config.USE_NORMALS
        self.num_category = config.NUM_CATEGORY
        self.process_data = True
        self.uniform = True
        split = config.subset
        self.subset = config.subset

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print_log('The size of %s data is %d' % (split, len(self.datapath)), logger='ModelNet')

        if self.uniform:
            self.save_path = os.path.join(self.root,
                                          'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(self.root,
                                          'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))

        if self.process_data:
            if not os.path.exists(self.save_path):
                print_log('Processing data %s (only running in the first time)...' % self.save_path, logger='ModelNet')
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index]
                    cls = self.classes[self.datapath[index][0]]
                    cls = np.array([cls]).astype(np.int32)
                    point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                    if self.uniform:
                        point_set = farthest_point_sample(point_set, self.npoints)
                    else:
                        point_set = point_set[0:self.npoints, :]

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print_log('Load processed data from %s...' % self.save_path, logger='ModelNet')
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            label = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]

        point_set[:, 0:3] = pc_normalize_np(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        return point_set, label[0]

    def __getitem__(self, index):
        points, label = self._get_item(index)
        pt_idxs = np.arange(0, points.shape[0])  # 2048
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)
        current_points = points[pt_idxs].copy()
        current_points = torch.from_numpy(current_points).float()
        return 'ModelNet', 'sample', (current_points, label)


class ModelNet40Cls(Dataset):
    def __init__(self, root, num_points, transforms=None, train=True, normalize=True,
                 xyz_only=True, uniform=False, subset10=False, cache_size=15000):
        self.root = root
        self.npoints = num_points
        self.transforms = transforms
        self.uniform = uniform
        self.normalize = normalize
        self.xyz_only = xyz_only
        self.cache_size = cache_size
        self.cache = {}  # from index to (point_set, cls) tuple

        subset = 'modelnet10' if subset10 else 'modelnet40'
        self.catfile = os.path.join(root, f'{subset}_shape_names.txt')
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        train_name = f'{subset}_train.txt'
        test_name = f'{subset}_test.txt'
        shape_ids = {'train': [line.rstrip() for line in open(os.path.join(root, train_name))],
                     'test': [line.rstrip() for line in open(os.path.join(root, test_name))]}
        split = 'train' if train else 'test'
        self.datapath = [
            ('_'.join(x.split('_')[:-1]), os.path.join(root, '_'.join(x.split('_')[:-1]), x) + '.txt')
            for x in shape_ids[split]
        ]

        print(f'The size of {split} datasets is {len(self.datapath)}')

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            shape_name, file_path = self.datapath[index]
            cls = self.classes[shape_name]
            point_set = np.loadtxt(file_path, delimiter=',').astype(np.float32)
            if self.uniform:
                point_set = point_set[:self.npoints]
            else:
                point_set = farthest_point_sample(point_set, self.npoints)
            if self.normalize:
                point_set[:, :3] = pc_normalize_np(point_set[:, :3])
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, np.array([cls], dtype=np.int32))

        if self.xyz_only:
            point_set = point_set[:, :3]
        if self.transforms:
            point_set = self.transforms(point_set)

        return point_set, cls

    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    import open3d as o3d

    root = '/Users/ryan/Documents/datasets/modelnet40_normal_resampled/'
    # Load ModelNet40 dataset
    ps = os.path.join(root, 'airplane', 'airplane_0050.txt')
    points = np.loadtxt(ps, delimiter=',').astype(np.float32)

    # Create a 3D plot to visualize the point cloud
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(airplane_points[:, 0], airplane_points[:, 1], airplane_points[:, 2], s=1, c='blue')
    # plt.show()

    # 读取点云文件
    # 将 NumPy 数组转换为 Open3D 的点云对象
    print(points.shape)
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(points[:, 0:3])
    source_pcd.normals = o3d.utility.Vector3dVector(points[:, 3:6])
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(points[:, 0:3])
    target_pcd.normals = o3d.utility.Vector3dVector(points[:, 3:6])
