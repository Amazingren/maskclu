"""
ScanNet v2 data preprocessing.
Extract point clouds data from .ply files to genrate .pickle files for training and testing.
Author: Wenxuan Wu
Date: July 2018
"""

import os
import pickle

import numpy as np
from plyfile import PlyData
from torch.utils.data import Dataset


def remove_unano(scene_data, scene_label, scene_data_id):
    keep_idx = np.where((scene_label > 0) & (
            scene_label < 41))  # 0: unanotated
    scene_data_clean = scene_data[keep_idx]
    scene_label_clean = scene_label[keep_idx]
    scene_data_id_clean = scene_data_id[keep_idx]
    return scene_data_clean, scene_label_clean, scene_data_id_clean


test_class = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
              10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]


def gen_label_map():
    label_map = np.zeros(41)
    for i in range(41):
        if i in test_class:
            label_map[i] = test_class.index(i)
        else:
            label_map[i] = 0
    # print(label_map)
    return label_map


def gen_pickle(split="val", keep_unanno=False, root="scannet"):
    if split == 'test':
        root_new = root + "/scans_test"
    else:
        root_new = root + "/scans"
    file_list = "scannetv2_%s.txt" % split
    with open(file_list) as fl:
        scene_id = fl.read().splitlines()

    scene_data = []
    scene_data_labels = []
    scene_data_id = []
    scene_data_num = []
    label_map = gen_label_map()
    for i in range(len(scene_id)):  # len(scene_id)
        print('process...', i)
        scene_namergb = os.path.join(
            root_new, scene_id[i], scene_id[i] + '_vh_clean_2.ply')
        scene_xyzlabelrgb = PlyData.read(scene_namergb)
        scene_vertex_rgb = scene_xyzlabelrgb['vertex']
        scene_data_tmp = np.stack((scene_vertex_rgb['x'], scene_vertex_rgb['y'],
                                   scene_vertex_rgb['z'], scene_vertex_rgb['red'],
                                   scene_vertex_rgb['green'], scene_vertex_rgb['blue']), axis=-1).astype(np.float32)
        scene_points_num = scene_data_tmp.shape[0]
        scene_point_id = np.array([c for c in range(scene_points_num)])
        if not keep_unanno:
            scene_name = os.path.join(
                root_new, scene_id[i], scene_id[i] + '_vh_clean_2.labels.ply')
            scene_xyzlabel = PlyData.read(scene_name)
            scene_vertex = scene_xyzlabel['vertex']
            scene_data_label_tmp = scene_vertex['label']
            scene_data_tmp, scene_data_label_tmp, scene_point_id_tmp = remove_unano(
                scene_data_tmp, scene_data_label_tmp, scene_point_id)
            scene_data_label_tmp = label_map[scene_data_label_tmp]
        elif split != 'test':
            scene_name = os.path.join(
                root_new, scene_id[i], scene_id[i] + '_vh_clean_2.labels.ply')
            scene_xyzlabel = PlyData.read(scene_name)
            scene_vertex = scene_xyzlabel['vertex']
            scene_point_id_tmp = scene_point_id
            scene_data_label_tmp = scene_vertex['label']
            scene_data_label_tmp[np.where(scene_data_label_tmp > 40)] = 0
            scene_data_label_tmp = label_map[scene_data_label_tmp]
        else:
            scene_data_label_tmp = np.zeros(
                (scene_data_tmp.shape[0])).astype(np.int32)
            scene_point_id_tmp = scene_point_id
        scene_data.append(scene_data_tmp)
        scene_data_labels.append(scene_data_label_tmp)
        scene_data_id.append(scene_point_id_tmp)
        scene_data_num.append(scene_points_num)

    if not keep_unanno:
        out_path = os.path.join(root, "scannet_%s_rgb21c_pointid.pickle" % (split))
    else:
        out_path = os.path.join(root, "scannet_%s_rgb21c_pointid_keep_unanno.pickle" % (split))
    pickle_out = open(out_path, "wb")
    pickle.dump(scene_data, pickle_out, protocol=0)
    pickle.dump(scene_data_labels, pickle_out, protocol=0)
    pickle.dump(scene_data_id, pickle_out, protocol=0)
    pickle.dump(scene_data_num, pickle_out, protocol=0)
    pickle_out.close()


class ScanNet(Dataset):
    def __init__(self, data_root='scannet', num_point=8192, partition='train', block_size=1.5,
                 sample_rate=1.0, transform=None, use_rgb=False):
        self.partition = partition
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        xyz_all = []
        label_all = []
        if not isinstance(partition, list):
            partition = [partition]
        for i in partition:
            data_file = os.path.join(
                    data_root, 'scannet_{}_rgb21c_pointid.pickle'.format(i))
            file_pickle = open(data_file, 'rb')
            _xyz_all = pickle.load(file_pickle)
            _label_all = pickle.load(file_pickle)
            file_pickle.close()
            xyz_all.append(_xyz_all)
            label_all.append(_label_all)
        xyz_all = np.hstack(xyz_all)
        label_all = np.hstack(label_all)
        self.label_all = []  # for change 0-20 to 0-19 + 255
        self.room_coord_min, self.room_coord_max = [], []
        num_point_all = []
        for index in range(len(xyz_all)):
            xyz, label = xyz_all[index], label_all[index]  # xyzrgb, N*6; l, N
            coord_min, coord_max = np.amin(
                xyz, axis=0)[:3], np.amax(xyz, axis=0)[:3]
            self.room_coord_min.append(coord_min)
            self.room_coord_max.append(coord_max)
            num_point_all.append(label.size)
            # we have set all ignore_class to 0
            # class 0 is also ignored
            # so we set all them as 255
            label_new = label - 1
            label_new[label == 0] = 255
            self.label_all.append(label_new.astype(np.uint8))
        sample_prob = num_point_all / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point)
        room_idxs = []
        for index in range(len(xyz_all)):
            room_idxs.extend(
                [index] * int(round(sample_prob[index] * num_iter)))
        self.room_idxs = np.array(room_idxs)
        self.xyz_all = xyz_all

        # whether load RGB information
        self.use_rgb = use_rgb

        print("Totally {} samples in {} set.".format(len(self.room_idxs), partition))

    def __getitem__(self, idx):
        room_idx = self.room_idxs[idx]
        points = self.xyz_all[room_idx]  # N * 6
        if not self.use_rgb:
            points = points[:, :3]
        labels = self.label_all[room_idx]  # N
        n_points = points.shape[0]

        point_idxs = None
        center = None
        for i in range(10):
            center = points[np.random.choice(n_points)][:3]
            block_min = center - [self.block_size /
                                  2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size /
                                  2.0, self.block_size / 2.0, 0]
            block_min[2] = self.room_coord_min[room_idx][2]
            block_max[2] = self.room_coord_max[room_idx][2]
            point_idxs = np.where((points[:, 0] >= block_min[0]) &
                                  (points[:, 0] <= block_max[0]) &
                                  (points[:, 1] >= block_min[1]) &
                                  (points[:, 1] <= block_max[1]))[0]
            if point_idxs.size == 0:
                continue
            vidx = np.ceil((points[point_idxs, :3] - block_min) /
                           (block_max - block_min) * [31.0, 31.0, 62.0])
            vidx = np.unique(vidx[:, 0] * 31.0 * 62.0 +
                             vidx[:, 1] * 62.0 + vidx[:, 2])
            if ((labels[point_idxs] != 255).sum() / point_idxs.size >= 0.7) and (vidx.size/31.0/31.0/62.0 >= 0.02):
                break

        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(
                point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(
                point_idxs, self.num_point, replace=True)
        # normalize
        selected_points = points[selected_point_idxs, :]  # num_point * 3/6
        num_feats = 9 if self.use_rgb else 6
        current_points = np.zeros((self.num_point, num_feats))  # num_point * 6/9
        current_points[:, -3] = selected_points[:, 0] / self.room_coord_max[room_idx][0]
        current_points[:, -2] = selected_points[:, 1] / self.room_coord_max[room_idx][1]
        current_points[:, -1] = selected_points[:, 2] / self.room_coord_max[room_idx][2]
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        current_points[:, 0:3] = selected_points[:, 0:3]
        if self.use_rgb:
            current_points[:, 3:6] = selected_points[:, 3:6] / 255.
        current_labels = labels[selected_point_idxs]
        if self.transform is not None:
            current_points, current_labels = self.transform(
                current_points, current_labels)
        return current_points, current_labels

    def __len__(self):
        return len(self.room_idxs)


if __name__ == '__main__':
    # modify this path to your Scannet v2 dataset Path
    root = "../data/ScanNet"
    gen_pickle(split='train', keep_unanno=False, root=root)
    gen_pickle(split='val', keep_unanno=False, root=root)
    gen_pickle(split='val', keep_unanno=True, root=root)
    gen_pickle(split='test', keep_unanno=True, root=root)

    print('Done!!!')
