import os
import numpy as np
import open3d as o3d
import torch
from sklearn import metrics
from torch import nn
from tqdm import tqdm

from models.partseg import SegTransformer

class_choices = ['airplane', 'bag', 'cap', 'car', 'chair', 'earphone', 'guitar', 'knife', 'lamp', 'laptop', 'motorbike',
                 'mug', 'pistol', 'rocket', 'skateboard', 'table']
seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if y.is_cuda:
        return new_y.cuda()
    return new_y


def calculate_shape_IoU(pred_np, seg_np, label, class_choice, visual=False):
    if not visual:
        label = label.squeeze()
    shape_ious = []
    for shape_idx in range(seg_np.shape[0]):
        if not class_choice:
            start_index = index_start[label[shape_idx]]
            num = seg_num[label[shape_idx]]
            parts = range(start_index, start_index + num)
        else:
            parts = range(seg_num[label[0]])
        part_ious = []
        for part in parts:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            if U == 0:
                iou = 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))
    return shape_ious


def save_ply(filepath, points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalize colors to [0, 1]
    o3d.io.write_point_cloud(filepath, pcd)
    print(f"PLY visualization file saved in {filepath}")


def visualization(args, visu, visu_format, data, pred, seg, label, partseg_colors, class_choice):
    global class_indexs
    visu = visu.split('_')
    for i in range(0, data.shape[0]):
        RGB = []
        RGB_gt = []
        skip = False
        classname = class_choices[int(label[i])]
        class_index = class_indexs[int(label[i])]
        if visu[0] != 'all':
            if len(visu) != 1:
                if visu[0] != classname or visu[1] != str(class_index):
                    skip = True
                else:
                    False
            elif visu[0] != classname:
                skip = True
            else:
                False
        elif class_choice is not None:
            skip = True
        else:
            False
        if skip:
            class_indexs[int(label[i])] += 1
        else:
            output_dir = f'{args.exp_name}/visualization/{classname}'
            os.makedirs(output_dir, exist_ok=True)

            for j in range(0, data.shape[2]):
                RGB.append(partseg_colors[int(pred[i][j])])
                RGB_gt.append(partseg_colors[int(seg[i][j])])

            pred_np = [pred[i].cpu().numpy()]
            seg_np = [seg[i].cpu().numpy()]
            xyz_np = data[i].cpu().numpy()
            xyzRGB = np.concatenate((xyz_np.transpose(1, 0), np.array(RGB)), axis=1)
            xyzRGB_gt = np.concatenate((xyz_np.transpose(1, 0), np.array(RGB_gt)), axis=1)
            IoU = calculate_shape_IoU(np.array(pred_np), np.array(seg_np), label[i].cpu().numpy(), class_choice,
                                      visual=True)
            IoU = str(round(IoU[0], 4))
            filepath = f'{output_dir}/{classname}_{class_index}_pred_{IoU}.{visu_format}'
            filepath_gt = f'{output_dir}/{classname}_{class_index}_gt.{visu_format}'

            if visu_format == 'txt':
                np.savetxt(filepath, xyzRGB, fmt='%s', delimiter=' ')
                np.savetxt(filepath_gt, xyzRGB_gt, fmt='%s', delimiter=' ')
                print(f'TXT visualization file saved in {filepath}')
                print(f'TXT visualization file saved in {filepath_gt}')
            elif visu_format == 'ply':
                save_ply(filepath, xyzRGB[:, :3], xyzRGB[:, 3:])
                save_ply(filepath_gt, xyzRGB_gt[:, :3], xyzRGB_gt[:, 3:])
            else:
                print(f'ERROR!! Unknown visualization format: {visu_format}, please use txt or ply.')
                exit()
            class_indexs[int(label[i])] += 1


def vis_part(args, io, test_loader, num_part):
    # Try to load models
    # seg_num_all = test_loader.dataset.seg_num_all
    seg_start_index = test_loader.dataset.seg_start_index
    partseg_colors = test_loader.dataset.partseg_colors
    model = SegTransformer(num_part).cuda()
    # criterion = MODEL.get_loss().cuda()
    model.apply(inplace_relu)
    print('# generator parameters:', sum(param.numel() for param in model.parameters()))

    if args.ckpts is not None:
        model.load_model_from_ckpt(args.ckpts)

    model = nn.DataParallel(model)
    model = model.eval()
    test_true_cls = []
    test_pred_cls = []
    test_true_seg = []
    test_pred_seg = []
    test_label_seg = []
    for data, label, seg in tqdm(test_loader, leave=False):
        seg = seg - seg_start_index
        # label_one_hot = np.zeros((label.shape[0], 16))
        # for idx in range(label.shape[0]):
        #     label_one_hot[idx, label[idx]] = 1
        # label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))
        label_one_hot = to_categorical(label, 16)
        data, label_one_hot, seg = data.cuda(), label_one_hot.cuda(), seg.cuda()
        data = data
        seg_pred = model(data, label_one_hot)
        pred = seg_pred.max(dim=2)[1]
        seg_np = seg.cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        test_true_cls.append(seg_np.reshape(-1))
        test_pred_cls.append(pred_np.reshape(-1))
        test_true_seg.append(seg_np)
        test_pred_seg.append(pred_np)
        test_label_seg.append(label.reshape(-1))
        # visiualization
        visualization(args, args.visu, args.visu_format, data, pred, seg, label, partseg_colors, args.class_choice)
    if args.visu != '':
        print('Visualization Failed: You can only choose a point cloud shape to '
              'visualize within the scope of the test class')
    test_true_cls = np.concatenate(test_true_cls)
    test_pred_cls = np.concatenate(test_pred_cls)
    test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
    test_true_seg = np.concatenate(test_true_seg, axis=0)
    test_pred_seg = np.concatenate(test_pred_seg, axis=0)
    test_label_seg = np.concatenate(test_label_seg)
    test_ious = calculate_shape_IoU(test_pred_seg, test_true_seg, test_label_seg, args.class_choice)
    outstr = ('Test :: test acc: %.6f, test avg acc: %.6f, '
              'test iou: %.6f' % (test_acc, avg_per_class_acc, np.mean(test_ious)))
    io.cprint(outstr)

