from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from datasets.modelnet import ModelNet, ModelNet40
from datasets.scannet import ScanObjectNN
from datasets.shapenet import ShapeNet

from datasets.transform import *


def finetune_data(datatype, num_points, args):
    train_transforms = transforms.Compose([
        # PointcloudSphereCrop(p_keep=args.crop),
        Jitter(sigma=0.001, clip=0.0025),
        PointcloudToTensor(),
        # PointcloudNormalize(),
        # PointcloudUpSampling(max_num_points=num_points * 2, centroid="random"),
        # PointcloudRandomCrop(p=0.95, min_num_points=num_points),
        # PointcloudRandomCutout(p=0.5, min_num_points=num_points),
        PointcloudScale(p=1),
        PointcloudRotatePerturbation(p=1),
        # PointcloudTranslate(p=1),
        # PointcloudRandomInputDropout(p=1),
    ])

    eval_transforms = transforms.Compose([
        PointcloudToTensor(),
        # PointcloudNormalize(),
    ])
    if datatype == "ModelNet40":
        print("Dataset: ModelNet40")
        train_dset = ModelNet(args.root, num_points=num_points, partition='train', transforms=train_transforms,
                              unsup=False,
                              xyz_only=(not args.normal))
        test_dset = ModelNet(args.root, num_points=num_points, partition='test', transforms=eval_transforms,
                             unsup=False,
                             xyz_only=(not args.normal))
    elif datatype == "ShapeNet":
        print("Dataset: ShapeNet")
        train_dset = ShapeNet(args.root, num_points, partition='train', transforms=train_transforms, unsup=args.unsup)
        test_dset = ShapeNet(args.root, num_points, partition='test', transforms=train_transforms, unsup=args.unsup)
    elif datatype == "ScanNet":
        print("Dataset: ScanObjectNN")
        train_dset = ScanObjectNN(args.root, num_points, partition='train', transforms=train_transforms, unsup=False)
        test_dset = ScanObjectNN(args.root, num_points, partition='test', transforms=train_transforms, unsup=True)
    elif datatype == "ModelNetSVM":
        print("Dataset: ModelNet40")
        train_dset = ModelNet40(args.root, num_points, partition='train', transforms=train_transforms)
        test_dset = ModelNet40(args.root, num_points, partition='test', transforms=eval_transforms)
    else:
        raise NotImplementedError
    return train_dset, test_dset


def pretrain_data(datatype, num_points, svm_n_points, args):
    train_transforms = transforms.Compose([
        # PointcloudSphereCrop(p_keep=args.crop),
        Jitter(sigma=0.001, clip=0.0025),
        PointcloudToTensor(),
        # PointcloudUpSampling(max_num_points=num_points, centroid="random"),
        PointcloudScale(p=1),
        PointcloudRotatePerturbation(p=1),
        PointcloudTranslate(p=1),
        # PointcloudNormalize(),
    ])

    eval_transforms = transforms.Compose([
        PointcloudToTensor(),
        # PointcloudNormalize(),
    ])
    pre_train_dset = ShapeNet(args.root, num_points, partition='train', transforms=train_transforms)
    pre_test_dset = ShapeNet(args.root, num_points, partition='test', transforms=train_transforms)

    if datatype == "ModelNet40":
        print("Dataset: ModelNet40")
        train_dset = ModelNet(args.root, svm_n_points, partition='train', transforms=eval_transforms, unsup=args.unsup,
                              xyz_only=(not args.normal))
        test_dset = ModelNet(args.root, svm_n_points, partition='test', transforms=eval_transforms, unsup=False,
                             xyz_only=(not args.normal))
    elif datatype == "ScanNet":
        print("Dataset: ScanObjectNN")
        train_dset = ScanObjectNN(args.root, svm_n_points, partition='train', transforms=eval_transforms, unsup=True)
        test_dset = ScanObjectNN(args.root, svm_n_points, partition='test', transforms=eval_transforms, unsup=True)
    elif datatype == "ModelNetSVM":
        print("Dataset: ModelNet40")
        train_dset = ModelNet40(args.root, svm_n_points, partition='train', transforms=eval_transforms)
        test_dset = ModelNet40(args.root, svm_n_points, partition='test', transforms=eval_transforms)
    else:
        train_dset, test_dset = pre_train_dset, pre_test_dset

    return pre_train_dset, train_dset, test_dset


def _build_dataloader(dset, mode, batch_size=32, num_workers=4):
    return DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=mode == "train",
        num_workers=num_workers,
        # pin_memory=True,
        drop_last=True, )


def train_dataloader(train_dset):
    train_loader = _build_dataloader(train_dset, mode="train")
    return train_loader


def val_dataloader(val_dset):
    return _build_dataloader(val_dset, mode="val")


def pretrain_dataloader(datatype, num_points, svm_n_points, args, batch_size=32, num_workers=4):
    pre_train_dset, train_dset, test_dset = pretrain_data(datatype, num_points, svm_n_points, args)
    pre_train_loader = _build_dataloader(pre_train_dset, mode="train", batch_size=batch_size, num_workers=num_workers)
    train_loader = _build_dataloader(train_dset, mode="train", batch_size=batch_size, num_workers=num_workers)
    test_loader = _build_dataloader(test_dset, mode="test", batch_size=batch_size,
                                    num_workers=num_workers)
    return pre_train_loader, train_loader, test_loader


def get_dataset(args, train_batch_size=4, test_batch_size=4, num_workers=4):
    train_dset, test_dset = finetune_data(args.datatype, args.num_point, args)
    train_loader = _build_dataloader(train_dset, mode="train", batch_size=train_batch_size,
                                     num_workers=num_workers)
    test_loader = _build_dataloader(test_dset, mode="test", batch_size=test_batch_size,
                                    num_workers=num_workers)
    return train_loader, test_loader
