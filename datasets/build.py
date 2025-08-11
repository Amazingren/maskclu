import os

import yaml

from datasets.modelnet import ModelNet, ModelNetFewShot
from datasets.scanobject import ScanObjectNN, ScanObjectNNHardest
from datasets.shapenet import ShapeNet


def build_dataset_from_cfg(base_path, others_cfg):
    """
    Build dataset (e.g., ShapeNet, ModelNet40) from the base configuration and additional settings.

    Args:
        base_path (str or EasyDict): Path to the base YAML configuration or configuration dictionary.
        others_cfg (dict): Additional configurations like subset, npoints, etc.

    Returns:
        dataset: Dataset object created based on the configurations.
    """
    # Check if base_path is a string or a dictionary (EasyDict)
    if isinstance(base_path, (str, bytes, os.PathLike)):
        # If base_path is a file path, load the configuration from the file
        with open(base_path, 'r') as f:
            base_cfg = yaml.safe_load(f)
    else:
        # If base_path is already a dict-like object, use it as the base configuration
        base_cfg = base_path

    # Merge the base configuration with the additional options from others_cfg
    base_cfg.update(others_cfg)

    # Ensure the required parameters are present in the config
    assert 'NAME' in base_cfg, "Dataset NAME missing in config"
    assert 'DATA_PATH' in base_cfg, "DATA_PATH missing in config"
    # assert 'PC_PATH' in base_cfg, "PC_PATH missing in config"
    assert 'N_POINTS' in base_cfg, "N_POINTS missing in config"

    # Directly pass the base_cfg as a dictionary to the dataset class
    if base_cfg['NAME'] == 'ShapeNet':
        dataset = ShapeNet(base_cfg)
    elif base_cfg['NAME'] == 'ModelNet':
        dataset = ModelNet(base_cfg)
    elif base_cfg['NAME'] == 'ScanObjectNN':
        dataset = ScanObjectNN(base_cfg)
    elif base_cfg['NAME'] == 'ScanObjectNN_hardest':
        dataset = ScanObjectNNHardest(base_cfg)
    elif base_cfg['NAME'] == 'ModelNetFewShot':
        dataset = ModelNetFewShot(base_cfg)
    else:
        raise ValueError(f"Unknown dataset: {base_cfg['NAME']}")

    return dataset
