import importlib
import numpy as np
import random
import torch
import torch.utils.data
from copy import deepcopy
from functools import partial
from os import path as osp

from basicsr.utils import get_root_logger, scandir
from basicsr.utils.registry import DATASET_REGISTRY

__all__ = ['build_dataset', 'build_dataloader']

# automatically scan and import dataset modules for registry
# scan all the files under the data folder with '_dataset' in file names
data_folder = osp.dirname(osp.abspath(__file__))
dataset_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(data_folder) if v.endswith('_dataset.py')]
# import all the dataset modules
_dataset_modules = [importlib.import_module(f'basicsr.data.{file_name}') for file_name in dataset_filenames]


def build_dataset(dataset_opt):
    """Build dataset from options.

    Args:
        dataset_opt (dict): Configuration for dataset. It must contain:
            name (str): Dataset name.
            type (str): Dataset type.
    """
    dataset_opt = deepcopy(dataset_opt)
    # 确保scale参数从顶层配置传递到数据集配置
    if 'scale' not in dataset_opt and 'scale' in dataset_opt.get('parent_opt', {}):
        dataset_opt['scale'] = dataset_opt['parent_opt']['scale']
    dataset = DATASET_REGISTRY.get(dataset_opt['type'])(dataset_opt)
    logger = get_root_logger()
    logger.info(f'Dataset [{dataset.__class__.__name__}] - {dataset_opt["name"]} is built.')
    return dataset


def build_dataloader(dataset, dataset_opt, num_gpu=1, sampler=None, seed=None):
    """Build dataloader.

    Args:
        dataset (torch.utils.data.Dataset): Dataset.
        dataset_opt (dict): Dataset options. It contains the following keys:
            phase (str): 'train' or 'val'.
            num_worker_per_gpu (int): Number of workers for each GPU.
            batch_size_per_gpu (int): Training batch size for each GPU.
        num_gpu (int): Number of GPUs. Used only in the train phase.
            Default: 1.
        sampler (torch.utils.data.sampler): Data sampler. Default: None.
        seed (int | None): Seed. Default: None
    """
    phase = dataset_opt['phase']
    if phase == 'train':
        batch_size = dataset_opt['batch_size_per_gpu']
        num_workers = dataset_opt['num_worker_per_gpu']
        dataloader_args = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            num_workers=num_workers,
            sampler=sampler,
            drop_last=True,
            pin_memory=True)
        if seed is not None:
            dataloader_args['worker_init_fn'] = partial(worker_init_fn, num_workers=num_workers, seed=seed)
    elif phase in ['val', 'test']:  # validation
        dataloader_args = dict(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)
    else:
        raise ValueError(f"Wrong dataset phase: {phase}. Supported ones are 'train', 'val' and 'test'.")

    return torch.utils.data.DataLoader(**dataloader_args)


def worker_init_fn(worker_id, num_workers, seed):
    # Set the worker seed to num_workers * worker_id + seed
    worker_seed = num_workers * worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
