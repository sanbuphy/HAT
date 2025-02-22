import os
import os.path as osp
import torch
import torchvision
from torch.hub import download_url_to_file, get_dir
from urllib.parse import urlparse
import time

def set_random_seed(seed):
    """Set random seeds."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_time_str():
    """Get current time str."""
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())

def mkdir_and_rename(path):
    """mkdirs. If path exists, rename it with timestamp and create a new one.

    Args:
        path (str): Folder path.
    """
    if os.path.exists(path):
        new_name = path + '_archived_' + get_time_str()
        print(f'Path already exists. Rename it to {new_name}', flush=True)
        os.rename(path, new_name)
    os.makedirs(path, exist_ok=True)

def make_exp_dirs(opt):
    """Make dirs for experiments."""
    path_opt = opt['path'].copy()
    if opt['is_train']:
        # 如果experiments_root不存在，则使用默认路径
        if 'experiments_root' not in path_opt:
            path_opt['experiments_root'] = 'experiments'
        
        experiments_root = path_opt.pop('experiments_root')
        experiments_root = osp.join(experiments_root, opt['name'])
        
        # 创建必要的子目录
        required_paths = {
            'log': 'log',
            'tb_logger': 'tb_logger',
            'results': 'results',
            'checkpoint': 'checkpoint',
        }
        
        # 确保基本目录存在
        os.makedirs(experiments_root, exist_ok=True)
        
        # 创建所有必要的子目录
        for key, path in path_opt.items():
            if (key not in ['strict_load', 'pretrain_network_g', 'resume'] and 
                isinstance(path, str)):  # 只处理字符串类型的路径
                os.makedirs(osp.join(experiments_root, path), exist_ok=True)
        
        # 确保必要的子目录存在
        for key, default_path in required_paths.items():
            if key not in path_opt:
                path_opt[key] = default_path
            if isinstance(path_opt[key], str):
                os.makedirs(osp.join(experiments_root, path_opt[key]), exist_ok=True)
        
        opt['path']['experiments_root'] = experiments_root

def check_resume(opt, resume_iter):
    """Check resume states and pretrain_network paths.

    Args:
        opt (dict): Options.
        resume_iter (int): Resume iteration.
    """
    logger = get_root_logger()
    if opt['path']['resume_state']:
        # get all the networks
        networks = [key for key in opt.keys() if key.startswith('network_')]
        flag_pretrain = False
        for network in networks:
            if opt['path'].get(f'pretrain_{network}') is not None:
                flag_pretrain = True
        if flag_pretrain:
            logger.warning('pretrain_network path will be ignored during resuming.')
        # set pretrained model paths
        for network in networks:
            name = f'pretrain_{network}'
            basename = network.replace('network_', '')
            if opt['path'].get('ignore_resume_networks') is None or (network
                    not in opt['path']['ignore_resume_networks']):
                opt['path'][name] = osp.join(opt['path']['models'],
                                           f'{basename}_{resume_iter}.pth')
                logger.info(f"Set {name} to {opt['path'][name]}")

def sizeof_fmt(size, suffix='B'):
    """Get human readable file size.

    Args:
        size (int): File size.
        suffix (str): Suffix. Default: 'B'.

    Return:
        str: Formatted file size.
    """
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(size) < 1024.0:
            return f'{size:3.1f} {unit}{suffix}'
        size /= 1024.0
    return f'{size:3.1f} Y{suffix}'

def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.
    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.
    Returns:
        A generator for all the interested files with relative paths.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive and os.path.isdir(entry.path):
                    yield from _scandir(entry.path, suffix, recursive)

    return _scandir(dir_path, suffix, recursive)
