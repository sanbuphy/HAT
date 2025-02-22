import datetime
import logging
import time

def get_env_info():
    """Get environment information.
    """
    import torch
    import torchvision

    msg = '\n'
    msg += ('\nVersion Information: '
            f'\n\tPyTorch: {torch.__version__}'
            f'\n\tTorchVision: {torchvision.__version__}')
    msg += '\n'

    return msg

def init_tb_logger(log_dir):
    from torch.utils.tensorboard import SummaryWriter
    tb_logger = SummaryWriter(log_dir=log_dir)
    return tb_logger

def init_wandb_logger(opt):
    """We now only use wandb to sync tensorboard log."""
    import wandb
    logger = get_root_logger()

    project = opt['logger']['wandb']['project']
    resume_id = opt['logger']['wandb'].get('resume_id')
    if resume_id:
        wandb_id = resume_id
        resume = 'allow'
        logger.warning(f'Resume wandb logger with id={wandb_id}.')
    else:
        wandb_id = wandb.util.generate_id()
        resume = 'never'

    # Initialize wandb environment
    wandb.init(
        project=project,
        id=wandb_id,
        resume=resume,
        dir=opt['root_path'],
        settings=wandb.Settings(start_method="thread"))

    logger.info(f'Use wandb logger with id={wandb_id}; project={project}.')

class MessageLogger():
    """Message logger for printing.

    Args:
        opt (dict): Config. It contains the following keys:
            name (str): Exp name.
            logger (dict): Contains 'print_freq' (str) for logger interval.
            train (dict): Contains 'total_iter' (int) for total iters.
            use_tb_logger (bool): Use tensorboard logger.
        start_iter (int): Start iter. Default: 1.
        tb_logger (obj:`tb_logger`): Tensorboard logger. Default： None.
    """

    def __init__(self, opt, start_iter=1, tb_logger=None):
        self.exp_name = opt['name']
        self.interval = opt['logger']['print_freq']
        self.start_iter = start_iter
        self.max_iters = opt['train']['total_iter']
        self.use_tb_logger = opt['logger']['use_tb_logger']
        self.tb_logger = tb_logger
        self.start_time = time.time()
        self.logger = get_root_logger()
        
        # 初始化时间统计
        self.iter_times = []
        self.window_size = 100  # 使用最近100次迭代的平均值
        self.total_time = 0
        self.n_samples = 0

    def reset_start_time(self):
        self.start_time = time.time()
        self.iter_times = []
        self.total_time = 0
        self.n_samples = 0

    def __call__(self, log_vars):
        """Format logging message.

        Args:
            log_vars (dict): It contains the following keys:
                epoch (int): Epoch number.
                iter (int): Current iter.
                lrs (list): List for learning rates.

                time (float): Iter time.
                data_time (float): Data time for each iter.
        """
        # epoch, iter, learning rates
        epoch = log_vars.pop('epoch')
        current_iter = log_vars.pop('iter')
        lrs = log_vars.pop('lrs')

        message = (f'[{self.exp_name[:5]}..][epoch:{epoch:3d}, iter:{current_iter:8,d}, lr:(')
        for v in lrs:
            message += f'{v:.3e},'
        message += ')] '

        # time and estimated time
        if 'time' in log_vars.keys():
            iter_time = log_vars.pop('time')
            data_time = log_vars.pop('data_time')

            # 更新时间统计
            self.iter_times.append(iter_time)
            if len(self.iter_times) > self.window_size:
                self.iter_times.pop(0)
            
            # 使用滑动窗口计算平均速度
            avg_time = sum(self.iter_times) / len(self.iter_times)
            
            # 计算总体平均速度(用于稳定性检查)
            self.total_time += iter_time
            self.n_samples += 1
            overall_avg = self.total_time / self.n_samples
            
            # 使用滑动窗口平均和总体平均的加权组合
            # 随着训练进行,更倾向于使用滑动窗口平均
            weight = min(self.n_samples / 1000, 0.8)  # 最大权重0.8
            effective_time = weight * avg_time + (1 - weight) * overall_avg
            
            # 计算剩余时间
            remaining_iters = self.max_iters - current_iter
            eta_sec = effective_time * remaining_iters
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            
            message += f'[eta: {eta_str}, '
            message += f'time (data): {iter_time:.3f} ({data_time:.3f})] '

        # other items, especially losses
        for k, v in log_vars.items():
            message += f'{k}: {v:.4e} '
            # tensorboard logger
            if self.use_tb_logger and 'debug' not in self.exp_name:
                if k.startswith('l_'):
                    self.tb_logger.add_scalar(f'losses/{k}', v, current_iter)
                else:
                    self.tb_logger.add_scalar(k, v, current_iter)
        self.logger.info(message)

class AvgTimer():
    """Average timer."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.start_time = 0

    def start(self):
        """Start the timer."""
        self.start_time = time.time()

    def record(self):
        self.count += 1
        self.cur_time = time.time()
        if self.start_time == 0:
            self.start_time = self.cur_time
        else:
            self.sum += self.cur_time - self.start_time
            self.avg = self.sum / self.count
            self.start_time = self.cur_time

    def get_avg_time(self):
        return self.avg

def get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=None):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added.

    Args:
        logger_name (str): root logger name. Default: 'basicsr'.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        logging.Logger: The root logger.
    """
    logger = logging.getLogger(logger_name)
    # if the logger has been initialized, just return it
    if logger.hasHandlers():
        return logger

    # Check if we are in a distributed environment
    try:
        from accelerate import Accelerator
        accelerator = Accelerator()
        is_main_process = accelerator.is_main_process
    except ImportError:
        is_main_process = True

    format_str = '%(asctime)s %(levelname)s: %(message)s'
    if is_main_process:
        logging.basicConfig(format=format_str, level=log_level)
    else:
        logging.basicConfig(format=format_str, level=logging.ERROR)

    if log_file is not None and is_main_process:
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter(format_str))
        logger.addHandler(file_handler)

    return logger
