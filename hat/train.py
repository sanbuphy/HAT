# flake8: noqa
import os.path as osp
import os

import hat.archs
import hat.data
import hat.models
from basicsr.train import train_pipeline
from accelerate import Accelerator, DistributedDataParallelKwargs
from loguru import logger
if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    
    # 配置DDP参数
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    
    # 使用正确的Accelerator参数
    accelerator = Accelerator(
        mixed_precision='fp16',
        device_placement=True,
        kwargs_handlers=[ddp_kwargs],
        log_with="wandb",  # 启用wandb日志
    )
    
    # 设置环境变量，确保wandb只在主进程初始化
    if accelerator.is_main_process:
        logger.info("启用wandb日志")
        os.environ['WANDB_SPAWN_METHOD'] = 'thread'
    
    train_pipeline(root_path, accelerator)
