import datetime
import logging
import math
import time
import torch
from os import path as osp
from loguru import logger

from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import build_model
from basicsr.utils import (AvgTimer, MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str,
                           init_tb_logger, init_wandb_logger, make_exp_dirs, mkdir_and_rename, scandir)
from basicsr.utils.options import copy_opt_file, dict2str, parse_options


def init_tb_loggers(opt):
    # initialize wandb logger before tensorboard logger to allow proper sync
    if (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project')
                                                     is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, ('should turn on tensorboard when using wandb')
        # Only initialize wandb logger on the main process
        if opt.get('rank', 0) == 0:
            init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join(opt['root_path'], 'tb_logger', opt['name']))
    return tb_logger


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loaders = None, []
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            logger.info("正在准备训练数据集...")
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            # 传递顶层配置到数据集配置
            dataset_opt['parent_opt'] = opt
            train_set = build_dataset(dataset_opt)
            train_loader = build_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                sampler=None,  # Let accelerator handle the sampling
                seed=opt['manual_seed'])

            total_iters = int(opt['train']['total_iter'])
            
        elif phase.split('_')[0] == 'val':
            logger.info("正在准备验证数据集...")
            # 传递顶层配置到验证数据集配置
            dataset_opt['parent_opt'] = opt
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set, dataset_opt, num_gpu=opt['num_gpu'], sampler=None, seed=opt['manual_seed'])
            logger.info(f'验证集 {dataset_opt["name"]} 中的图像/文件夹数量: {len(val_set)}')
            val_loaders.append(val_loader)
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, val_loaders, train_set, dataset_enlarge_ratio, total_iters


def load_resume_state(opt):
    resume_state_path = None
    if opt['auto_resume']:
        state_path = osp.join('experiments', opt['name'], 'training_states')
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
                opt['path']['resume_state'] = resume_state_path
    else:
        if opt['path'].get('resume_state'):
            resume_state_path = opt['path']['resume_state']

    if resume_state_path is None:
        resume_state = None
    else:
        logger.info(f"正在从{resume_state_path}恢复训练状态...")
        device_id = torch.cuda.current_device()
        resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))
        check_resume(opt, resume_state['iter'])
    return resume_state


def train_pipeline(root_path, accelerator=None):
    # parse options, set distributed setting, set random seed
    opt, args = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path

    # Update distributed training settings based on accelerator if provided
    if accelerator is not None:
        opt['dist'] = True
        opt['rank'] = accelerator.process_index
        opt['world_size'] = accelerator.num_processes

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # load resume states if necessary
    resume_state = load_resume_state(opt)
    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join(opt['root_path'], 'tb_logger', opt['name']))

    # copy the yml file to the experiment root
    copy_opt_file(args.opt, opt['path']['experiments_root'])

    # WARNING: should not use get_root_logger in the above codes, including the called functions
    # Otherwise the logger will not be properly initialized
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    # initialize wandb and tb loggers
    tb_logger = init_tb_loggers(opt)

    # create train and validation dataloaders
    logger.info("开始创建数据加载器...")
    result = create_train_val_dataloader(opt, logger)
    train_loader, val_loaders, train_set, dataset_enlarge_ratio, total_iters = result

    # create model
    logger.info("开始构建模型...")
    if accelerator is not None:
        opt['accelerator'] = accelerator
    model = build_model(opt)
    if resume_state:  # resume training
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"从epoch: {resume_state['epoch']}, iter: {resume_state['iter']}恢复训练.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        start_epoch = 0
        current_iter = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # wrap model and data loaders with accelerator if provided
    if accelerator is not None:
        logger.info("使用accelerator准备模型和数据加载器...")
        model, train_loader = accelerator.prepare(model, train_loader)
        val_loaders = [accelerator.prepare(val_loader) for val_loader in val_loaders]
        
        # 在accelerator准备好后重新计算batch size和epochs
        batch_size_per_gpu = opt['datasets']['train']['batch_size_per_gpu']
        world_size = accelerator.num_processes
        effective_batch_size = batch_size_per_gpu * world_size
    else:
        batch_size_per_gpu = opt['datasets']['train']['batch_size_per_gpu']
        world_size = opt['num_gpu']
        effective_batch_size = batch_size_per_gpu * world_size

    # 重新计算epochs相关信息
    num_iter_per_epoch = math.ceil(len(train_set) * dataset_enlarge_ratio / effective_batch_size)
    total_epochs = math.ceil(total_iters / num_iter_per_epoch)
    
    should_log = True if accelerator is None else accelerator.is_main_process
    if should_log:
        logger.info('训练数据统计信息:'
                    f'\n\t训练图像数量: {len(train_set)}'
                    f'\n\t数据集扩充比例: {dataset_enlarge_ratio}'
                    f'\n\t每个GPU的批次大小: {batch_size_per_gpu}'
                    f'\n\tGPU数量: {world_size}'
                    f'\n\t实际总批次大小: {effective_batch_size}'
                    f'\n\t每个epoch所需迭代次数: {num_iter_per_epoch}'
                    f'\n\t总epochs数: {total_epochs}; 总迭代次数: {total_iters}.')
    
    # dataloader prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'使用 {prefetch_mode} 预取数据加载器')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f"Wrong prefetch_mode {prefetch_mode}. Supported ones are: None, 'cuda', 'cpu'.")

    # training
    if should_log:
        logger.info(f'开始训练,从epoch: {start_epoch}, iter: {current_iter}')
        logger.info(f'总训练epochs数: {total_epochs}')
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()
    last_hundred_iter_time = start_time  # 记录上一个100 iter的时间点

    for epoch in range(start_epoch, total_epochs + 1):
        if should_log:
            logger.info(f'当前正在进行第 {epoch}/{total_epochs} 个epoch的训练')
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_timer.record()

            current_iter += 1
            if current_iter > total_iters:
                break
            # update learning rate
            model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            # training
            model.feed_data(train_data)
            
            if accelerator is not None:
                with accelerator.accumulate(model):
                    model.optimize_parameters(current_iter)
            else:
                model.optimize_parameters(current_iter)
                
            iter_timer.record()
            if current_iter == 1:
                # reset start time in msg_logger for more accurate eta_time
                # not work in resume mode
                msg_logger.reset_start_time()
                last_hundred_iter_time = time.time()  # 重置第一个100 iter的起始时间
            # log
            if current_iter % opt['logger']['print_freq'] == 0:
                # Only log on main process when using accelerator
                should_log = True if accelerator is None else accelerator.is_main_process
                if should_log:
                    log_vars = {'epoch': epoch, 'iter': current_iter}
                    log_vars.update({'lrs': model.get_current_learning_rate()})
                    log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
                    log_vars.update(model.get_current_log())
                    msg_logger(log_vars)

            # 每100个step打印耗时统计
            if current_iter % 100 == 0:
                should_log = True if accelerator is None else accelerator.is_main_process
                if should_log:
                    # 计算这100个iter实际花费的时间
                    current_time = time.time()
                    hundred_iter_cost = current_time - last_hundred_iter_time
                    last_hundred_iter_time = current_time  # 更新时间点
                    
                    # 计算总耗时和预计剩余时间
                    time_sec = int(current_time - start_time)
                    total_time = str(datetime.timedelta(seconds=time_sec))
                    # 使用最近100 iter的速度来预估剩余时间
                    remaining_iters = total_iters - current_iter
                    eta_sec = (remaining_iters / 100) * hundred_iter_cost
                    eta_time = str(datetime.timedelta(seconds=int(eta_sec)))
                    
                    logger.info(f'耗时统计 iter: {current_iter}/{total_iters}'
                              f'\n\t最近100个iter耗时: {hundred_iter_cost:.2f}秒 (速度: {100/hundred_iter_cost:.2f}it/s)'
                              f'\n\t已训练时间: {total_time}'
                              f'\n\t预计剩余时间: {eta_time}')

            # save models and training states
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                if accelerator is not None:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        logger.info('保存模型和训练状态.')
                        model.save(epoch, current_iter)
                else:
                    logger.info('保存模型和训练状态.')
                    model.save(epoch, current_iter)

            # validation
            if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
                if len(val_loaders) > 1:
                    logger.warning('Multiple validation datasets are *only* supported by SRModel.')
                if accelerator is not None:
                    accelerator.wait_for_everyone()
                for val_loader in val_loaders:
                    model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])

            data_timer.start()
            iter_timer.start()
            train_data = prefetcher.next()
        # end of iter

        # 在每个epoch结束时记录训练信息到wandb
        if (opt['logger'].get('wandb') is not None and 
            opt['logger']['wandb'].get('project') is not None and 
            'debug' not in opt['name']):
            should_log = True if accelerator is None else accelerator.is_main_process
            if should_log:
                # 获取当前的训练状态
                log_dict = {}
                # 记录基本信息
                log_dict['epoch'] = epoch
                log_dict['iter'] = current_iter
                log_dict['lr'] = model.get_current_learning_rate()[0]  # 取第一个学习率
                # 记录训练loss等信息
                current_log = model.get_current_log()
                log_dict.update({f'train/{k}': v for k, v in current_log.items()})
                # 记录验证集指标(如果有的话)
                if hasattr(model, 'metric_results'):
                    log_dict.update({f'val/{k}': v for k, v in model.metric_results.items()})
                # 记录时间信息
                log_dict['time/iter'] = iter_timer.get_avg_time()
                log_dict['time/data'] = data_timer.get_avg_time()
                
                # 更新到wandb
                import wandb
                wandb.log(log_dict, step=current_iter)
                
                if should_log:
                    logger.info(f'Epoch {epoch} 训练信息已记录到wandb')

    # end of epoch

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'训练结束. 耗时: {consumed_time}')
    logger.info('保存最新模型.')
    
    if accelerator is not None:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
            if opt.get('val') is not None:
                for val_loader in val_loaders:
                    model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
    else:
        model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
        if opt.get('val') is not None:
            for val_loader in val_loaders:
                model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
                
    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
