import os
import time
import torch
from collections import OrderedDict
from copy import deepcopy
from torch.nn.parallel import DataParallel, DistributedDataParallel

from basicsr.models import lr_scheduler as lr_scheduler
from basicsr.utils import get_root_logger


class BaseModel():
    """Base model."""

    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu')
        self.is_train = opt['is_train']
        self.schedulers = []
        self.optimizers = []
        self.model_ema = None
        
        # 设置 accelerator
        if 'accelerator' in opt:
            self.accelerator = opt['accelerator']
            if self.accelerator is not None:
                self.device = self.accelerator.device

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def save(self, epoch, current_iter):
        """Save networks and training state."""
        pass

    def validation(self, dataloader, current_iter, tb_logger, save_img=False):
        """Validation function.

        Args:
            dataloader (torch.utils.data.DataLoader): Validation dataloader.
            current_iter (int): Current iteration.
            tb_logger (tensorboard logger): Tensorboard logger.
            save_img (bool): Whether to save images. Default: False.
        """
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def get_current_log(self):
        return self.log_dict

    def model_to_device(self, net):
        """Model to device. It also warps models with DistributedDataParallel or
        Accelerate when needed.
        """
        net = net.to(self.device)
        
        # 如果使用 Accelerate，让 Accelerator 处理模型
        if hasattr(self, 'accelerator') and self.accelerator is not None:
            return net
            
        return net

    def get_bare_model(self, net):
        """Get bare model, especially under wrapping with
        DistributedDataParallel.
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module
        return net

    def print_network(self, net):
        """Print the str and parameter number of a network.

        Args:
            net (nn.Module)
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module
        net_cls_str = f'{net.__class__.__name__}'
        net_str = str(net)
        net_params = sum(map(lambda x: x.numel(), net.parameters()))
        logger = get_root_logger()
        logger.info(
            f'Network: {net_cls_str}, with parameters: {net_params:,d}')
        logger.info(net_str)

    def _set_lr(self, lr_groups_l):
        """Set learning rate for warmup.

        Args:
            lr_groups_l (list): List for lr_groups, each for an optimizer.
        """
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group['lr'] = lr

    def _get_init_lr(self):
        """Get the initial lr, which is set by the scheduler.
        """
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append(
                [v['initial_lr'] for v in optimizer.param_groups])
        return init_lr_groups_l

    def update_learning_rate(self, current_iter, warmup_iter=-1):
        """Update learning rate.

        Args:
            current_iter (int): Current iteration.
            warmup_iter (int)： Warmup iter numbers. -1 for no warmup.
                Default： -1.
        """
        if current_iter > 1:
            for scheduler in self.schedulers:
                scheduler.step()
        # set up warm-up learning rate
        if current_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            # currently only support linearly warm up
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append(
                    [v / warmup_iter * current_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)

    def get_current_learning_rate(self):
        return [
            param_group['lr']
            for param_group in self.optimizers[0].param_groups
        ]

    def get_optimizer(self, optim_type, params, **kwargs):
        """Get optimizer by optimizer type.

        Args:
            optim_type (str): Optimizer type.
            params (list[torch.nn.Parameter] | list[dict]): Parameters to optimize.
            **kwargs: Other arguments for optimizer.

        Returns:
            torch.optim.Optimizer: Optimizer.
        """
        # 检查是否使用 Accelerate
        if hasattr(self, 'accelerator'):
            # 如果使用了 Accelerate，我们需要确保参数在正确的设备上
            if isinstance(params, (list, tuple)) and isinstance(params[0], dict):
                for param_group in params:
                    param_group['params'] = [p.to(self.device) for p in param_group['params']]
            else:
                params = [p.to(self.device) for p in params]

        if optim_type == 'Adam':
            optimizer = torch.optim.Adam(params, **kwargs)
        elif optim_type == 'AdamW':
            optimizer = torch.optim.AdamW(params, **kwargs)
        elif optim_type == 'SGD':
            optimizer = torch.optim.SGD(params, **kwargs)
        elif optim_type == 'Lion':  # 添加对 Lion 优化器的支持
            try:
                from lion_pytorch import Lion
                optimizer = Lion(params, **kwargs)
            except ImportError:
                raise ImportError("Please install lion_pytorch to use Lion optimizer")
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supported yet.')

        return optimizer

    def setup_schedulers(self):
        """Set up schedulers."""
        train_opt = self.opt['train']
        # 如果没有scheduler配置，直接返回
        if 'scheduler' not in train_opt:
            return
            
        scheduler_type = train_opt['scheduler'].pop('type')
        
        # 根据调度器类型创建调度器
        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            for optimizer in self.optimizers:
                scheduler = lr_scheduler.MultiStepRestartLR(
                    optimizer=optimizer,
                    **train_opt['scheduler']
                )
                if hasattr(self, 'accelerator') and self.accelerator is not None:
                    scheduler = self.accelerator.prepare(scheduler)
                self.schedulers.append(scheduler)
                
        elif scheduler_type == 'CosineAnnealingLR':
            for optimizer in self.optimizers:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer,
                    **train_opt['scheduler']
                )
                if hasattr(self, 'accelerator') and self.accelerator is not None:
                    scheduler = self.accelerator.prepare(scheduler)
                self.schedulers.append(scheduler)
                
        elif scheduler_type == 'CosineAnnealingRestartLR':
            for optimizer in self.optimizers:
                scheduler = lr_scheduler.CosineAnnealingRestartLR(
                    optimizer=optimizer,
                    **train_opt['scheduler']
                )
                if hasattr(self, 'accelerator') and self.accelerator is not None:
                    scheduler = self.accelerator.prepare(scheduler)
                self.schedulers.append(scheduler)
                
        elif scheduler_type == 'LinearLR':
            for optimizer in self.optimizers:
                scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer=optimizer,
                    **train_opt['scheduler']
                )
                if hasattr(self, 'accelerator') and self.accelerator is not None:
                    scheduler = self.accelerator.prepare(scheduler)
                self.schedulers.append(scheduler)
                
        elif scheduler_type == 'ExponentialLR':
            for optimizer in self.optimizers:
                scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer=optimizer,
                    **train_opt['scheduler']
                )
                if hasattr(self, 'accelerator') and self.accelerator is not None:
                    scheduler = self.accelerator.prepare(scheduler)
                self.schedulers.append(scheduler)
                
        else:
            raise NotImplementedError(f'Scheduler {scheduler_type} is not implemented yet.')

    def save_network(self, net, net_label, current_iter, param_key='params'):
        """Save networks.

        Args:
            net (nn.Module | list[nn.Module]): Network(s) to be saved.
            net_label (str): Network label.
            current_iter (int): Current iter number.
            param_key (str | list[str]): The parameter key(s) to save network.
                Default: 'params'.
        """
        if current_iter == -1:
            current_iter = 'latest'
        save_filename = f'{net_label}_{current_iter}.pth'
        save_path = os.path.join(self.opt['path']['models'], save_filename)

        net = net if isinstance(net, list) else [net]
        param_key = param_key if isinstance(param_key, list) else [param_key]
        assert len(net) == len(param_key), 'The lengths of net and param_key should be the same.'

        save_dict = {}
        for net_, param_key_ in zip(net, param_key):
            net_ = self.get_bare_model(net_)
            state_dict = net_.state_dict()
            for key, param in state_dict.items():
                if key.startswith('module.'):  # remove unnecessary 'module.'
                    key = key[7:]
                state_dict[key] = param.cpu()
            save_dict[param_key_] = state_dict
        torch.save(save_dict, save_path)

    def _print_different_keys_loading(self, crt_net, load_net, strict=True):
        """Print keys with different name or different size when loading models.

        1. Print keys with different names.
        2. If strict=False, print the same key but with different tensor size.
            It also ignore these keys with different sizes.

        Args:
            crt_net (torch model): Current network.
            load_net (dict): Loaded network.
            strict (bool): Whether strictly loaded. Default: True.
        """
        crt_net = self.get_bare_model(crt_net)
        crt_net = crt_net.state_dict()
        crt_net_keys = set(crt_net.keys())
        load_net_keys = set(load_net.keys())

        logger = get_root_logger()
        if crt_net_keys != load_net_keys:
            logger.warning('Current net - loaded net:')
            for v in sorted(list(crt_net_keys - load_net_keys)):
                logger.warning(f'  {v}')
            logger.warning('Loaded net - current net:')
            for v in sorted(list(load_net_keys - crt_net_keys)):
                logger.warning(f'  {v}')

        # check the size for the same keys
        if not strict:
            common_keys = crt_net_keys & load_net_keys
            for k in common_keys:
                if crt_net[k].size() != load_net[k].size():
                    logger.warning(
                        f'Size different, ignore [{k}]: crt_net: '
                        f'{crt_net[k].shape}; load_net: {load_net[k].shape}')
                    load_net[k + '.ignore'] = load_net.pop(k)

    def load_network(self, net, load_path, strict=True, param_key='params'):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        logger = get_root_logger()
        net = self.get_bare_model(net)
        load_net = torch.load(
            load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            if param_key not in load_net and 'params' in load_net:
                param_key = 'params'
                logger.info('Loading: params_ema does not exist, use params.')
            load_net = load_net[param_key]
        logger.info(
            f'Loading {net.__class__.__name__} model from {load_path}, with param key: [{param_key}].')
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        self._print_different_keys_loading(net, load_net, strict)
        net.load_state_dict(load_net, strict=strict)

    def save_training_state(self, epoch, current_iter):
        """Save training states during training, which will be used for
        resuming.

        Args:
            epoch (int): Current epoch.
            current_iter (int): Current iteration.
        """
        if current_iter != -1:
            state = {
                'epoch': epoch,
                'iter': current_iter,
                'optimizers': [],
                'schedulers': []
            }
            for o in self.optimizers:
                state['optimizers'].append(o.state_dict())
            for s in self.schedulers:
                state['schedulers'].append(s.state_dict())
            save_filename = f'{current_iter}.state'
            save_path = os.path.join(self.opt['path']['training_states'],
                                   save_filename)
            torch.save(state, save_path)

    def resume_training(self, resume_state):
        """Reload the optimizers and schedulers for resumed training.

        Args:
            resume_state (dict): Resume state.
        """
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(
            self.optimizers), 'Wrong lengths of optimizers'
        assert len(resume_schedulers) == len(
            self.schedulers), 'Wrong lengths of schedulers'
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)

    def reduce_loss_dict(self, loss_dict):
        """reduce loss dict.

        In distributed training, it averages the losses among different GPUs .

        Args:
            loss_dict (OrderedDict): Loss dict.
        """
        with torch.no_grad():
            log_dict = OrderedDict()
            for name, value in loss_dict.items():
                log_dict[name] = value.mean().item()
            return log_dict

    def model_ema_update(self, decay=0.999):
        """Model exponential moving average (EMA) update.

        Args:
            decay (float): EMA decay. Default: 0.999.
        """
        if self.model_ema is None:
            return
            
        net_g = self.get_bare_model(self.net_g)
        net_g_params = dict(net_g.named_parameters())
        net_g_ema_params = dict(self.net_g_ema.named_parameters())

        for k in net_g_ema_params.keys():
            if decay == 0:
                net_g_ema_params[k].data.copy_(net_g_params[k].data)
            else:
                net_g_ema_params[k].data.mul_(decay).add_(net_g_params[k].data, alpha=1 - decay)

    def setup_ema_model(self, decay=0.999, model=None):
        """Setup EMA model.

        Args:
            decay (float): EMA decay. Default: 0.999.
            model (nn.Module): Model to setup EMA. Default: None.
        """
        if model is None:
            model = self.net_g
            
        # 创建 EMA 模型
        self.model_ema = deepcopy(model)
        self.model_ema.eval()
        self.model_ema.requires_grad_(False)
        
        # 如果使用 Accelerate，需要将 EMA 模型也放到正确的设备上
        if hasattr(self, 'accelerator') and self.accelerator is not None:
            self.model_ema = self.model_ema.to(self.device)
            
        # 初始化 EMA 模型参数
        self.model_ema_update(decay=0)
