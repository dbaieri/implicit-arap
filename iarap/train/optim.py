import torch.optim as opt

from types import ModuleType
from typing import Dict, Type, Tuple, List
from dataclasses import dataclass, field

from iarap.config.base_config import FactoryConfig


@dataclass
class OptimizerConfig(FactoryConfig):

    _module: ModuleType = opt
    lr: float = 1e-3

@dataclass
class AdamConfig(OptimizerConfig):

    _name: str = 'Adam'
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0
    amsgrad: bool = False

@dataclass
class SchedulerConfig(FactoryConfig):

    _module: ModuleType = opt.lr_scheduler

@dataclass
class MultiStepSchedulerConfig(SchedulerConfig):

    _name: str = 'MultiStepLR'
    milestones: Tuple[int] = (1000, 2000, 5000)
    gamma: float = 0.5
