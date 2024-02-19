from __future__ import annotations

import iarap.data as data
import iarap.model.nn as nn

from typing import Dict, Type
from dataclasses import dataclass, field
from tqdm import tqdm

from iarap.config.base_config import InstantiateConfig
from iarap.data.mesh import MeshDataConfig
from iarap.model.sdf import NeuralSDFConfig
from iarap.train.optim import AdamConfig, MultiStepSchedulerConfig



class SDFTrainer:

    def __init__(self, config: SDFTrainerConfig):
        
        self.config = config

        self.setup_data()
        self.setup_model()
        self.setup_optimizer()

    def setup_data(self):
        self.data = self.config.data.setup()

    def setup_model(self):
        self.model = self.config.model.setup()

    def setup_optimizer(self):
        param_groups = list(self.model.parameters())
        self.optimizer = self.config.optimizer.setup(params=param_groups)
        self.scheduler = self.config.scheduler.setup(optimizer=self.optimizer)

    def run(self):
        print(self.config)
        print("Starting SDF training procedure.")
        for it in tqdm(range(self.config.num_steps)):
            pass



@dataclass
class SDFTrainerConfig(InstantiateConfig):

    _target: Type = field(default_factory=lambda: SDFTrainer)
    num_steps: int = 1
    data: MeshDataConfig = MeshDataConfig()
    model: NeuralSDFConfig = NeuralSDFConfig()
    optimizer: AdamConfig = AdamConfig()
    scheduler: MultiStepSchedulerConfig = MultiStepSchedulerConfig()
