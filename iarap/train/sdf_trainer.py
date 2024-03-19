from __future__ import annotations

import torch
import os

from typing import Type
from dataclasses import dataclass, field

from iarap.config.base_config import InstantiateConfig
from iarap.data.mesh import MeshDataConfig
from iarap.model.nn.loss import IGRConfig
from iarap.model.neural_sdf import NeuralSDFConfig
from iarap.train.trainer import Trainer
from iarap.train.optim import AdamConfig, MultiStepSchedulerConfig



class SDFTrainer(Trainer):

    def __init__(self, config: SDFTrainerConfig):
        super(SDFTrainer, self).__init__(config)

    def train_step(self, batch):                
        n_surf = self.config.data.surf_sample
        x_in = torch.cat([batch['surface_sample'], batch['space_sample']], dim=1)
        model_out = self.model(x_in, with_grad=True, differentiable_grad=True)

        surf_sdf, space_sdf = torch.split(model_out['dist'], n_surf, dim=1)
        surf_grad, space_grad = torch.split(model_out['grad'], n_surf, dim=1)

        loss_dict = self.loss(surf_sdf, space_sdf, surf_grad, space_grad, batch['normal_sample'])
        return loss_dict
    
    def postprocess(self):
        ckpt_dir = self.logger.dir + '/checkpoints/'
        print("Saving model weights in: {}".format(ckpt_dir))
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(self.model.state_dict(), ckpt_dir + '/neural_sdf.pt')
        


@dataclass
class SDFTrainerConfig(InstantiateConfig):

    _target: Type = field(default_factory=lambda: SDFTrainer)
    num_steps: int = 10000
    data: MeshDataConfig = MeshDataConfig()
    model: NeuralSDFConfig = NeuralSDFConfig()
    loss: IGRConfig = IGRConfig()
    optimizer: AdamConfig = AdamConfig()
    scheduler: MultiStepSchedulerConfig = MultiStepSchedulerConfig()
