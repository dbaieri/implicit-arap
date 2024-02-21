from __future__ import annotations

import torch
import numpy as np

import iarap.data as data
import iarap.model.nn as nn

from typing import Dict, Literal, Tuple, Type
from dataclasses import dataclass, field
from tqdm import tqdm
from pathlib import Path

from iarap.config.base_config import InstantiateConfig
from iarap.model.sdf import NeuralSDF, NeuralSDFConfig
from iarap.train.optim import AdamConfig, MultiStepSchedulerConfig
from iarap.train.trainer import Trainer
from iarap.utils import delaunay, detach_model



class DeformTrainer(Trainer):

    def __init__(self, config: DeformTrainerConfig):
        super(DeformTrainer, self).__init__(config)
        self.device = self.config.device

    def setup_data(self):
        self.loader = [0]

    def setup_model(self):
        self.source: NeuralSDF = self.config.model.setup().to(self.config.device).eval()
        detach_model(self.source)
        self.model: NeuralSDF = self.config.model.setup().to(self.config.device).train()
        self.source.load_state_dict(torch.load(self.config.pretrained_shape))
        if self.config.pretrained_init:
            self.model.load_state_dict(torch.load(self.config.pretrained_shape))
        # self.loss = self.config.loss.setup()

    def train_step(self, batch):

        plane_coords = np.random.uniform(-1, 1, size=(self.config.delaunay_sample, 2)) * self.config.plane_coords_scale
        triangles = delaunay(pts_np=plane_coords, out_device=self.device)

        surf_sample = self.source.sample_zero_level_set(self.config.zero_samples,
                                                        self.config.near_surface_threshold,
                                                        self.config.attempts_per_step,
                                                        self.config.domain_bounds,
                                                        self.config.num_projections).detach()
        tangent_planes = self.source.tangent_plane(surf_sample)
        plane_coords = torch.cat([torch.from_numpy(plane_coords).to(self.device, torch.float),
                                  torch.zeros(*plane_coords.shape[:-1], 1, device=self.device)], dim=-1)
        tangent_coords = (tangent_planes.unsqueeze(1) @ plane_coords.view(1, -1, 3, 1)).squeeze() 
        tangent_pts = tangent_coords + surf_sample.unsqueeze(1) 
        surface_verts = tangent_pts
        for it in range(self.config.num_projections):
            surface_verts = self.source.project_nearest(surface_verts)  # n m 3

        # Apply triangulation to each set of m points in surface_verts
        triangles = triangles.unsqueeze(0) + (surface_verts.shape[1] * torch.arange(0, surface_verts.shape[0], device=self.device).view(-1, 1, 1))
        import vedo
        mesh = vedo.Mesh([surface_verts.view(-1, 3).cpu().detach(), triangles.view(-1, 3).cpu().long()])
        vedo.show(mesh).close()

        loss_dict = self.loss()
        return loss_dict
    
    def postprocess(self):
        return


@dataclass
class DeformTrainerConfig(InstantiateConfig):

    _target: Type = field(default_factory=lambda: DeformTrainer)

    num_steps: int = 1
    pretrained_init: bool = True
    pretrained_shape: Path = Path('./assets/weights/armadillo.pt')
    delaunay_sample: int = 100
    zero_samples: int = 100
    attempts_per_step: int = 10000
    near_surface_threshold: float = 0.05
    domain_bounds: Tuple[float, float] = (-1, 1)
    num_projections: int = 5
    plane_coords_scale: float = 0.05
    device: Literal['cpu', 'cuda'] = 'cuda'

    model: NeuralSDFConfig = NeuralSDFConfig()
    loss: None = None
    optimizer: AdamConfig = AdamConfig()
    scheduler: MultiStepSchedulerConfig = MultiStepSchedulerConfig()