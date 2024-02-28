from __future__ import annotations

import vedo
import random
import torch
import numpy as np
import torch.nn.functional as F

import iarap.data as data
import iarap.model.nn as nn

from typing import Dict, Literal, Tuple, Type
from dataclasses import dataclass, field
from tqdm import tqdm
from pathlib import Path

from iarap.config.base_config import InstantiateConfig
from iarap.model.sdf import NeuralSDF, NeuralSDFConfig
from iarap.model.arap import ARAPMesh
from iarap.train.optim import AdamConfig, MultiStepSchedulerConfig
from iarap.train.trainer import Trainer
from iarap.utils import delaunay, detach_model



class DeformTrainer(Trainer):

    def __init__(self, config: DeformTrainerConfig):
        super(DeformTrainer, self).__init__(config)
        self.device = self.config.device
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

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

        surf_sample = self.source.sample_zero_level_set(self.config.zero_samples,
                                                        self.config.near_surface_threshold,
                                                        self.config.attempts_per_step,
                                                        self.config.domain_bounds,
                                                        self.config.num_projections).detach()
        patch_normals = F.normalize(self.source.gradient(surf_sample), dim=-1)
        tangent_planes = self.source.tangent_plane(surf_sample)

        rho = np.sqrt(np.random.uniform(0, self.config.plane_coords_scale, size=(self.config.delaunay_sample, 1)))
        theta = np.random.uniform(0, 2 * np.pi, size=(self.config.delaunay_sample, 1))
        plane_coords = np.concatenate([rho * np.cos(theta), rho * np.sin(theta)], axis=-1)
        # plane_coords = np.random.uniform(-1, 1, size=(self.config.delaunay_sample, 2)) * self.config.plane_coords_scale
        triangles = delaunay(pts_np=plane_coords, out_device=self.device)

        plane_coords = torch.cat([torch.from_numpy(plane_coords).to(self.device, torch.float),
                                  torch.zeros(*plane_coords.shape[:-1], 1, device=self.device)], dim=-1)
        tangent_coords = (tangent_planes.unsqueeze(1) @ plane_coords.view(1, -1, 3, 1)).squeeze() 
        tangent_pts = tangent_coords + surf_sample.unsqueeze(1) 
        surface_verts = tangent_pts
        for it in range(self.config.num_projections):
            # surface_verts = self.source.sphere_trace(surface_verts, patch_normals.unsqueeze(-2))  # n m 3
            surface_verts = self.source.project_nearest(surface_verts)  # n m 3

        # Apply triangulation to each set of m points in surface_verts
        triangles = triangles.unsqueeze(0) + (surface_verts.shape[1] * torch.arange(0, surface_verts.shape[0], device=self.device).view(-1, 1, 1))
        
        surface_verts_flat = surface_verts.view(-1, 3)
        triangles_flat = triangles.view(-1, 3)

        mesh = ARAPMesh(surface_verts_flat, triangles_flat, device=self.device)
        errored_patches = mesh.debug_patches(self.config.delaunay_sample, self.config.zero_samples)
        print(errored_patches)
        print(len(errored_patches))
        colors = torch.zeros_like(surface_verts)
        colors[..., 1] = 255.0
        colors[errored_patches, ..., 0] = 255.0
        colors[errored_patches, ..., 1] = 0.0

        vis_mesh = vedo.Mesh([surface_verts_flat.cpu().detach(), triangles_flat.cpu().long()]).wireframe()
        vis_mesh.pointcolors = colors.view(-1, 3).cpu()
        tangents = vedo.Mesh([tangent_pts.cpu().detach().view(-1, 3), triangles_flat.cpu().long()], c='black').wireframe()
        normals = vedo.Arrows(surf_sample.detach().cpu().view(-1, 3), (surf_sample + patch_normals * 0.1).cpu().detach().view(-1, 3))
        vedo.show(vis_mesh, tangents, normals).close()  # , points).close()
        '''
        import trimesh
        m = trimesh.load('assets\\mesh\\cactus.obj', force='mesh')
        centroid = torch.tensor(m.centroid, device='cuda')
        surface_verts = torch.tensor(m.vertices).float().cuda()[:, [0, 2, 1]]
        triangles = torch.tensor(m.faces).long().cuda()
        surface_verts -= centroid.unsqueeze(0)
        surface_verts /= surface_verts.abs().max()
        surface_verts *= 0.8
        mesh = ARAPMesh(surface_verts, triangles, device=self.device)
        
        fixed_verts = (surface_verts[:, 1] < -0.3).nonzero().squeeze()
        handle_verts = (((surface_verts[:, 0] < 0.2) & (surface_verts[:, 0] > -0.2)) & \
                        ((surface_verts[:, 1] < 0.4) & (surface_verts[:, 1] >  0.2)) & \
                        (surface_verts[:, 2] < -0.2)).nonzero().squeeze()
        handle_verts_pos = surface_verts[handle_verts, :] + torch.tensor([[0.0, -0.2, -0.2]], device=self.device)
        
        mesh.precompute_laplacian()
        mesh.precompute_reduced_laplacian(fixed_verts, handle_verts)
        test = mesh.solve(fixed_verts, handle_verts, handle_verts_pos, n_its=5)
        '''

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
    zero_samples: int = 1000
    attempts_per_step: int = 10000
    near_surface_threshold: float = 0.05
    domain_bounds: Tuple[float, float] = (-1, 1)
    num_projections: int = 5
    plane_coords_scale: float = 0.02
    device: Literal['cpu', 'cuda'] = 'cuda'
    seed: int = 123456

    model: NeuralSDFConfig = NeuralSDFConfig()
    loss: None = None
    optimizer: AdamConfig = AdamConfig()
    scheduler: MultiStepSchedulerConfig = MultiStepSchedulerConfig()