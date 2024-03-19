from __future__ import annotations
import os

import vedo
import random
import torch
import numpy as np
import torch.nn.functional as F
import yaml

import iarap.data as data
import iarap.model.nn as nn

from typing import Dict, List, Literal, Tuple, Type
from dataclasses import dataclass, field
from tqdm import tqdm
from pathlib import Path

from iarap.config.base_config import InstantiateConfig
from iarap.model.nn.loss import DeformationLossConfig, PatchARAPLoss
from iarap.model.neural_rtf import NeuralRTF, NeuralRTFConfig
from iarap.model.neural_sdf import NeuralSDF, NeuralSDFConfig
from iarap.model.arap import ARAPMesh
from iarap.train.optim import AdamConfig, MultiStepSchedulerConfig
from iarap.train.trainer import Trainer
from iarap.utils import delaunay, detach_model


DEBUG = False



class DeformTrainer(Trainer):

    def __init__(self, config: DeformTrainerConfig):
        super(DeformTrainer, self).__init__(config)
        self.device = self.config.device
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

    def setup_data(self):
        # Avoids rewriting training function, does a single iteration per epoch
        self.loader = [0]  
        # Configure handles
        handle_cfg = yaml.load(open(self.config.handles_spec, 'r'), yaml.Loader)
        handle_dir = self.config.handles_spec.parent
        static = [np.loadtxt(handle_dir / "parts" / f"{f}.txt") for f in handle_cfg['handles']['static']['positions']]
        moving = [np.loadtxt(handle_dir / "parts" / f"{f}.txt") for f in handle_cfg['handles']['moving']['positions']]
        transforms = [np.loadtxt(handle_dir / "transforms" / f"{f}.txt") for f in  handle_cfg['handles']['moving']['transform']]
        self.handles_static = torch.from_numpy(np.concatenate(static, axis=0)).to(self.config.device, torch.float)
        moving_pos = torch.from_numpy(np.concatenate(moving, axis=0)).to(self.config.device, torch.float)
        moving_trans = torch.from_numpy(np.concatenate(transforms, axis=0)).to(self.config.device, torch.float)
        self.handles_moving = torch.cat([moving_pos, moving_trans], dim=-1)

    def setup_model(self):
        self.source: NeuralSDF = self.config.shape_model.setup().to(self.config.device).eval()
        self.source.load_state_dict(torch.load(self.config.pretrained_shape))
        detach_model(self.source)
        self.model: NeuralRTF = self.config.rotation_model.setup().to(self.config.device).train()
        self.model.set_sdf_callable(self.source.distance)
        self.loss = self.config.loss.setup()

    def sample_domain(self, nsamples):
        scale = self.config.domain_bounds[1] - self.config.domain_bounds[0]
        return torch.rand(nsamples, 3, device=self.device) * scale + self.config.domain_bounds[0]

    def train_step(self, batch):

        handles = torch.cat([self.handles_moving[:, :3], self.handles_static], dim=0)
        for it in range(self.config.num_projections):
            handles = self.source.project_nearest(handles).detach()
        handle_values = torch.cat(
            [self.handles_moving[:, 3:], handles[self.handles_moving.shape[0]:, :]], dim=0
        ).detach()

        surf_sample = self.source.sample_zero_level_set(self.config.zero_samples - handles.shape[0],
                                                        self.config.near_surface_threshold,
                                                        self.config.attempts_per_step,
                                                        self.config.domain_bounds,
                                                        self.config.num_projections).detach()
        surf_sample = torch.cat([handles, surf_sample], dim=0)
        space_sample = self.sample_domain(self.config.space_samples)
        samples = torch.cat([surf_sample, space_sample], dim=0)
        
        sdf_outs = self.source(samples, with_grad=True)
        sample_dist, patch_normals = sdf_outs['dist'], F.normalize(sdf_outs['grad'], dim=-1)
        tangent_planes = self.source.tangent_plane(samples)

        rho = np.sqrt(np.random.uniform(0, self.config.plane_coords_scale, size=(self.config.delaunay_sample-1, 1)))
        theta = np.random.uniform(0, 2 * np.pi, size=(self.config.delaunay_sample-1, 1))
        plane_coords = np.concatenate([rho * np.cos(theta), rho * np.sin(theta)], axis=-1)
        plane_coords = np.concatenate([np.zeros((1, 2)), plane_coords], axis=0)
        # plane_coords = np.random.uniform(-1, 1, size=(self.config.delaunay_sample, 2)) * self.config.plane_coords_scale
        triangles = delaunay(pts_np=plane_coords, out_device=self.device)

        plane_coords = torch.cat([torch.from_numpy(plane_coords).to(self.device, torch.float),
                                  torch.zeros(*plane_coords.shape[:-1], 1, device=self.device)], dim=-1)
        tangent_coords = (tangent_planes.unsqueeze(1) @ plane_coords.view(1, -1, 3, 1)).squeeze() 
        tangent_pts = tangent_coords + samples.unsqueeze(1) 
        level_set_verts = tangent_pts
        for it in range(self.config.num_projections):
            # surface_verts = self.source.sphere_trace(surface_verts, patch_normals.unsqueeze(-2))  # n m 3
            # surface_verts = self.source.project_nearest(surface_verts)  # n m 3
            level_set_verts = self.source.project_level_sets(level_set_verts, sample_dist)  # n m 3
        
        if DEBUG:
            # Apply triangulation to each set of m points in surface_verts
            triangles_all = triangles.unsqueeze(0) + (level_set_verts.shape[1] * torch.arange(0, level_set_verts.shape[0], device=self.device).view(-1, 1, 1))

            surface_verts_flat = level_set_verts.view(-1, 3)
            triangles_flat = triangles_all.view(-1, 3)
            
            # w = self.loss.arap_loss.get_cot_weights(level_set_verts, triangles).sum(dim=-1).view(-1)
            # cmap = vedo.color_map(w.cpu().detach()) * 255.
            # cmap = torch.zeros_like(surface_verts_flat)
            

            vis_mesh = vedo.Mesh([surface_verts_flat.cpu().detach(), triangles_flat.cpu().long()]).wireframe()
            # vis_mesh.pointcolors = cmap
            # tangents = vedo.Mesh([tangent_pts.cpu().detach().view(-1, 3), triangles_flat.cpu().long()], c='black').wireframe()
            normals = vedo.Arrows(samples.detach().cpu().view(-1, 3), (samples + patch_normals * 0.02).cpu().detach().view(-1, 3))
            vedo.show(vis_mesh, normals).close()  
        
        if DEBUG:
            import trimesh
            m = trimesh.load('assets\\mesh\\cactus.obj', force='mesh')
            centroid = torch.tensor(m.centroid, device='cuda')
            mesh_vert = torch.tensor(m.vertices).float().cuda()[:, [0, 2, 1]]
            mesh_triv = torch.tensor(m.faces).long().cuda()
            mesh_vert -= centroid.unsqueeze(0)
            mesh_vert /= mesh_vert.abs().max()
            mesh_vert *= 0.8
            mesh = ARAPMesh(mesh_vert, mesh_triv, device=self.device)
            
            fixed_verts = (mesh_vert[:, 1] < -0.65).nonzero().squeeze()
            handle_verts = ((mesh_vert[:, 1] > 0.65) & \
                            ((mesh_vert[:, 2] > -0.2) & (mesh_vert[:, 2] < 0.2))).nonzero().squeeze()
            handle_verts_pos = mesh_vert[handle_verts, :] + torch.tensor([[0.0, -0.5, 0.5]], device=self.device)
            
            test = mesh.solve(fixed_verts, handle_verts, handle_verts_pos, track_energy=True, n_its=200)
            base_mesh = vedo.Mesh([mesh.vertices.cpu(), mesh.faces.cpu()], c='black').wireframe()
            colors = torch.zeros_like(test)
            colors[:, 2] = 255.0
            colors[fixed_verts, 0] = 255.0
            colors[fixed_verts, 2] = 0.0
            colors[handle_verts, 1] = 255.0
            colors[handle_verts, 2] = 0.0
            base_mesh.pointcolors = colors.cpu()
            arap_mesh = vedo.Mesh([test.cpu().detach(), mesh.faces.cpu()])
            transforms = vedo.Arrows(mesh_vert[handle_verts, :].cpu(), 
                                     handle_verts_pos.cpu(), 
                                     shaft_radius=0.01, head_radius=0.03, head_length=0.1,
                                     alpha=0.4)
            vedo.show(base_mesh, arap_mesh, transforms).close()
            
        rtf_out = self.model(level_set_verts.detach())
        rotations = rtf_out['rot']
        translations = rtf_out['transl']

        handle_idx = torch.stack([
            torch.arange(0, handles.shape[0], device=self.device, dtype=torch.long),
            torch.zeros(handles.shape[0], device=self.device, dtype=torch.long)
        ], dim=-1)
        moving_idx = handle_idx[:self.handles_moving.shape[0], :]
        static_idx = handle_idx[self.handles_moving.shape[0]:, :]
        loss_dict = self.loss(level_set_verts.detach(), 
                              triangles, rotations, translations,
                              moving_idx, static_idx, 
                              handle_values, 2)  # self.step % 2)
        return loss_dict
    
    def postprocess(self):
        ckpt_dir = self.logger.dir + '/checkpoints/'
        print("Saving model weights in: {}".format(ckpt_dir))
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(self.model.state_dict(), ckpt_dir + '/neural_rotation.pt')


@dataclass
class DeformTrainerConfig(InstantiateConfig):

    _target: Type = field(default_factory=lambda: DeformTrainer)

    num_steps: int = 1
    pretrained_shape: Path = None
    handles_spec: Path = None
    delaunay_sample: int = 100
    zero_samples: int = 1000
    space_samples: int = 1000
    attempts_per_step: int = 10000
    near_surface_threshold: float = 0.05
    domain_bounds: Tuple[float, float] = (-1, 1)
    num_projections: int = 5
    plane_coords_scale: float = 0.02
    device: Literal['cpu', 'cuda'] = 'cuda'
    seed: int = 123456

    shape_model: NeuralSDFConfig = NeuralSDFConfig()
    rotation_model: NeuralRTFConfig = NeuralRTFConfig()
    loss: DeformationLossConfig = DeformationLossConfig()
    optimizer: AdamConfig = AdamConfig()
    scheduler: MultiStepSchedulerConfig = MultiStepSchedulerConfig()