from __future__ import annotations
import os

import vedo
import random
import mcubes
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
from iarap.utils.meshing import get_patch_mesh, sphere_random_uniform, sphere_sunflower, gaussian_max_norm, sphere_gaussian_radius



class DeformTrainer(Trainer):

    def __init__(self, config: DeformTrainerConfig):
        super(DeformTrainer, self).__init__(config)
        self.device = self.config.device
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        self.handles = torch.cat([self.handles_moving[:, :3], self.handles_static], dim=0)
        for it in range(self.config.num_projections):
            self.handles = self.source.project_nearest(self.handles).detach()

    def setup_data(self):
        # Avoids rewriting training function, does a single iteration per epoch
        self.loader = [0]  
        # Configure handles
        handle_cfg = yaml.load(open(self.config.handles_spec, 'r'), yaml.Loader)
        handle_dir = self.config.handles_spec.parent
        assert 'handles' in handle_cfg.keys(), f"Handle specification not found in file {self.config.handles_spec}"
        if 'static' in handle_cfg['handles'].keys() and len(handle_cfg['handles']['static']['positions']) > 0:
            static = [np.loadtxt(handle_dir / "parts" / f"{f}.txt").reshape(-1, 3) for f in handle_cfg['handles']['static']['positions']]
            static = np.concatenate(static, axis=0)
            if len(static.shape) < 2:
                static = np.expand_dims(static, 0)
            self.handles_static = torch.from_numpy(static).to(self.config.device, torch.float)
        else:
            self.handles_static = torch.empty((0, 3), device=self.config.device, dtype=torch.float)
        if 'moving' in handle_cfg['handles'].keys() and len(handle_cfg['handles']['moving']['positions']) > 0:
            assert len(handle_cfg['handles']['moving']['positions']) == len(handle_cfg['handles']['moving']['transform']),\
                "It is required to specify one transform for each handle set"
            moving = [np.loadtxt(handle_dir / "parts" / f"{f}.txt").reshape(-1, 3) for f in handle_cfg['handles']['moving']['positions']]
            transforms = [np.loadtxt(handle_dir / "transforms" / f"{f}.txt").reshape(-1, 3) for f in handle_cfg['handles']['moving']['transform']]
            moving = np.concatenate(moving, axis=0)
            transforms = np.concatenate(transforms, axis=0)
            if len(moving.shape) < 2:
                moving = np.expand_dims(moving, axis=0)
                transforms = np.expand_dims(transforms, axis=0)
            moving_pos = torch.from_numpy(moving).to(self.config.device, torch.float)
            moving_trans = torch.from_numpy(transforms).to(self.config.device, torch.float)
            self.handles_moving = torch.cat([moving_pos, moving_trans], dim=-1)
        else:
            self.handles_moving = torch.empty((0, 6), device=self.config.device, dtype=torch.float)

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

    def local_patch_meshing(self):
        surf_sample = self.source.sample_zero_level_set(self.config.zero_samples - self.handles.shape[0],
                                                        self.config.near_surface_threshold,
                                                        self.config.attempts_per_step,
                                                        self.config.domain_bounds,
                                                        self.config.num_projections).detach()
        surf_sample = torch.cat([self.handles, surf_sample], dim=0)
        space_sample = self.sample_domain(self.config.space_samples)
        samples = torch.cat([surf_sample, space_sample], dim=0)
        
        sdf_outs = self.source(samples, with_grad=True)
        sample_dist, patch_normals = sdf_outs['dist'], sdf_outs['grad']
        tangent_planes = self.source.tangent_plane(samples, patch_normals)

        plane_coords, triangles = get_patch_mesh(sphere_random_uniform, 
                                                 delaunay,
                                                 self.config.delaunay_sample,
                                                 self.config.plane_coords_scale,
                                                 self.device)

        tangent_coords = (tangent_planes.unsqueeze(1) @ plane_coords.view(1, -1, 3, 1)).squeeze() 
        tangent_pts = tangent_coords + samples.unsqueeze(1) 
        level_set_verts = tangent_pts
        for it in range(self.config.num_projections):
            level_set_verts = self.source.project_level_sets(level_set_verts, sample_dist)  # n m 3
        
        return level_set_verts, triangles

    def train_step(self, batch):

        handle_moving_gt = self.handles_moving[:, 3:]
        handle_static_gt = self.handles[self.handles_moving.shape[0]:, :].detach()

        patch_verts, triangles = self.local_patch_meshing()
                    
        rtf_out = self.model(patch_verts.detach())
        rotations = rtf_out['rot']
        translations = rtf_out['transl']
        # test_invert = self.model.inverse((rotations @ level_set_verts[..., None]).squeeze(-1) + translations)

        handle_idx = torch.stack([
            torch.arange(0, self.handles.shape[0], device=self.device, dtype=torch.long),
            torch.zeros(self.handles.shape[0], device=self.device, dtype=torch.long)
        ], dim=-1)
        moving_idx = handle_idx[:self.handles_moving.shape[0], :]
        static_idx = handle_idx[self.handles_moving.shape[0]:, :]
        loss_dict = self.loss(patch_verts.detach(), 
                              triangles, rotations, translations,
                              moving_idx, static_idx, handle_moving_gt, handle_static_gt)
        
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


    
class MCDeformTrainer(Trainer):

    def __init__(self, config: MCDeformTrainerConfig):
        super(MCDeformTrainer, self).__init__(config)
        self.device = self.config.device
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        self.handles = torch.cat([self.handles_moving[:, :3], self.handles_static], dim=0)
        for it in range(self.config.num_projections):
            self.handles = self.source.project_nearest(self.handles).detach()

    def setup_data(self):
        # Avoids rewriting training function, does a single iteration per epoch
        self.loader = [0]  
        # Configure handles
        handle_cfg = yaml.load(open(self.config.handles_spec, 'r'), yaml.Loader)
        handle_dir = self.config.handles_spec.parent
        assert 'handles' in handle_cfg.keys(), f"Handle specification not found in file {self.config.handles_spec}"
        if 'static' in handle_cfg['handles'].keys() and len(handle_cfg['handles']['static']['positions']) > 0:
            static = [np.loadtxt(handle_dir / "parts" / f"{f}.txt") for f in handle_cfg['handles']['static']['positions']]
            static = np.concatenate(static, axis=0)
            if len(static.shape) < 2:
                static = np.expand_dims(static, 0)
            self.handles_static = torch.from_numpy(static).to(self.config.device, torch.float)
        else:
            self.handles_static = torch.empty((0, 3), device=self.config.device, dtype=torch.float)
        if 'moving' in handle_cfg['handles'].keys() and len(handle_cfg['handles']['moving']['positions']) > 0:
            assert len(handle_cfg['handles']['moving']['positions']) == len(handle_cfg['handles']['moving']['transform']),\
                "It is required to specify one transform for each handle set"
            moving = [np.loadtxt(handle_dir / "parts" / f"{f}.txt") for f in handle_cfg['handles']['moving']['positions']]
            transforms = [np.loadtxt(handle_dir / "transforms" / f"{f}.txt") for f in  handle_cfg['handles']['moving']['transform']]
            moving = np.concatenate(moving, axis=0)
            transforms = np.concatenate(transforms, axis=0)
            if len(moving.shape) < 2:
                moving = np.expand_dims(moving, axis=0)
                transforms = np.expand_dims(transforms, axis=0)
            moving_pos = torch.from_numpy(moving).to(self.config.device, torch.float)
            moving_trans = torch.from_numpy(transforms).to(self.config.device, torch.float)
            self.handles_moving = torch.cat([moving_pos, moving_trans], dim=-1)
        else:
            self.handles_moving = torch.empty((0, 6), device=self.config.device, dtype=torch.float)

    def setup_model(self):
        self.source: NeuralSDF = self.config.shape_model.setup().to(self.config.device).eval()
        self.source.load_state_dict(torch.load(self.config.pretrained_shape))
        detach_model(self.source)
        with torch.no_grad():
            steps = torch.linspace(self.config.domain_bounds[0], 
                                self.config.domain_bounds[1], 
                                self.config.mc_resolution,
                                device=self.config.device)
            xx, yy, zz = torch.meshgrid(steps, steps, steps, indexing="ij")
            volume_pts = torch.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T.float()   
            f_eval = []
            for sample in tqdm(torch.split(volume_pts, self.config.chunk, dim=0)):
                f_eval.append(self.source(sample.contiguous())['dist'].cpu())
            self.volume_sdf = torch.cat(f_eval, dim=0).reshape(*([self.config.mc_resolution] * 3)).numpy()

        self.model: NeuralRTF = self.config.rotation_model.setup().to(self.config.device).train()
        self.model.set_sdf_callable(self.source.distance)
        self.loss = self.config.loss.setup()

    def sample_domain(self, nsamples):
        scale = self.config.domain_bounds[1] - self.config.domain_bounds[0]
        return torch.rand(nsamples, 3, device=self.device) * scale + self.config.domain_bounds[0]

    def train_step(self, batch):

        handle_moving_gt = self.handles_moving[:, 3:]
        handle_static_gt = self.handles[self.handles_moving.shape[0]:, :].detach()

        level = random.random() * \
            (self.config.mc_level_bounds[1] - self.config.mc_level_bounds[0]) + \
            self.config.mc_level_bounds[0]

        try:
            verts, faces = mcubes.marching_cubes(self.volume_sdf, level)
            verts /= self.config.mc_resolution // 2
            verts -= 1.0
            faces = faces.astype(np.int32)
        except:
            verts = np.empty([0, 3], dtype=np.float32)
            faces = np.empty([0, 3], dtype=np.int32)
        verts = torch.from_numpy(verts).float().to(self.config.device)
        faces = torch.from_numpy(faces).long().to(self.config.device)
        
        query_pts = torch.cat([verts, self.handles], dim=0).unsqueeze(0)

        rtf_out = self.model(query_pts.detach())
        rotations = rtf_out['rot']
        translations = rtf_out['transl']

        n = query_pts.shape[1]
        h = self.handles.shape[0]
        moving_idx = torch.arange(n - h, n - h + self.handles_moving.shape[0], 
                                  device=self.config.device)
        moving_idx = torch.stack([torch.zeros_like(moving_idx), moving_idx], dim=-1)
        static_idx = torch.arange(n - h + self.handles_moving.shape[0], n, 
                                  device=self.config.device)
        static_idx = torch.stack([torch.zeros_like(static_idx), static_idx], dim=-1)

        loss_dict = self.loss(query_pts.detach(), 
                              faces, rotations, translations,
                              moving_idx, static_idx, handle_moving_gt, handle_static_gt)
        
        return loss_dict
    
    def postprocess(self):
        ckpt_dir = self.logger.dir + '/checkpoints/'
        print("Saving model weights in: {}".format(ckpt_dir))
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(self.model.state_dict(), ckpt_dir + '/neural_rotation.pt')

        
@dataclass
class MCDeformTrainerConfig(InstantiateConfig):

    _target: Type = field(default_factory=lambda: MCDeformTrainer)

    num_steps: int = 1
    pretrained_shape: Path = None
    handles_spec: Path = None
    mc_resolution: int = 64
    mc_level_bounds: Tuple[float, float] = (-0.1, 0.2)
    domain_bounds: Tuple[float, float] = (-1, 1)
    num_projections: int = 5
    device: Literal['cpu', 'cuda'] = 'cuda'
    chunk: int = 65536
    seed: int = 123456

    shape_model: NeuralSDFConfig = NeuralSDFConfig()
    rotation_model: NeuralRTFConfig = NeuralRTFConfig()
    loss: DeformationLossConfig = DeformationLossConfig()
    optimizer: AdamConfig = AdamConfig()
    scheduler: MultiStepSchedulerConfig = MultiStepSchedulerConfig()