from __future__ import annotations

import torch
import mcubes
import os
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import torch.nn.functional as F

from typing import Tuple, Type, Literal
from dataclasses import dataclass, field
from pathlib import Path
from tqdm import tqdm

from iarap.config.base_config import InstantiateConfig
from iarap.model.neural_rtf import NeuralRTFConfig
from iarap.model.neural_sdf import NeuralSDFConfig
from iarap.utils import gradient, euler_to_rotation
from iarap.utils.misc import detach_model



class Animator:

    def __init__(self, config: AnimatorConfig):
        self.config = config
        self.out_dir = self.config.load_deformation.parent / 'video'
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.setup_model()

    def setup_model(self):
        self.shape_model = self.config.shape_model.setup().to(self.config.device)
        self.shape_model.load_state_dict(torch.load(self.config.load_shape))
        detach_model(self.shape_model)
        self.deformation_model = self.config.deformation_model.setup().to(self.config.device)
        self.deformation_model.load_state_dict(torch.load(self.config.load_deformation))
        detach_model(self.deformation_model)
        vol = self.make_volume()
        self.cached_sdf = self.evaluate_model(vol).numpy()

    def make_volume(self):
        steps = torch.linspace(self.config.min_coord, 
                               self.config.max_coord, 
                               self.config.resolution,
                               device=self.config.device)
        xx, yy, zz = torch.meshgrid(steps, steps, steps, indexing="ij")
        return torch.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T.float()

    @torch.no_grad()
    def evaluate_model(self, pts_volume):    
        f_eval = []
        for sample in tqdm(torch.split(pts_volume, self.config.chunk, dim=0)):
            f_eval.append(self.sdf_functional(sample.contiguous()).cpu())
        f_volume = torch.cat(f_eval, dim=0).reshape(*([self.config.resolution] * 3))
        return f_volume
    
    def extract_mesh(self, level=0.0):
        try:
            verts, faces = mcubes.marching_cubes(self.cached_sdf, level)
            verts /= self.config.resolution // 2
            verts -= 1.0
        except:
            verts = np.empty([0, 3], dtype=np.float32)
            faces = np.empty([0, 3], dtype=np.int32)
        return verts, faces
    
    def deform_mesh(self, verts, alpha=1.0):
        verts = torch.from_numpy(verts).to(self.config.device, torch.float)
        out_verts = []
        for sample in torch.split(verts, self.config.chunk, dim=0):
            outputs = self.deformation_model(sample, return_euler=True)  # transform(sample)
            rot = euler_to_rotation(outputs['euler'] * alpha)
            transl = outputs['transl'] * alpha
            transformed = (rot @ sample[..., None]).squeeze(-1) + transl
            out_verts.append(transformed.cpu().detach().numpy())
        verts = np.concatenate(out_verts, axis=0)
        return verts

    def sdf_functional(self, query):
        model_out = self.shape_model(query)
        return model_out['dist']
    
    def project_nearest(self, query, n_its=5, level=0.0):
        query = torch.from_numpy(query).float().to(self.config.device).view(-1, 3).requires_grad_()
        for i in range(n_its):
            dist = self.sdf_functional(query) - level
            grad = F.normalize(gradient(dist, query), dim=-1)
            query = (query - dist * grad).detach().requires_grad_()
        return query.detach()

    def run(self):

        ps.set_ground_plane_mode("none")
        ps.set_window_size(*self.config.window_size)
        ps.set_window_resizable(True)

        ps.init()

        verts, faces = self.extract_mesh()
        ps.look_at_dir(self.config.camera_origin, self.config.camera_target, self.config.camera_up)
        
        steps = torch.linspace(self.config.alpha_min, self.config.alpha_max, self.config.num_frames, device=self.config.device)
        for i in tqdm(range(steps.shape[0])):
            t = steps[i]
            verts_t = self.deform_mesh(verts, alpha=t)
            ps.register_surface_mesh("NeuralSDF", verts_t, faces, enabled=True)
            ps.screenshot(str(self.out_dir / f'frame_{str(i).zfill(5)}.png'))

        if self.config.ffmpeg_path is not None:
            os.chdir(self.out_dir)
            os.system(f'{self.config.ffmpeg_path} -framerate 30 \
                                                  -i "./frame_00%03d.png" \
                                                  -c:v libx264 -pix_fmt yuv420p \
                                                  out.mp4')



@dataclass
class AnimatorConfig(InstantiateConfig):

    _target: Type = field(default_factory=lambda: Animator)

    load_shape: Path = Path('./assets/weights/armadillo.pt')
    load_deformation: Path = None
    shape_model: NeuralSDFConfig = NeuralSDFConfig()
    deformation_model: NeuralRTFConfig = NeuralRTFConfig()

    alpha_min: float = 0.0
    alpha_max: float = 1.0
    num_frames: int = 100

    min_coord: float = -1.0
    max_coord: float =  1.0
    resolution: int = 512
    chunk: int = 65536

    camera_origin: Tuple[float, float, float] = (-1.0, 0.0, 3.5)
    camera_target: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    camera_up:     Tuple[float, float, float] = (0.0, 1.0, 0.0)
    window_size: Tuple[int, int] = (1600, 1200)

    ffmpeg_path: str = None
    device: Literal['cpu', 'cuda'] = 'cuda'
