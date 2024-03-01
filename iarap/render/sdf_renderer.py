from __future__ import annotations

import torch
import vedo

from typing import Type, Literal
from dataclasses import dataclass, field
from pathlib import Path
from tqdm import tqdm

from iarap.config.base_config import InstantiateConfig
from iarap.model.rot_net import NeuralRFConfig
from iarap.model.sdf import NeuralSDFConfig



class SDFRenderer:

    def __init__(self, config: SDFRendererConfig):
        self.config = config
        self.setup_model()

    def setup_model(self):
        self.shape_model = self.config.shape_model.setup().to(self.config.device)
        self.shape_model.load_state_dict(torch.load(self.config.load_shape))
        if self.config.load_deformation is not None:
            self.deformation_model = self.config.deformation_model.setup().to(self.config.device)
            self.deformation_model.load_state_dict(torch.load(self.config.load_deformation))
        else:
            self.deformation_model = None

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
            sample = sample.contiguous()
            if self.deformation_model is not None:
                rotations = self.deformation_model(sample)['rot']
                sample = (rotations @ sample[..., None]).squeeze(-1)
            model_out = self.shape_model(sample)
            f_eval.append(model_out['dist'].cpu())

        f_volume = torch.cat(f_eval, dim=0).reshape(*([self.config.resolution] * 3))
        return f_volume
    
    def render_volumetric_function(self, volume):
        volume = vedo.Volume(volume)
        plt = vedo.applications.IsosurfaceBrowser(volume, use_gpu=True, c='gold')
        plt.show(axes=7, bg2='lb').close()

    def run(self):
        volume = self.make_volume()
        f_volume = self.evaluate_model(volume)
        self.render_volumetric_function(f_volume.numpy())
        ## TODO export geometry with marching cubes

    
        


@dataclass
class SDFRendererConfig(InstantiateConfig):

    _target: Type = field(default_factory=lambda: SDFRenderer)
    load_shape: Path = Path('./assets/weights/armadillo.pt')
    load_deformation: Path = None
    min_coord: float = -1.0
    max_coord: float =  1.0
    resolution: int = 512
    chunk: int = 65536
    device: Literal['cpu', 'cuda'] = 'cuda'
    shape_model: NeuralSDFConfig = NeuralSDFConfig()
    deformation_model: NeuralRFConfig = NeuralRFConfig()
