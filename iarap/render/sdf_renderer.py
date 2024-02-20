from __future__ import annotations

import torch
import vedo

from typing import Type, Literal
from dataclasses import dataclass, field
from pathlib import Path
from tqdm import tqdm

from iarap.config.base_config import InstantiateConfig
from iarap.model.sdf import NeuralSDFConfig



class SDFRenderer:

    def __init__(self, config: SDFRendererConfig):
        self.config = config
        self.setup_model()

    def setup_model(self):
        self.model = self.config.model.setup().to(self.config.device)
        self.model.load_state_dict(torch.load(self.config.load_checkpoint))

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
            model_out = self.model(sample.contiguous())
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
    load_checkpoint: Path = Path('./assets/weights/armadillo.pt')
    min_coord: float = -1.0
    max_coord: float =  1.0
    resolution: int = 512
    chunk: int = 65536
    device: Literal['cpu', 'cuda'] = 'cuda'
    model: NeuralSDFConfig = NeuralSDFConfig()
