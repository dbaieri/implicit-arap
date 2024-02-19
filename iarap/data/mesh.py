import torch
import trimesh
import kaolin as kal

from pathlib import Path
from typing import Literal, Dict, Type
from torch import Tensor
from jaxtyping import Float
from torch.utils.data import Dataset
from dataclasses import dataclass, field

from iarap.config.base_config import InstantiateConfig


@dataclass
class MeshDataConfig(InstantiateConfig):

    _target: Type = field(default_factory=lambda: MeshData)
    file: Path = Path('./assets/armadillo.ply')
    surf_sample: int = 8192
    space_sample: int = 8192
    noise_scale: float = 0.05
    device: Literal['cpu', 'cuda'] = 'cpu'



class MeshData(Dataset):

    def __init__(self, config: MeshDataConfig):
        super(MeshData, self).__init__()
        mesh: trimesh.Trimesh = trimesh.load(config.file, force='mesh')
        centroid = torch.tensor(mesh.centroid, device=config.device)
        verts = torch.tensor(mesh.vertices)
        faces = torch.tensor(mesh.faces)
        self.geometry = kal.rep.SurfaceMesh(verts, faces).to(config.device)
        self.normalize(centroid, 0.8)
        self.surf_sample = config.surf_sample
        self.space_sample = config.space_sample
        self.uniform_ratio = 1/8
        self.noise_scale = config.noise_scale
        self.domain = torch.tensor([[-1, -1, -1], [1, 1, 1]]).to(config.device)
        self.device = config.device

    def normalize(self, shift: Float[Tensor, "3"], domain_ratio: float):
        self.geometry.vertices -= shift.unsqueeze(0)
        self.geometry.vertices /= self.geometry.vertices.abs().max()
        self.geometry.vertices *= domain_ratio

    def len(self):
        return 1
    
    def __getitem__(self, index) -> Dict[str, Float[Tensor, "*batch d"]]:
        surf_sample, face_idx = kal.ops.mesh.sample_points(self.geometry.vertices, 
                                                           self.geometry.faces)
        normal_sample = self.geometry.face_normals.mean(dim=-2)[face_idx]
        
        unif_sample = torch.rand(int(self.space_sample * self.uniform_ratio), 3, device=self.device) 
        shift = self.domain[0:1, :]
        scale = self.domain[1:, :] - shift
        unif_sample = (unif_sample * scale) + shift

        gauss_sample = torch.randn(int(self.space_sample * (1 - self.uniform_ratio)), 3, device=self.device) 
        gauss_sample = surf_sample + gauss_sample * self.noise_scale

        return {
            'surface_sample': surf_sample,
            'normal_sample': normal_sample,
            'space_sample': torch.cat([gauss_sample, unif_sample], dim=0)
        }
        