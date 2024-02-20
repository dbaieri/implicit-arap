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
    file: Path = Path('./assets/mesh/armadillo.ply')
    surf_sample: int = 16384
    space_sample: int = 16384
    uniform_ratio: float = 1/8
    local_noise_scale: float = 0.05
    device: Literal['cpu', 'cuda'] = 'cuda'



class MeshData:

    def __init__(self, config: MeshDataConfig):
        super(MeshData, self).__init__()
        mesh: trimesh.Trimesh = trimesh.load(config.file, force='mesh')
        centroid = torch.tensor(mesh.centroid, device=config.device)
        verts = torch.tensor(mesh.vertices)
        faces = torch.tensor(mesh.faces)
        self.geometry = kal.rep.SurfaceMesh(verts, faces).to_batched().to(config.device)
        self.normalize(centroid, 0.8)
        self.surf_sample = config.surf_sample
        self.space_sample = config.space_sample
        assert self.space_sample <= self.surf_sample
        self.uniform_ratio = config.uniform_ratio
        self.noise_scale = config.local_noise_scale
        self.domain = torch.tensor([[-1, -1, -1], [1, 1, 1]]).to(config.device)
        self.device = config.device
        self.dataset = MeshDataset(self)

    def normalize(self, shift: Float[Tensor, "3"], domain_ratio: float):
        self.geometry.vertices -= shift.unsqueeze(0)
        self.geometry.vertices /= self.geometry.vertices.abs().max()
        self.geometry.vertices *= domain_ratio


class MeshDataset(Dataset):

    def __init__(self, mesh: MeshData):
        super(MeshDataset, self).__init__()
        self.mesh = mesh

    def __len__(self):
        return 1
    
    def __getitem__(self, index) -> Dict[str, Float[Tensor, "*batch d"]]:
        surf_sample, face_idx = kal.ops.mesh.sample_points(self.mesh.geometry.vertices, 
                                                           self.mesh.geometry.faces,
                                                           self.mesh.surf_sample)
        surf_sample, face_idx = surf_sample.squeeze().float(), face_idx.squeeze()
        normal_sample = self.mesh.geometry.face_normals.mean(dim=-2)[0, face_idx].float()
        
        n_uniform = int(self.mesh.space_sample * self.mesh.uniform_ratio)
        unif_sample = torch.rand(n_uniform, 3, device=self.mesh.device) 
        shift = self.mesh.domain[0:1, :]
        scale = self.mesh.domain[1:, :] - shift
        unif_sample = (unif_sample * scale) + shift

        n_gaussian = int(self.mesh.space_sample * (1 - self.mesh.uniform_ratio))
        gauss_sample = torch.randn(n_gaussian, 3, device=self.mesh.device) 
        gauss_sample = surf_sample[:n_gaussian, :] + gauss_sample * self.mesh.noise_scale

        return {
            'surface_sample': surf_sample,
            'normal_sample': normal_sample,
            'space_sample': torch.cat([gauss_sample, unif_sample], dim=0)
        }
        