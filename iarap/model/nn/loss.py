
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from jaxtyping import Float
from typing import Type
from dataclasses import dataclass, field

from iarap.config.base_config import InstantiateConfig


@dataclass
class IGRConfig(InstantiateConfig):

    _target: Type = field(default_factory=lambda: ImplicitGeometricRegularization)
    zero_sdf_surface_w: float = 1.0
    eikonal_error_w: float = 0.01
    normals_error_w: float = 0.01
    zero_penalty_w: float = 0.05


class ImplicitGeometricRegularization(nn.Module):

    def __init__(self, config: IGRConfig):
        super(ImplicitGeometricRegularization, self).__init__()
        self.config = config
        self.zero_loss = ZeroLoss()
        self.zero_penalty = ZeroPenalty()
        self.eikonal_loss = EikonalLoss()
        self.normals_loss = NormalsLoss()

    def forward(self, 
                pred_sdf_surf: Float[Tensor, "*batch 1"],
                pred_sdf_space: Float[Tensor, "*batch 1"],
                grad_sdf_surf: Float[Tensor, "*batch 3"],
                grad_sdf_space: Float[Tensor, "*batch 3"],
                surf_normals: Float[Tensor, "*batch 3"]):
        zero_loss = self.config.zero_sdf_surface_w * self.zero_loss(pred_sdf_surf)
        zero_penalty = self.config.zero_penalty_w * self.zero_penalty(pred_sdf_space)
        eik_loss = self.config.eikonal_error_w * self.eikonal_loss(torch.cat([grad_sdf_space, grad_sdf_surf], dim=1))
        normals_loss = self.config.normals_error_w * self.normals_loss(grad_sdf_surf, surf_normals)
        return {'zero_loss': zero_loss, 
                'eikonal_loss': eik_loss, 
                'zero_penalty': zero_penalty,
                'normals_loss': normals_loss}

class EikonalLoss(nn.Module):

    def __init__(self):
        super(EikonalLoss, self).__init__()

    def forward(self, grad: Float[Tensor, "*batch 3"]) -> Float[Tensor, "1"]:
        return (grad.norm(dim=-1) - 1.0).abs().mean()
    

class NormalsLoss(nn.Module):

    def __init__(self):
        super(NormalsLoss, self).__init__()

    def forward(self, 
                grad: Float[Tensor, "*batch 3"],
                normals: Float[Tensor, "*batch 3"]) -> Float[Tensor, "1"]:
        return (1 - F.cosine_similarity(grad, normals, dim=-1)).mean()
    
    
class ZeroLoss(nn.Module):

    def __init__(self):
        super(ZeroLoss, self).__init__()

    def forward(self, dist: Float[Tensor, "*batch 1"]) -> Float[Tensor, "1"]:
        return dist.abs().mean()
    

class ZeroPenalty(nn.Module):

    def __init__(self, scale: float=100):
        super(ZeroPenalty, self).__init__()
        self.scale = scale

    def forward(self, dist: Float[Tensor, "*batch 1"]) -> Float[Tensor, "1"]:
        error = torch.exp(-self.scale * dist.abs())
        return error.mean()