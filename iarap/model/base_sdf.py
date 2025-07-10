import torch

import torch.autograd as ad
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from jaxtyping import Float
from typing import Optional, Tuple

from iarap.utils import cross_skew_matrix



class SDF(nn.Module):

    """
    Abstract base classes for torch-based signed distance functions.
    """

    def __init__(self, dim: int):
        super(SDF, self).__init__()
        self.dim = dim

    def distance(self, x_in: Float[Tensor, "*batch in_dim"]) -> Float[Tensor, "*batch 1"]:
        raise NotImplementedError()
    
    def gradient(self, 
                 x_in: Float[Tensor, "*batch in_dim"],
                 dist: Optional[Float[Tensor, "*batch 1"]] = None,
                 differentiable: bool = False) -> Float[Tensor, "*batch 3"]:
        if dist is None:
            x = x_in.requires_grad_()
            dist = self.distance(x)

        d_outputs = torch.ones_like(dist)
        return ad.grad(dist, x_in, d_outputs, 
                       create_graph=differentiable, 
                       retain_graph=differentiable, 
                       only_inputs=True)[0]
    
    def project_nearest(self, 
                        x_in: Float[Tensor, "*batch in_dim"],
                        dist: Optional[Float[Tensor, "*batch 1"]] = None,
                        grad: Optional[Float[Tensor, "*batch 3"]] = None,
                        differentiable: bool = False) -> Float[Tensor, "*batch 3"]:
        x = x_in.requires_grad_() if grad is None else x_in
        if dist is None:
            dist = self.distance(x)
        if grad is None:
            grad = self.gradient(x, dist, differentiable)
        return x_in - dist * F.normalize(grad, dim=-1)
    
    def project_level_sets(self, 
                           x_in: Float[Tensor, "*batch sample 3"],
                           origin_dist: Optional[Float[Tensor, "*batch 1"]]
                           ) -> Float[Tensor, "*batch 3"]:
        x = x_in.requires_grad_()
        dist = self.distance(x)
        grad = self.gradient(x, dist)
        return x_in - (dist - origin_dist.unsqueeze(1)) * F.normalize(grad, dim=-1)
    
    def sphere_trace(self, 
                     x_in: Float[Tensor, "*batch 3"],
                     dir: Float[Tensor, "*batch 3"]) -> Float[Tensor, "*batch 3"]:
        dist = self.distance(x_in)
        return x_in - dist * dir
    
    def tangent_plane(self, 
                      x_in: Float[Tensor, "*batch in_dim"],
                      grad: Optional[Float[Tensor, "*batch 3"]] = None,
                      differentiable: bool = False) -> Float[Tensor, "*batch 3 3"]:
        if grad is None:
            x = x_in.requires_grad_() if grad is None else x_in
            dist = self.distance(x)
            grad = self.gradient(x, dist, differentiable)
        d = x_in.shape[-1]
        normal = F.normalize(grad, dim=-1)
        I = torch.eye(d, device=x_in.device).view(*([1] * (x_in.dim() - 1)), d, d).expand(*x_in.shape[:-1], -1, -1)
        z = I[..., 2]
        v = torch.linalg.cross(normal, z, dim=-1)
        c = (z * normal).sum(dim=-1)
        cross_matrix = cross_skew_matrix(v)
        scale = ((1 - c) / v.norm(dim=-1).pow(2)).view(-1, 1, 1)
        return I + cross_matrix + (cross_matrix @ cross_matrix) * scale
        # return F.normalize(I - (normal.unsqueeze(-2) * normal.unsqueeze(-1)), dim=-2)

    def sample_zero_level_set(self,
                              num_samples: int,
                              threshold: float = 0.05,
                              samples_per_step: int = 10000,
                              bounds: Tuple[float, float] = (-1, 1),
                              num_projections: int = 1):
        n_samples = 0
        sampled_pts = []
        device = next(self.parameters()).device
        with torch.no_grad():
            while n_samples < num_samples:
                unif = torch.rand(samples_per_step, 3, device=device) * (bounds[1] - bounds[0]) + bounds[0]
                sdf = self.distance(unif)
                close = unif[sdf.squeeze().abs() < threshold, :]
                sampled_pts.append(close)
                n_samples += close.shape[0]
        sampled_pts = torch.cat(sampled_pts, dim=0)[:num_samples, :]
        for it in range(num_projections):
            sampled_pts = self.project_nearest(sampled_pts)
        return sampled_pts
    
