import torch
import math
import numpy as np

from typing import Literal, Callable, Tuple
from jaxtyping import Float, Int
from torch import Tensor
from scipy.spatial import Delaunay


PHI = (1 + math.sqrt(5)) / 2


def delaunay(pts: torch.Tensor = None, 
             pts_np: np.ndarray = None,
             out_device: Literal['cpu', 'cuda'] = 'cpu'):
    if pts_np is None:
        pts_np = pts.cpu().detach().numpy().reshape(-1, 2)
    out_device = pts.device if pts is not None else out_device
    triv = Delaunay(pts_np).simplices
    return torch.from_numpy(triv).to(device=out_device, dtype=torch.long)

def sphere_random_uniform(n: int,
                          radius: float,
                          device: Literal['cpu', 'cuda'] = 'cpu') -> Float[Tensor, "n 2"]:
    # rho = np.sqrt(np.random.uniform(0, radius ** 2, size=(n - 1, 1))) # )
    rho = (torch.rand((n - 1, 1), device=device) * radius ** 2).sqrt()
    # theta = np.random.uniform(0, 2 * np.pi, size=(n - 1, 1))
    theta = torch.rand((n - 1, 1), device=device) * 2 * torch.pi
    vert = torch.cat([rho * theta.cos(), rho * theta.sin()], dim=-1)
    return torch.cat([torch.zeros((1, 2), device=device), vert], dim=0)

def sphere_sunflower(n: int,
                     radius: float,
                     device: Literal['cpu', 'cuda'] = 'cpu') -> Float[Tensor, "n 2"]:
    angle_stride = 2 * torch.pi / PHI ** 2
    k = torch.arange(1, n + 1, device=device, dtype=torch.float).unsqueeze(-1)
    r = ((k - 0.5).sqrt() / math.sqrt(n - 1 / 2)) * radius
    theta = k * angle_stride
    return torch.cat([r * theta.cos(), r * theta.sin()], dim=-1)

def sphere_gaussian_radius(n: int,
                           radius: float,
                           device: Literal['cpu', 'cuda'] = 'cpu') -> Float[Tensor, "n 2"]:
    rho = torch.randn((n - 1, 1), device=device).abs()
    rho = rho / (rho.max() / radius)
    theta = torch.rand((n - 1, 1), device=device) * 2 * torch.pi
    vert = torch.cat([rho * theta.cos(), rho * theta.sin()], dim=-1)
    return torch.cat([torch.zeros((1, 2), device=device), vert], dim=0)

def gaussian_max_norm(n: int,
                      radius: float,
                      device: Literal['cpu', 'cuda'] = 'cpu') -> Float[Tensor, "n 2"]:
    pts = torch.randn(n - 1, 2, device=device)
    vert = pts / (pts.norm(dim=-1).max() / radius)
    return torch.cat([torch.zeros((1, 2), device=device), vert], dim=0)


def get_patch_mesh(point_generator: Callable,
                   point_triangulator: Callable,
                   num_points: int,
                   radius: float,
                   device: Literal['cpu', 'cuda'] = 'cpu'
                   ) -> Tuple[Float[Tensor, "num_points 3"], Int[Tensor, "m 3"]]:
    vert = point_generator(num_points, radius, device)
    triv = point_triangulator(vert, out_device=device)
    vert = torch.cat([vert, torch.zeros(num_points, 1, device=device)], dim=-1)
    return vert, triv