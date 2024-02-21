import torch
import numpy as np

from typing import Literal
from scipy.spatial import Delaunay


def delaunay(pts: torch.Tensor = None, 
             pts_np: np.ndarray = None,
             out_device: Literal['cpu', 'cuda'] = 'cpu'):
    if pts_np is None:
        pts_np = pts.cpu().detach().numpy().reshape(-1, 2)
    out_device = pts.device if pts is not None else out_device
    triv = Delaunay(pts_np).simplices
    return torch.from_numpy(triv).to(device=out_device, dtype=torch.long)