import torch
import torch.nn as nn
import numpy as np

from typing import Dict, Any, List
from dataclasses import field


def to_immutable_dict(d: Dict[str, Any]):
    """Method to convert mutable dict to default factory dict

    Args:
        d: dictionary to convert into default factory dict for dataclass
    """
    return field(default_factory=lambda: dict(d))

def to_immutable_list(l: List[Any]):
    """Method to convert mutable dict to default factory dict

    Args:
        d: dictionary to convert into default factory dict for dataclass
    """
    return field(default_factory=lambda: list(l))

def detach_model(m: nn.Module):
    for p in m.parameters():
        p.requires_grad = False

def gather_nd_torch(params, indices, batch_dim=1):
    batch_dims = params.size()[:batch_dim]  # [b1, ..., bn]
    batch_size = np.cumprod(list(batch_dims))[-1]  # b1 * ... * bn
    c_dim = params.size()[-1]  # c
    grid_dims = params.size()[batch_dim:-1]  # [g1, ..., gm]
    n_indices = indices.size(-2)  # x
    n_pos = indices.size(-1)  # m

    # reshape leadning batch dims to a single batch dim
    params = params.reshape(batch_size, *grid_dims, c_dim)
    indices = indices.reshape(batch_size, n_indices, n_pos)

    # build gather indices
    # gather for each of the data point in this "batch"
    batch_enumeration = torch.arange(batch_size).unsqueeze(1)
    gather_dims = [indices[:, :, i] for i in range(len(grid_dims))]
    gather_dims.insert(0, batch_enumeration)
    gathered = params[gather_dims]

    # reshape back to the shape with leading batch dims
    gathered = gathered.reshape(*batch_dims, n_indices, c_dim)
    return gathered


def trilinear_interp(grid_3d, sampling_points, resolution):

    grid_3d_shape = grid_3d.size()
    sampling_points_shape = sampling_points.size()
    voxel_cube_shape = grid_3d_shape[-4:-1]  # [H, W, D]
    batch_dims = sampling_points_shape[:-2]  # [A1, ..., An]
    num_points = sampling_points_shape[-2]  # M

    sampling_points = (sampling_points + 1.0) / 2.0  # Now [0; 1]
    dx = 1.0 / resolution
    sampling_points = sampling_points / dx
    bottom_left = torch.floor(sampling_points)
    top_right = bottom_left + 1
    bottom_left_index = bottom_left.type(torch.int32)
    top_right_index = top_right.type(torch.int32)
    
    x0_index, y0_index, z0_index = torch.unbind(bottom_left_index, dim=-1)
    x1_index, y1_index, z1_index = torch.unbind(top_right_index, dim=-1)
    index_x = torch.concat([x0_index, x1_index, x0_index, x1_index,
                            x0_index, x1_index, x0_index, x1_index], dim=-1)
    index_y = torch.concat([y0_index, y0_index, y1_index, y1_index,
                            y0_index, y0_index, y1_index, y1_index], dim=-1)
    index_z = torch.concat([z0_index, z0_index, z0_index, z0_index,
                            z1_index, z1_index, z1_index, z1_index], dim=-1)
    indices = torch.stack([index_x, index_y, index_z], dim=-1)


    # clip_value_max = torch.from_numpy(np.ndarray(list(voxel_cube_shape)) - 1)
    # clip_value_min = torch.zeros_like(clip_value_max)
    indices = torch.clamp(indices, min=0, max=resolution-1)

    content = gather_nd_torch(
        params=grid_3d, indices=indices.long(), batch_dim=len(batch_dims))

    distance_to_bottom_left = sampling_points - bottom_left
    distance_to_top_right = top_right - sampling_points
    x_x0, y_y0, z_z0 = torch.unbind(distance_to_bottom_left, dim=-1)
    x1_x, y1_y, z1_z = torch.unbind(distance_to_top_right, dim=-1)
    weights_x = torch.concat([x1_x, x_x0, x1_x, x_x0,
                              x1_x, x_x0, x1_x, x_x0], dim=-1)
    weights_y = torch.concat([y1_y, y1_y, y_y0, y_y0,
                              y1_y, y1_y, y_y0, y_y0], dim=-1)
    weights_z = torch.concat([z1_z, z1_z, z1_z, z1_z,
                              z_z0, z_z0, z_z0, z_z0], dim=-1)

    weights = weights_x * weights_y * weights_z
    weights = weights.unsqueeze(-1)

    interpolated_values = weights * content

    shape = interpolated_values.shape
    return interpolated_values.reshape(*shape[:-2], 8, num_points, shape[-1]).sum(dim=-3)
    # return sum(torch.split(interpolated_values, [num_points] * 8, dim=-2))