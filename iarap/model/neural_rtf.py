from re import M
import torch

import torch.nn as nn
import numpy as np

from torch import Tensor
from jaxtyping import Float
from typing import Dict, Type, Tuple, Any
from dataclasses import dataclass, field

from iarap.config.base_config import InstantiateConfig
from iarap.model.base_sdf import SDF
from iarap.model.nn import MLP, FourierFeatsEncoding, InvertibleRtMLP, InvertibleMLP3D
from iarap.model.nn.mlp import MLPConfig
from iarap.utils import to_immutable_dict, euler_to_rotation

    
def fixed_point_Rt_invert(g, y, iters=15, verbose=False):
    with torch.no_grad():
        x = y
        dim = x.size(-1)
        for i in range(iters):
            block_out = g(x)
            rot, transl = block_out[..., :3], block_out[..., 3:]
            rot = euler_to_rotation(rot)
            x = (rot.transpose(-1, -2) @ (y - transl)[..., None]).squeeze(-1)
            if verbose:
                block_out = g(x)
                rot, transl = block_out[..., :3], block_out[..., 3:]
                rot = euler_to_rotation(rot)
                test = (rot @ x[..., None]).squeeze(-1) + transl
                err = (y - test).view(-1, dim).norm(dim=-1).mean()
                err = err.detach().cpu().item()
                print("iter:%d err:%s" % (i, err))
    return x



@dataclass
class NeuralRTFConfig(InstantiateConfig):

    _target: Type = field(default_factory=lambda: NeuralRTF)
    network: InstantiateConfig = MLPConfig(
        in_dim=3,
        num_layers=8,
        layer_width=256,
        out_dim=6,
        skip_connections=(4,),
        activation='Softplus',
        act_defaults={'beta': 100},
        num_frequencies=6,
        encoding_with_input=True,
        geometric_init=True
    )


class NeuralRTF(SDF):

    def __init__(self, config: NeuralRTFConfig):
        super(NeuralRTF, self).__init__(config.network.in_dim)
        self.config = config
        self.network = self.config.network.setup()
        self.sdf_callable = lambda x: x.norm(dim=-1, keepdim=True) - 0.5

    def set_sdf_callable(self, dist_fn):
        self.sdf_callable = dist_fn

    def distance(self, x_in: Float[Tensor, "*batch in_dim"]) -> Float[Tensor, "*batch 1"]:
        x = self.inverse(x_in)
        return self.sdf_callable(x)

    def forward(self, 
                x_in: Float[Tensor, "*batch in_dim"],
                return_euler: bool = False
                ) -> Dict[str, Float[Tensor, "*batch f"]]:
        outputs = {}
        rt = self.network(x_in)
        if isinstance(rt, torch.Tensor):
            euler, transl = rt[..., :3], rt[..., 3:]
            rot = euler_to_rotation(euler)
            if return_euler:
                outputs['euler'] = euler
        else:
            _, rot, transl = rt
        outputs['rot'] = rot
        outputs['transl'] = transl  
        return outputs
    
    def deform(self, 
               x_in: Float[Tensor, "*batch in_dim"],
               ) -> Float[Tensor, "*batch in_dim"]:
        outputs = self(x_in)
        return (outputs['rot'] @ x_in[..., None]).squeeze(-1) + outputs['transl']
        # return rotated
        # return self.network.inverse(x_in)
        # return fixed_point_invert(self.model, x_in)
    
    def inverse(self,
                x_in: Float[Tensor, "*batch in_dim"],
                ) -> Float[Tensor, "*batch in_dim"]:
        return fixed_point_Rt_invert(self.network, x_in)