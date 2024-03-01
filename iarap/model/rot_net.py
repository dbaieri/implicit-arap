import torch

import torch.autograd as ad
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch import Tensor
from jaxtyping import Float
from typing import Optional, Dict, Type, Tuple, Any
from dataclasses import dataclass, field

from iarap.config.base_config import InstantiateConfig
from iarap.model.nn import MLP, FourierFeatsEncoding
from iarap.utils import to_immutable_dict, cross_skew_matrix

    

@dataclass
class NeuralRFConfig(InstantiateConfig):

    _target: Type = field(default_factory=lambda: NeuralRF)
    in_dim: int = 3
    num_layers: int = 8
    layer_width: int = 256
    out_dim: int = 3
    skip_connections: Tuple[int] = (4,)
    activation: str = 'Softplus'
    act_defaults: Dict[str, Any] = to_immutable_dict({'beta': 100})
    num_frequencies: int = 6
    encoding_with_input: bool = True
    geometric_init: bool = True


class NeuralRF(nn.Module):

    def __init__(self, config: NeuralRFConfig):
        super(NeuralRF, self).__init__()
        self.config = config
        self.dim = config.in_dim
        self.encoding = FourierFeatsEncoding(self.config.in_dim,
                                             self.config.num_frequencies,
                                             self.config.encoding_with_input)
        self.network = MLP(self.encoding.get_out_dim(),
                           self.config.num_layers,
                           self.config.layer_width,
                           self.config.out_dim,
                           self.config.skip_connections,
                           getattr(nn, self.config.activation)(**self.config.act_defaults))
        if self.config.geometric_init:
            self.geometric_init()

    def geometric_init(self):
        for j, lin in enumerate(self.network.layers):
            if j == len(self.network.layers) - 1:
                torch.nn.init.zeros_(lin.weight)
                torch.nn.init.zeros_(lin.bias)
                return  # Don't do weight normalization
            elif self.encoding is not None and j == 0:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.constant_(lin.weight, 0.0)
                torch.nn.init.normal_(lin.weight[:, :self.dim], 0.0, np.sqrt(2) / np.sqrt(lin.out_features))
            elif self.encoding is not None and j in self.network.skip_connections:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(lin.out_features))
                torch.nn.init.constant_(lin.weight[:, self.dim:self.network.in_dim], 0.0)
            else:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(lin.out_features))
            self.network.layers[j] = nn.utils.parametrizations.weight_norm(lin)

    def euler(self, x_in: Float[Tensor, "*batch in_dim"]) -> Float[Tensor, "*batch 3"]:
        return self.network(self.encoding(x_in))
    
    def euler_to_rotation(self, 
                          euler: Float[Tensor, "*batch 3"]
                          ) -> Float[Tensor, "*batch 3 3"]:
        coords = [f/2.0 for f in torch.split(euler, 1, dim=-1)]
        cx, cy, cz = [torch.cos(f) for f in coords]
        sx, sy, sz = [torch.sin(f) for f in coords]

        quaternion = torch.cat([
            cx*cy*cz - sx*sy*sz, cx*sy*sz + cy*cz*sx,
            cx*cz*sy - sx*cy*sz, cx*cy*sz + sx*cz*sy
        ], dim=-1)

        norm_quat = quaternion / quaternion.norm(p=2, dim=-1, keepdim=True)
        w, x, y, z = torch.split(norm_quat, 1, dim=-1)

        B = quaternion.shape[:-1]

        w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
        wx, wy, wz = w * x, w * y, w * z
        xy, xz, yz = x * y, x * z, y * z

        rot_mat = torch.stack([
            w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
            2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
            2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2
        ], dim=-1).view(*B, 3, 3)
        return rot_mat
     
    def forward(self, 
                x_in: Float[Tensor, "*batch in_dim"],
                return_euler: bool = False
                ) -> Dict[str, Float[Tensor, "*batch f"]]:
        outputs = {}
        euler = self.euler(x_in)
        outputs['rot'] = self.euler_to_rotation(euler)
        if return_euler:
            outputs['euler'] = euler
        return outputs