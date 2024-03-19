import torch

import torch.nn as nn
import numpy as np

from torch import Tensor
from jaxtyping import Float
from typing import Dict, Type, Tuple, Any
from dataclasses import dataclass, field

from iarap.config.base_config import InstantiateConfig
from iarap.model.base_sdf import SDF
from iarap.model.nn import MLP, FourierFeatsEncoding
from iarap.utils import to_immutable_dict, euler_to_rotation

    

@dataclass
class NeuralRTFConfig(InstantiateConfig):

    _target: Type = field(default_factory=lambda: NeuralRTF)
    in_dim: int = 3
    num_layers: int = 8
    layer_width: int = 256
    out_dim: int = 6
    skip_connections: Tuple[int] = (4,)
    activation: str = 'Softplus'
    act_defaults: Dict[str, Any] = to_immutable_dict({'beta': 100})
    num_frequencies: int = 6
    encoding_with_input: bool = True
    geometric_init: bool = True


class NeuralRTF(SDF):

    def __init__(self, config: NeuralRTFConfig):
        super(NeuralRTF, self).__init__(config.in_dim)
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
        self.sdf_callable = lambda x: x.norm(dim=-1, keepdim=True) - 0.5

    def set_sdf_callable(self, dist_fn):
        self.sdf_callable = dist_fn

    def distance(self, x_in: Float[Tensor, "*batch in_dim"]) -> Float[Tensor, "*batch 1"]:
        x = self.transform(x_in)
        return self.sdf_callable(x)

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

    # def euler(self, x_in: Float[Tensor, "*batch in_dim"]) -> Float[Tensor, "*batch 3"]:
    #     return self.network(self.encoding(x_in))

    def forward(self, 
                x_in: Float[Tensor, "*batch in_dim"],
                return_euler: bool = False
                ) -> Dict[str, Float[Tensor, "*batch f"]]:
        outputs = {}
        # euler = self.euler(x_in)
        rt = self.network(self.encoding(x_in))
        euler, transl = rt[..., :3], rt[..., 3:]
        outputs['rot'] = euler_to_rotation(euler)
        outputs['transl'] = transl  
        if return_euler:
            outputs['euler'] = euler
        return outputs
    
    def transform(self, 
                  x_in: Float[Tensor, "*batch in_dim"],
                  ) -> Float[Tensor, "*batch in_dim"]:
        outputs = self(x_in)
        rotated = (outputs['rot'] @ x_in[..., None]).squeeze(-1) + outputs['transl']
        return rotated