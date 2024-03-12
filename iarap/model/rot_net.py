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
from iarap.utils import to_immutable_dict, euler_to_rotation

    

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

    def forward(self, 
                x_in: Float[Tensor, "*batch in_dim"],
                return_euler: bool = False
                ) -> Dict[str, Float[Tensor, "*batch f"]]:
        outputs = {}
        euler = self.euler(x_in)
        outputs['rot'] = euler_to_rotation(euler)
        if return_euler:
            outputs['euler'] = euler
        return outputs