import torch

import torch.autograd as ad
import torch.nn as nn
import numpy as np

from torch import Tensor
from jaxtyping import Float
from typing import Optional, Dict, Type, Tuple, Any
from dataclasses import dataclass, field

from iarap.config.base_config import InstantiateConfig
from iarap.model.nn import MLP, FourierFeatsEncoding
from iarap.utils import to_immutable_dict



class SDF(nn.Module):

    def __init__(self, dim: int):
        super(SDF, self).__init__()
        self.dim = dim

    def distance(self, x_in: Float[Tensor, "*batch in_dim"]) -> Float[Tensor, "*batch 1"]:
        raise NotImplementedError()
    
    def gradient(self, 
                 x_in: Float[Tensor, "*batch in_dim"],
                 dist: Optional[Float[Tensor, "*batch 1"]] = None) -> Float[Tensor, "*batch 3"]:
        raise NotImplementedError()
    

@dataclass
class NeuralSDFConfig(InstantiateConfig):

    _target: Type = field(default_factory=lambda: NeuralSDF)
    in_dim: int = 3
    num_layers: int = 8
    layer_width: int = 256
    out_dim: int = 1
    skip_connections: Tuple[int] = (4,)
    activation: str = 'Softplus'
    act_defaults: Dict[str, Any] = to_immutable_dict({'beta': 100})
    num_frequencies: int = 6
    encoding_with_input: bool = True
    geometric_init: bool = True
    radius_init: float = 0.5


class NeuralSDF(SDF):

    def __init__(self, config: NeuralSDFConfig):
        super(NeuralSDF, self).__init__(config.in_dim)
        self.config = config
        self.network = MLP(self.config.in_dim,
                           self.config.num_layers,
                           self.config.layer_width,
                           self.config.out_dim,
                           self.config.skip_connections,
                           getattr(nn, self.config.activation)(**self.config.act_defaults))
        self.encoding = FourierFeatsEncoding(self.config.in_dim,
                                             self.config.num_frequencies,
                                             self.config.encoding_with_input)
        if self.config.geometric_init:
            self.geometric_init(self.config.radius_init)

    def geometric_init(self, rad):
        for j, lin in enumerate(self.network.layers):
            if j == len(self.network.layers) - 1:
                torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(lin.in_features), std=0.0001)
                torch.nn.init.constant_(lin.bias, -rad)
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

    def distance(self, x_in: Float[Tensor, "*batch in_dim"]) -> Float[Tensor, "*batch 1"]:
        return self.network(self.encoding(x_in))
    
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
                       only_inputs=True)
    
    def forward(self, 
                x_in: Float[Tensor, "*batch in_dim"],
                with_grad: bool = False,
                differentiable_grad: bool = False) -> Dict[str, Float[Tensor, "*batch f"]]:
        outputs = {}
        x = x_in
        if with_grad:
            x = x.requires_grad_()
        dist = self.distance(x)
        outputs['dist'] = dist
        if with_grad:
            grad = self.gradient(x, dist, differentiable_grad)
            outputs['grad'] = grad
        return outputs