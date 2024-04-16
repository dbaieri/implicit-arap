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
from iarap.model.nn.mlp import MLPConfig
from iarap.utils import to_immutable_dict



@dataclass
class NeuralSDFConfig(InstantiateConfig):

    _target: Type = field(default_factory=lambda: NeuralSDF)
    network: MLPConfig = MLPConfig(
        in_dim=3,
        num_layers=8,
        layer_width=256,
        out_dim=1,
        skip_connections=(4,),
        activation='Softplus',
        act_defaults={'beta': 100},
        num_frequencies=6,
        encoding_with_input=True,
        geometric_init=True
    )


class NeuralSDF(SDF):

    def __init__(self, config: NeuralSDFConfig):
        super(NeuralSDF, self).__init__(config.network.in_dim)
        self.config = config
        self.network = self.config.network.setup()

    def distance(self, x_in: Float[Tensor, "*batch in_dim"]) -> Float[Tensor, "*batch 1"]:
        return self.network(x_in)

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