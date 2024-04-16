from __future__ import annotations

import torch

import numpy as np
import torch.nn as nn

from torch import Tensor
from typing import Any, Dict, Optional, Tuple, Set, Type
from jaxtyping import Float
from dataclasses import dataclass, field

from iarap.config.base_config import InstantiateConfig
from iarap.model.nn.encoding import FourierFeatsEncoding
from iarap.utils.misc import to_immutable_dict


class MLP(nn.Module):
    """Multilayer perceptron

    Args:
        in_dim: Input layer dimension
        num_layers: Number of network layers
        layer_width: Width of each MLP layer
        out_dim: Output layer dimension. Uses layer_width if None.
        activation: intermediate layer activation function.
        out_activation: output activation function.
    """

    def __init__(
        self,
        config: MLPConfig
    ) -> None:
        super(MLP, self).__init__()
        self.in_dim = config.in_dim
        assert self.in_dim > 0
        self.out_dim = config.out_dim if config.out_dim is not None else config.layer_width
        self.num_layers = config.num_layers
        self.layer_width = config.layer_width
        self.skip_connections = config.skip_connections
        self._skip_connections: Set[int] = set(self.skip_connections) if self.skip_connections else set()
        self.activation = getattr(nn, config.activation)(**config.act_defaults)
        self.out_activation = config.out_activation
        if config.num_frequencies > 0:
            self.encoding = FourierFeatsEncoding(self.in_dim, config.num_frequencies, config.encoding_with_input)
            self.mlp_in_dim = self.encoding.get_out_dim()
        else:
            self.encoding = None
            self.mlp_in_dim = self.in_dim

        self.build_nn_modules()
        if config.geometric_init:
            self.geometric_init()

    def build_nn_modules(self) -> None:
        """Initialize multi-layer perceptron."""
        layers = []
        if self.num_layers == 1:
            layers.append(nn.Linear(self.mlp_in_dim, self.out_dim))
        else:
            for i in range(self.num_layers - 1):
                if i == 0:
                    assert i not in self._skip_connections, "Ignoring skip connection at layer 0."
                    layers.append(nn.Linear(self.mlp_in_dim, self.layer_width))
                elif i in self._skip_connections:
                    layers.append(nn.Linear(self.layer_width + self.mlp_in_dim, self.layer_width))
                else:
                    layers.append(nn.Linear(self.layer_width, self.layer_width))
            layers.append(nn.Linear(self.layer_width, self.out_dim))
        self.layers = nn.ModuleList(layers)

    def geometric_init(self):
        for j, lin in enumerate(self.layers):
            if j == len(self.layers) - 1:
                if self.out_dim > 1:
                    torch.nn.init.zeros_(lin.weight)
                    torch.nn.init.zeros_(lin.bias)
                    return  # Don't do weight normalization
                else:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(lin.in_features), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -0.5)
            elif self.encoding is not None and j == 0:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.constant_(lin.weight, 0.0)
                torch.nn.init.normal_(lin.weight[:, :self.in_dim], 0.0, np.sqrt(2) / np.sqrt(lin.out_features))
            elif self.encoding is not None and j in self.skip_connections:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(lin.out_features))
                torch.nn.init.constant_(lin.weight[:, self.in_dim:self.mlp_in_dim], 0.0)
            else:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(lin.out_features))
            self.layers[j] = nn.utils.parametrizations.weight_norm(lin)

    def forward(self, in_tensor: Float[Tensor, "*bs in_dim"]) -> Float[Tensor, "*bs out_dim"]:
        """Process input with a multilayer perceptron.

        Args:
            in_tensor: Network input

        Returns:
            MLP network output
        """
        in_tensor = self.encoding(in_tensor) if self.encoding is not None else in_tensor
        x = in_tensor
        for i, layer in enumerate(self.layers):
            # as checked in `build_nn_modules`, 0 should not be in `_skip_connections`
            if i in self._skip_connections:
                x = torch.cat([in_tensor, x], -1)
            x = layer(x)
            if self.activation is not None and i < len(self.layers) - 1:
                x = self.activation(x)
        if self.out_activation:
            x = self.activation(x)
        return x


@dataclass
class MLPConfig(InstantiateConfig):

    _target: Type = field(default_factory=lambda: MLP)

    in_dim: int = 3
    num_layers: int = 8
    layer_width: int = 256
    out_dim: int = 3
    geometric_init: bool = True
    skip_connections: Tuple[int] = (4,)
    activation: str = 'Softplus'
    act_defaults: Dict[str, Any] = to_immutable_dict({'beta': 100})
    out_activation: bool = False
    num_frequencies: int = 6
    encoding_with_input: bool = True