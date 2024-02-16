import torch

import torch.nn as nn

from torch import Tensor
from jaxtyping import Float


class FourierFeatsEncoding(nn.Module):
    """Multi-scale sinusoidal encodings. Support ``integrated positional encodings`` if covariances are provided.
    Each axis is encoded with frequencies ranging from 2^min_freq_exp to 2^max_freq_exp.

    Args:
        in_dim: Input dimension of tensor
        num_frequencies: Number of encoded frequencies per axis
        min_freq_exp: Minimum frequency exponent
        max_freq_exp: Maximum frequency exponent
        include_input: Append the input coordinate to the encoding
    """

    def __init__(
        self,
        in_dim: int,
        num_frequencies: int,
        include_input: bool = False
    ) -> None:
        super(FourierFeatsEncoding, self).__init__()

        assert in_dim > 0, "in_dim should be greater than zero"
        self.in_dim = in_dim
        self.num_frequencies = num_frequencies
        self.min_freq = 0.0
        self.max_freq = num_frequencies - 1.0
        self.include_input = include_input

    def get_out_dim(self) -> int:
        assert self.in_dim is not None, "Input dimension has not been set"
        out_dim = self.in_dim * self.num_frequencies * 2
        if self.include_input:
            out_dim += self.in_dim
        return out_dim

    def forward(
        self,
        in_tensor: Float[Tensor, "*bs input_dim"]
    ) -> Float[Tensor, "*bs output_dim"]:
        """Calculates NeRF encoding. 

        Args:
            in_tensor: For best performance, the input tensor should be between 0 and 1.
        Returns:
            Output values will be between -1 and 1
        """
        scaled_in_tensor = 2 * torch.pi * in_tensor  # scale to [0, 2pi]
        freqs = 2 ** torch.linspace(self.min_freq, self.max_freq, self.num_frequencies).to(in_tensor.device)
        scaled_inputs = scaled_in_tensor[..., None] * freqs  # [..., "input_dim", "num_scales"]
        scaled_inputs = scaled_inputs.view(*scaled_inputs.shape[:-2], -1)  # [..., "input_dim" * "num_scales"]

        encoded_inputs = torch.sin(torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1))

        if self.include_input:
            encoded_inputs = torch.cat([in_tensor, encoded_inputs], dim=-1)
        return encoded_inputs
