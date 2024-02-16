import torch

import torch.nn as nn

from torch import Tensor
from jaxtyping import Float




class EikonalLoss(nn.Module):

    def __init__(self):
        super(EikonalLoss, self).__init__()

    def forward(self, grad: Float[Tensor, "*batch 3"]) -> Float[Tensor, "1"]:
        return (grad.norm(dim=-1) - 1.0).pow(2).mean()
    

class NormalsLoss(nn.Module):

    def __init__(self):
        super(NormalsLoss, self).__init__()

    def forward(self, 
                grad: Float[Tensor, "*batch 3"],
                normals: Float[Tensor, "*batch 3"]) -> Float[Tensor, "1"]:
        return (grad - normals).pow(2).sum(dim=-1).mean()
    
    
class ZeroLoss(nn.Module):

    def __init__(self):
        super(ZeroLoss, self).__init__()

    def forward(self, dist: Float[Tensor, "*batch 3"]) -> Float[Tensor, "1"]:
        return dist.abs().mean()