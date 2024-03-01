import torch.nn as nn
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
