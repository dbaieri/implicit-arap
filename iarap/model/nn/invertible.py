from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np

from dataclasses import dataclass, field
from typing import Any, Dict, Type, Tuple

from iarap.config.base_config import InstantiateConfig
from iarap.model.nn.encoding import FourierFeatsEncoding, LipBoundedPosEnc
from iarap.model.nn.mlp import MLP
from iarap.utils.linalg import euler_to_rotation
from iarap.utils.misc import to_immutable_dict



def coordinate_split(form, mode, inverse=False):
    focus = {2 - mode if form == 0 else mode}
    other = sorted(list({0, 1, 2} - focus))
    focus = list(focus)
    return (focus, other) if not inverse else (other, focus)

def combine_coords(x_other, x_focus, other, focus):
    coords = [None] * 3
    coords[focus[0]] = x_focus
    coords[other[0]] = x_other[..., [0]]
    coords[other[1]] = x_other[..., [1]]
    return torch.cat(coords, dim=-1)

def euler2rot_2dinv(euler_angle):
    # (B1, ..., Bn, 1) -> (B1, ..., Bn, 2, 2)
    theta = euler_angle.unsqueeze(-1)
    rot = torch.cat((
        torch.cat((theta.cos(), -theta.sin()), -2),
        torch.cat((theta.sin(), theta.cos()), -2),
    ), -1)

    return rot

def euler2rot_2d(euler_angle):
    # (B1, ..., Bn, 1) -> (B1, ..., Bn, 2, 2)
    theta = euler_angle.unsqueeze(-1)
    rot = torch.cat((
        torch.cat((theta.cos(), theta.sin()), -2),
        torch.cat((-theta.sin(), theta.cos()), -2),
    ), -1)

    return rot


class InvertibleMLP3D(nn.Module):

    def __init__(self,
                 config: InvertibleMLP3DConfig):
        super(InvertibleMLP3D, self).__init__()

        self.blocks = config.num_blocks * 3  # At least one block per coordinate
        self.skips = config.skip_connections
        self.freqs = config.num_frequencies
        self.width = config.layer_width
        self.layers = config.num_layers
        self.activation = config.activation
        self.act_defaults = config.act_defaults

        self.make_deform_net()
        if config.geometric_init:
            self.geometric_init()

    def make_deform_net(self):
        # part a : xy -> z
        self.encoding_xy = FourierFeatsEncoding(2, self.freqs, True)
        in_channels = self.encoding_xy.get_out_dim()
        self.dims_xy = [in_channels] + [self.width for _ in range(self.layers)] + [1]
        
        self.blocks_xy = nn.ModuleList()
        num_layers = len(self.dims_xy)
        for i_b in range(self.blocks):
            deform = MLP(
                in_dim=in_channels,
                num_layers=num_layers-1,
                layer_width=self.width,
                skip_connections=self.skips,
                activation=self.activation,
                act_defaults=self.act_defaults,
                out_activation=True,
                geometric_init=False
            )
            deform_out = nn.Linear(self.width, self.dims_xy[-1]) 
            self.blocks_xy.append(nn.Sequential(deform, deform_out))

        # part b : z -> xy
        self.encoding_z = FourierFeatsEncoding(1, self.freqs, True)

        in_channels = self.encoding_z.get_out_dim()
        self.dims_z = [in_channels] + [self.width for _ in range(self.layers)] + [3]
        
        self.blocks_z = nn.ModuleList()
        num_layers = len(self.dims_z)
        for i_b in range(self.blocks):
            deform = MLP(
                in_dim=in_channels,
                num_layers=num_layers-1,
                layer_width=self.width,
                skip_connections=self.skips,
                activation=self.activation,
                act_defaults=self.act_defaults,
                out_activation=True,
                geometric_init=False
            )
            deform_out = nn.Linear(self.width, self.dims_z[-1]) 
            self.blocks_z.append(nn.Sequential(deform, deform_out))

    def geometric_init(self):
        def block_init(block, dims, in_dim, skips):
            deform, deform_out = block[0], block[1]
            for j, lin in enumerate(deform.layers):
                if self.freqs > 0 and j == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight, 0.0)
                    torch.nn.init.normal_(lin.weight[:, :in_dim], 0.0, np.sqrt(2) / np.sqrt(lin.out_features))
                elif self.freqs > 0 and j in skips:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(lin.out_features))
                    torch.nn.init.constant_(lin.weight[:, in_dim:dims[0]], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(lin.out_features))
                deform.layers[j] = nn.utils.parametrizations.weight_norm(lin)
            torch.nn.init.zeros_(deform_out.weight)
            torch.nn.init.zeros_(deform_out.bias)

        for i in range(self.blocks):
            block_init(self.blocks_xy[i], self.dims_xy, 2, self.skips)
            block_init(self.blocks_z[i], self.dims_z, 1, self.skips)

    def forward(self, input_pts):
        x = input_pts
        ones = torch.ones(*x.shape[:-1], 4, device=x.device)
        T = torch.diag_embed(ones)
        for i_b in range(self.blocks):
            form = (i_b // 3) % 2
            mode = i_b % 3
            focus, other = coordinate_split(form, mode)

            # part a
            x_focus, x_other = x[..., focus], x[..., other]  # x
            x_diff = self.blocks_xy[i_b](self.encoding_xy(x_other))
            x_focus = x_focus - x_diff

            # part b
            focus_rt = self.blocks_z[i_b](self.encoding_z(x_focus))

            rot_2d = euler2rot_2dinv(focus_rt[..., [0]])
            if focus == [1]:
                rot_2d = rot_2d.transpose(-1, -2)
            trans_2d = focus_rt[..., 1:]
            x_other = (rot_2d @ (x_other - trans_2d)[..., None]).squeeze(-1)
            x = combine_coords(x_other, x_focus, other, focus)

            # aggregate rototranslation on "other" coordinates
            t_i = torch.diag_embed(ones)
            R_i = torch.diag_embed(ones)
            euler_i = torch.zeros_like(x)
            euler_i[..., focus] = -focus_rt[..., [0]]
            R_i[..., 0:3, 0:3] = euler_to_rotation(euler_i)
            t_i[..., other, 3] = -trans_2d
            t_i[..., focus, 3] = -x_diff
            T = R_i @ t_i @ T
            
        R = T[..., 0:3, 0:3]
        t = T[..., 0:3, 3]
        out_x = (R @ input_pts[..., None]).squeeze() + t
        return out_x, R, t

    def inverse(self, input_pts):
        x = input_pts
        for i_b in range(self.blocks - 1, -1, -1):
            form = (i_b // 3) % 2
            mode = i_b % 3
            focus, other = coordinate_split(form, mode, inverse=True)

            # part a
            x_focus, x_other = x[..., focus], x[..., other]  # x
            focus_rt = self.blocks_z[i_b](self.encoding_z(x_other))

            rot_2d = euler2rot_2d(focus_rt[..., [0]])
            if other == [1]:
                rot_2d = rot_2d.transpose(-1, -2)
            trans_2d = focus_rt[..., 1:]
            x_focus = (rot_2d @ x_focus[..., None]).squeeze(-1) + trans_2d

            # part a
            x_diff = self.blocks_xy[i_b](self.encoding_xy(x_focus))

            x_other = x_other + x_diff
            x = combine_coords(x_focus, x_other, focus, other)

        return x



def fixed_point_invert(g, y, iters=15, verbose=False, op='rotation'):
    with torch.no_grad():
        x = y
        dim = x.size(-1)
        for i in range(iters):
            block_out = g(x)
            if op == 'rotation':
                rot = euler_to_rotation(block_out)
                x = (rot.transpose(-1, -2) @ y[..., None]).squeeze(-1)
            elif op == 'translation':
                x = y - block_out
            else:
                raise NotImplementedError()
            if verbose:
                if op == 'rotation':
                    rot = euler_to_rotation(g(x))
                    test = (rot @ x[..., None]).squeeze(-1)
                elif op == 'translation':
                    test = x + g(x)
                err = (y - test).view(-1, dim).norm(dim=-1).mean()
                err = err.detach().cpu().item()
                print("iter:%d err:%s" % (i, err))
    return x


class InvertibleBlock(nn.Module):

    def __init__(self, 
                 in_dim, 
                 hidden_dim, 
                 out_dim, 
                 num_blocks=1,
                 activation=nn.Softplus(beta=100),
                 num_freqs=None):
        super(InvertibleBlock, self).__init__()
        self.op = None
        self.dim = in_dim
        self.nblocks = num_blocks

        self.pos_enc_freq = num_freqs
        if self.pos_enc_freq is not None:
            inp_dim_af_pe = self.dim * (self.pos_enc_freq * 2 + 1)
            self.pos_enc = LipBoundedPosEnc(self.dim, self.pos_enc_freq)
        else:
            self.pos_enc = lambda x: x
            inp_dim_af_pe = in_dim

        self.blocks = nn.ModuleList()
        self.blocks.append(nn.utils.spectral_norm(nn.Linear(inp_dim_af_pe, hidden_dim)))
        for _ in range(self.nblocks):
            self.blocks.append(nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim)))
        self.blocks.append(nn.utils.spectral_norm(nn.Linear(hidden_dim, out_dim)))

        self.activation = activation

    def forward_g(self, x):
        orig_dim = len(x.size())
        if orig_dim == 2:
            x = x.unsqueeze(0)

        y = self.pos_enc(x)
        for block in self.blocks[:-1]:
            y = self.activation(block(y))
        y = self.blocks[-1](y)

        if orig_dim == 2:
            y = y.squeeze(0)

        return y
    
    def inverse(self, y, verbose=False, iters=15):
        return fixed_point_invert(
            lambda x: self.forward_g(x), y, iters=iters, verbose=verbose, op=self.op
        )


class InvertibleRotBlock(InvertibleBlock):

    def __init__(self,  
                 in_dim, 
                 hidden_dim, 
                 num_blocks=1,
                 activation=nn.Softplus(beta=100),
                 num_freqs=None):
        super(InvertibleRotBlock, self).__init__(
            in_dim, hidden_dim, in_dim, num_blocks, activation, num_freqs)
        self.op = 'rotation'
        # No spectral norm on last layer since it predicts euler angles
        self.blocks[-1] = nn.Linear(hidden_dim, in_dim)  

    def forward(self, x):
        euler = self.forward_g(x)
        rot = euler_to_rotation(euler)
        return (rot @ x[..., None]).squeeze(-1), euler

    
class InvertibleResidualBlock(InvertibleBlock):

    def __init__(self,  
                 in_dim, 
                 hidden_dim, 
                 num_blocks=1,
                 activation=nn.ReLU(),
                 num_freqs=None):
        super(InvertibleResidualBlock, self).__init__(
            in_dim, hidden_dim, in_dim, num_blocks, activation, num_freqs)
        self.op = 'translation'

    def forward(self, x):
        transl = self.forward_g(x)
        return x + transl, transl


class InvertibleRtMLP(nn.Module):

    def __init__(self, 
                 config: InvertibleRtMLPConfig,):
        super(InvertibleRtMLP, self).__init__()
        self.in_dim = config.in_dim
        self.hidden_dim = config.layer_width
        self.num_g_blocks = config.num_g_blocks
        self.num_freqs = config.num_frequencies
        self.activation = getattr(nn, config.activation)(**config.act_defaults)

        # Network modules
        self.blocks = nn.ModuleList()
        self.blocks.append(InvertibleRotBlock(self.in_dim, 
                                              self.hidden_dim,
                                              num_blocks=self.num_g_blocks, 
                                              activation=self.activation,
                                              num_freqs=self.num_freqs))
        self.blocks.append(InvertibleResidualBlock(self.in_dim, 
                                                   self.hidden_dim,
                                                   num_blocks=self.num_g_blocks, 
                                                   activation=self.activation,
                                                   num_freqs=self.num_freqs))

    def forward(self, x):
        out = x
        out, euler = self.blocks[0](out)  
        out, transl = self.blocks[-1](out)
        
        rot = euler_to_rotation(euler)
        out_rt = (rot @ x[..., None]).squeeze(-1) + transl
        return out_rt, rot, transl

    def inverse(self, y, verbose=False, iters=15):
        x = y
        for block in self.blocks[::-1]:
            x = block.inverse(x, verbose=verbose, iters=iters)
        return x
    

@dataclass
class InvertibleMLP3DConfig(InstantiateConfig):

    _target: Type = field(default_factory=lambda: InvertibleMLP3D)

    num_blocks: int = 1
    layer_width: int = 128
    num_layers: int = 2
    skip_connections: Tuple[int] = (0,)
    num_frequencies: int = 6
    geometric_init: bool = True
    activation: str = 'Softplus'
    act_defaults: Dict[str, Any] = to_immutable_dict({'beta': 100})


@dataclass
class InvertibleRtMLPConfig(InstantiateConfig):

    _target: Type = field(default_factory=lambda: InvertibleRtMLP)

    in_dim: int = 3
    layer_width: int = 256
    num_frequencies: int = 6
    num_g_blocks: int = 4
    activation: str = 'Softplus'
    act_defaults: Dict[str, Any] = to_immutable_dict({'beta': 100})