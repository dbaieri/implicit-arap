
import torch

import kaolin as kal
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from jaxtyping import Float, Int, Bool
from typing import Dict, Literal, Type
from dataclasses import dataclass, field

from iarap.config.base_config import InstantiateConfig


@dataclass
class IGRConfig(InstantiateConfig):

    _target: Type = field(default_factory=lambda: ImplicitGeometricRegularization)
    zero_sdf_surface_w: float = 1.0
    eikonal_error_w: float = 0.01
    normals_error_w: float = 0.01
    zero_penalty_w: float = 0.05


class ImplicitGeometricRegularization(nn.Module):

    def __init__(self, config: IGRConfig):
        super(ImplicitGeometricRegularization, self).__init__()
        self.config = config
        self.zero_loss = ZeroLoss()
        self.zero_penalty = ZeroPenalty()
        self.eikonal_loss = EikonalLoss()
        self.normals_loss = NormalsLoss()

    def forward(self, 
                pred_sdf_surf: Float[Tensor, "*batch 1"],
                pred_sdf_space: Float[Tensor, "*batch 1"],
                grad_sdf_surf: Float[Tensor, "*batch 3"],
                grad_sdf_space: Float[Tensor, "*batch 3"],
                surf_normals: Float[Tensor, "*batch 3"]
                ) -> Dict[str, Float[Tensor, "1"]]:
        zero_loss = self.config.zero_sdf_surface_w * self.zero_loss(pred_sdf_surf)
        zero_penalty = self.config.zero_penalty_w * self.zero_penalty(pred_sdf_space)
        eik_loss = self.config.eikonal_error_w * self.eikonal_loss(torch.cat([grad_sdf_space, grad_sdf_surf], dim=1))
        normals_loss = self.config.normals_error_w * self.normals_loss(grad_sdf_surf, surf_normals)
        return {'zero_loss': zero_loss, 
                'eikonal_loss': eik_loss, 
                'zero_penalty': zero_penalty,
                'normals_loss': normals_loss}


class EikonalLoss(nn.Module):

    def __init__(self):
        super(EikonalLoss, self).__init__()

    def forward(self, grad: Float[Tensor, "*batch 3"]) -> Float[Tensor, "1"]:
        return (grad.norm(dim=-1) - 1.0).abs().mean()
    

class NormalsLoss(nn.Module):

    def __init__(self):
        super(NormalsLoss, self).__init__()

    def forward(self, 
                grad: Float[Tensor, "*batch 3"],
                normals: Float[Tensor, "*batch 3"]) -> Float[Tensor, "1"]:
        return (1 - F.cosine_similarity(grad, normals, dim=-1)).mean()
    
    
class ZeroLoss(nn.Module):

    def __init__(self):
        super(ZeroLoss, self).__init__()

    def forward(self, dist: Float[Tensor, "*batch 1"]) -> Float[Tensor, "1"]:
        return dist.abs().mean()
    

class ZeroPenalty(nn.Module):

    def __init__(self, scale: float=100):
        super(ZeroPenalty, self).__init__()
        self.scale = scale

    def forward(self, dist: Float[Tensor, "*batch 1"]) -> Float[Tensor, "1"]:
        error = torch.exp(-self.scale * dist.abs())
        return error.mean()
    



@dataclass
class DeformationLossConfig(InstantiateConfig):

    _target: Type = field(default_factory=lambda: DeformationLoss)
    arap_loss_w: float = 1.0
    moving_handle_loss_w: float = 1.0
    static_handle_loss_w: float = 1.0


class DeformationLoss(nn.Module):

    def __init__(self, config: DeformationLossConfig):
        super(DeformationLoss, self).__init__()
        self.config = config
        self.arap_loss = PatchARAPLoss()
        self.handle_loss = nn.MSELoss()

    def forward(self,
                patch_verts: Float[Tensor, "p n 3"],
                faces: Int[Tensor, "m 3"],
                rotations: Float[Tensor, "p n 3 3"],
                translations: Float[Tensor, "p n 3"],
                moving_idx: Int[Tensor, "h_1 2"],
                static_idx: Int[Tensor, "h_2 2"],
                handle_value: Float[Tensor, "h 3"],
                alternation: Literal[0, 1]
                ) -> Float[Tensor, "1"]:
        patch_arap_loss = self.arap_loss(patch_verts, faces, rotations, translations, alternation)
        transformed_verts = (rotations @ patch_verts[..., None]).squeeze(-1) + translations
        moving_pos = transformed_verts[moving_idx[:, 0], moving_idx[:, 1], :]
        static_pos = transformed_verts[static_idx[:, 0], static_idx[:, 1], :]
        moving_handle_loss = self.handle_loss(moving_pos, handle_value[moving_idx[:, 0], :])
        static_handle_loss = self.handle_loss(static_pos, handle_value[static_idx[:, 0], :])
        '''
        handle_idx = torch.cat([moving_idx, static_idx], dim=0)
        handles = patch_verts[handle_idx[:, 0], handle_idx[:, 1], :].view(-1, 3)
        handle_trans = handle_value - handles
        d = torch.cdist(patch_verts.view(-1, 3), handles)
        w = 1. / (d + 1e-8)
        weight_mask = (d > 0.0).all(dim=-1)
        weighted_trans = (w[weight_mask, :] @ handle_trans) / w[weight_mask, :].sum(dim=-1, keepdim=True)
        zero_idx = (d == 0).nonzero()
        idw_interp = torch.zeros_like(patch_verts).view(-1, 3)
        idw_interp[zero_idx[:, 0], :] = handle_trans[zero_idx[:, 1], :]
        idw_interp[weight_mask, :] = weighted_trans
        interp_verts = patch_verts.view(-1, 3) + idw_interp

        handle_loss = ((transformed_verts.view(-1, 3) - interp_verts) ** 2).sum(dim=-1).mean()
        translation_loss = (translations ** 2).sum(dim=-1).mean()
        '''

        return {
            'arap_loss': patch_arap_loss * self.config.arap_loss_w,
            'moving_handle_loss': moving_handle_loss * self.config.moving_handle_loss_w,
            'static_handle_loss': static_handle_loss * self.config.static_handle_loss_w,
            # 'translation_loss': translation_loss
            # 'handle_loss': handle_loss * self.config.moving_handle_loss_w,
            # 'translation_loss': translation_loss * self.config.static_handle_loss_w
        }


class PatchARAPLoss(nn.Module):

    def __init__(self):
        super(PatchARAPLoss, self).__init__()

    def get_cot_weights(self,
                        patch_verts: Float[Tensor, "p n 3"],
                        faces: Int[Tensor, "m 3"]
                        ) -> Float[Tensor, "p n n"]:
        V = patch_verts.shape[1]

        face_verts = patch_verts[:, faces, :]
        v0, v1, v2 = face_verts[..., 0, :], face_verts[..., 1, :], face_verts[..., 2, :]
  
        idx = torch.cat([faces[:, :2], faces[:, 1:], faces[:, ::2]], dim=0).T

        A = (v1 - v2).norm(dim=-1)
        B = (v0 - v2).norm(dim=-1)
        C = (v0 - v1).norm(dim=-1)

        s = 0.5 * (A + B + C)
        area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=1e-12).sqrt()

        A2, B2, C2 = A * A, B * B, C * C
        cota = (B2 + C2 - A2) / area
        cotb = (A2 + C2 - B2) / area
        cotc = (A2 + B2 - C2) / area
        cot = torch.cat([cota, cotb, cotc], dim=1)
        cot /= 4.0

        w = torch.zeros((patch_verts.shape[0], V, V), device=patch_verts.device)
        w[:, idx[0, :], idx[1, :]] = cot

        w = w + w.transpose(-1, -2)
        return w

    def forward(self, 
                patch_verts: Float[Tensor, "p n 3"],
                faces: Int[Tensor, "m 3"],
                rotations: Float[Tensor, "p n 3 3"],
                translations: Float[Tensor, "p n 3"],
                alternation: Literal[0, 1]
                ) -> Float[Tensor, "1"]:
        w = self.get_cot_weights(patch_verts, faces)
        idx = torch.cat([faces[:, :2], faces[:, 1:], faces[:, ::2]], dim=0).T
        w_per_edge = w[:, idx[0, :], idx[1, :]]

        transformed_verts = (rotations @ patch_verts[..., None]).squeeze(-1) + translations
        rot_verts_edges = transformed_verts[:, idx[0, :], :] - transformed_verts[:, idx[1, :], :]

        edges_source = patch_verts[:, idx[0, :], :] - patch_verts[:, idx[1, :], :]
        rot_edges = (rotations[:, idx[0, :], ...] @ edges_source[..., None]).squeeze(-1) + translations[:, idx[0, :], :]

        if alternation == 0:
            rot_edges = rot_edges.detach()
        elif alternation == 1:
            rot_verts_edges = rot_verts_edges.detach()

        return (w_per_edge * (rot_edges - rot_verts_edges).pow(2).sum(dim=-1)).sum(dim=-1).mean(dim=0)