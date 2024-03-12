from __future__ import annotations

import torch
import vedo
import mcubes
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import torch.nn.functional as F

from typing import Type, Literal
from dataclasses import dataclass, field
from pathlib import Path
from tqdm import tqdm

from iarap.config.base_config import InstantiateConfig
from iarap.model.rot_net import NeuralRFConfig
from iarap.model.sdf import NeuralSDFConfig
from iarap.utils import trilinear_interp, gradient
from iarap.utils.misc import detach_model



class SDFRenderer:

    def __init__(self, config: SDFRendererConfig):
        self.config = config
        self.setup_model()

    def setup_model(self):
        self.shape_model = self.config.shape_model.setup().to(self.config.device)
        self.shape_model.load_state_dict(torch.load(self.config.load_shape))
        detach_model(self.shape_model)
        if self.config.load_deformation is not None:
            self.deformation_model = self.config.deformation_model.setup().to(self.config.device)
            self.deformation_model.load_state_dict(torch.load(self.config.load_deformation))
            detach_model(self.deformation_model)
        else:
            self.deformation_model = None
        # self.shape_color = torch.zeros([self.config.resolution] * 3, dtype=torch.float)
        vol = self.make_volume()
        self.cached_sdf = self.evaluate_model(vol).numpy()

    def make_volume(self):
        steps = torch.linspace(self.config.min_coord, 
                               self.config.max_coord, 
                               self.config.resolution,
                               device=self.config.device)
        xx, yy, zz = torch.meshgrid(steps, steps, steps, indexing="ij")
        return torch.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T.float()

    @torch.no_grad()
    def evaluate_model(self, pts_volume):    
        f_eval = []
        for sample in tqdm(torch.split(pts_volume, self.config.chunk, dim=0)):
            f_eval.append(self.sdf_functional(sample.contiguous()).cpu())
        f_volume = torch.cat(f_eval, dim=0).reshape(*([self.config.resolution] * 3))
        return f_volume
    
    def extract_mesh(self, level=0.0):
        try:
            verts, faces = mcubes.marching_cubes(self.cached_sdf, level)
            verts /= self.config.resolution // 2
            verts -= 1.0
        except:
            verts = np.empty([0, 3], dtype=np.float32)
            faces = np.empty([0, 3], dtype=np.int32)
        return verts, faces
    
    # def sdf_functional_numpy(self, query):
    #     sample = torch.from_numpy(query).float().to(self.config.device).reshape(-1, 3).contiguous()
    #     return self.sdf_functional(sample).reshape(-1).cpu().numpy()
    
    def sdf_functional(self, query):
        sample = query
        if self.deformation_model is not None:
            rotations = self.deformation_model(sample)['rot']
            sample = (rotations.transpose(-1, -2) @ sample[..., None]).squeeze(-1)
        model_out = self.shape_model(sample)
        return model_out['dist']
    
    def project_nearest(self, query, n_its=5):
        query = torch.from_numpy(query).float().to(self.config.device).view(-1, 3).requires_grad_()
        for i in range(n_its):
            dist = self.sdf_functional(query)
            grad = F.normalize(gradient(dist, query), dim=-1)
            query = (query - dist * grad).detach().requires_grad_()
        return query.detach()

    # def color_functional_numpy(self, query):
    #     # idx = self.point_3d_to_grid_idx(query)
    #     # return self.shape_color[idx[:, 0], idx[:, 1], idx[:, 2]]
    #     query = np.clip(query, -0.99, 0.99)
    #     return trilinear_interp(self.shape_color.unsqueeze(0).unsqueeze(-1), 
    #                             torch.from_numpy(query).float().view(1, -1, 3),
    #                             self.config.resolution).reshape(-1, 1).cpu().numpy()
    
    # def sphere_trace(self, query, dir, n_its=5):
    #     query = torch.from_numpy(query).float().to(self.config.device).view(-1, 3)
    #     dir = torch.from_numpy(dir).float().to(self.config.device).view(-1, 3)
    #     for i in range(n_its):
    #         dist = self.sdf_functional(query)
    #         query = query + dist * dir
    #     return query

    # def set_picked(self, query):
    #     idx = self.point_3d_to_grid_idx(query)
    #     self.shape_color[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0

    # def point_3d_to_grid_idx(self, query):
    #     query = np.clip(query, -0.99, 0.99)
    #     query = (query + 1.0) / 2.0  # Now [0; 1]
    #     dx = 1.0 / self.config.resolution
    #     return np.floor_divide(query, dx).astype(np.int32)

    def render_volumetric_function(self, volume):
        volume = vedo.Volume(volume)
        plt = vedo.applications.IsosurfaceBrowser(volume, use_gpu=True, c='gold')
        plt.show(axes=7, bg2='lb').close()

    def run(self):

        ps.set_enable_vsync(False)
        ps.set_ground_plane_mode("none")

        picks = []
        output_file = "Enter output path for picked points"
        viewed_level_set = 0.0
        verts, faces = self.extract_mesh(level=0)

        def custom_callback():
            io = psim.GetIO()
            nonlocal picks, output_file, verts, faces, viewed_level_set

            _, output_file = psim.InputText("Output file", output_file)
            _, viewed_level_set = psim.SliderFloat("Level set", viewed_level_set, v_min=-1.0, v_max=1.0)

            if io.MouseClicked[0] and io.KeyCtrl:
                screen_coords = io.MousePos
                world_pos = ps.screen_coords_to_world_position(screen_coords)
                print(world_pos)
                if np.abs(world_pos).max() <= 1.0 and not np.isinf(world_pos).any():
                    world_pos = self.project_nearest(world_pos, n_its=10).squeeze().cpu().numpy()
                    picks.append(world_pos)
                    ps.register_point_cloud("PickedPoints", np.stack(picks, axis=0), enabled=True)
                    # self.set_picked(np.expand_dims(world_pos, axis=0))

            if psim.Button("Render"):
                # This code is executed when the button is pressed
                verts, faces = self.extract_mesh(level=viewed_level_set)
                ps.register_surface_mesh("NeuralSDF", verts, faces, enabled=True)
                # ps.render_implicit_surface_scalar("NeuralSDF", 
                #                                   self.sdf_functional_numpy, 
                #                                   self.color_functional_numpy,
                #                                   mode='sphere_march',
                #                                   hit_dist=1e-7,
                #                                   enabled=True)
                print("Rendering finished")

            if psim.Button("Save points"):
                if len(picks) > 0:
                    print(f"Saving selected points at {output_file}")
                    np.savetxt(output_file, np.stack(picks, axis=0))
                else:
                    print("No points to save.")

            if psim.Button("Clear points"):
                if len(picks) > 0:
                    picks = []
                    ps.remove_point_cloud("PickedPoints")
                    # self.shape_color[...] = 0.0

            psim.TextUnformatted("Current picks:\n" + \
                                 (str(np.stack(picks, axis=0)) if len(picks) > 0 else str([])))

        ps.init()
        # ps.render_implicit_surface_scalar("NeuralSDF", 
        #                                   self.sdf_functional_numpy, 
        #                                   self.color_functional_numpy,
        #                                   mode='sphere_march',
        #                                   hit_dist=1e-7,
        #                                   enabled=True)
        ps.register_surface_mesh("NeuralSDF", verts, faces, enabled=True)
        ps.set_user_callback(custom_callback)
        ps.show()

        # volume = self.make_volume()
        # f_volume = self.evaluate_model(volume)
        # self.render_volumetric_function(f_volume.numpy())

    
        


@dataclass
class SDFRendererConfig(InstantiateConfig):

    _target: Type = field(default_factory=lambda: SDFRenderer)
    load_shape: Path = Path('./assets/weights/armadillo.pt')
    load_deformation: Path = None
    min_coord: float = -1.0
    max_coord: float =  1.0
    resolution: int = 512
    chunk: int = 65536
    device: Literal['cpu', 'cuda'] = 'cuda'
    shape_model: NeuralSDFConfig = NeuralSDFConfig()
    deformation_model: NeuralRFConfig = NeuralRFConfig()
