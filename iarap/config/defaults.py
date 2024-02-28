import tyro

from pathlib import Path

from iarap.data.mesh import MeshDataConfig
from iarap.model.nn.loss import IGRConfig
from iarap.model.sdf import NeuralSDFConfig
from iarap.render.sdf_renderer import SDFRendererConfig
from iarap.train.deform_trainer import DeformTrainerConfig
from iarap.train.optim import AdamConfig, MultiStepSchedulerConfig
from iarap.train.sdf_trainer import SDFTrainerConfig


train_sdf_entrypoint = SDFTrainerConfig(
    num_steps=10000,
    data=MeshDataConfig(
        file=Path('assets\mesh\\dragon.ply'),
        uniform_ratio=1.0
    ),
    model=NeuralSDFConfig(),
    loss=IGRConfig(
        zero_sdf_surface_w=3000,
        eikonal_error_w=100,
        normals_error_w=50,
        zero_penalty_w=3000
    ),
    optimizer=AdamConfig(
        lr=1e-4
    ),
    scheduler=MultiStepSchedulerConfig(
        milestones=(10000,)
    )
)

deform_sdf_entrypoint = DeformTrainerConfig(    
    num_steps=2000,
    pretrained_init=True,
    pretrained_shape=Path('./assets/weights/armadillo.pt'),
    delaunay_sample=30,
    zero_samples=200,
    attempts_per_step=10000,
    near_surface_threshold=0.01,
    domain_bounds=(-1, 1),
    num_projections=5,
    plane_coords_scale=0.001,
    device='cuda',
    model=NeuralSDFConfig(),
    loss=None,
    optimizer=AdamConfig(),
    scheduler=MultiStepSchedulerConfig()
)

render_sdf_entrypoint = SDFRendererConfig(
    load_checkpoint=Path('assets/weights/dragon.pt'),
)


Defaults = {
    'train-sdf': train_sdf_entrypoint,
    'deform-sdf': deform_sdf_entrypoint,
    'render-sdf': render_sdf_entrypoint
}

Commands = tyro.conf.SuppressFixed[tyro.conf.FlagConversionOff[
    tyro.extras.subcommand_type_from_defaults(defaults=Defaults)
]]
