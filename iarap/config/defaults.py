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

deform_sdf_entrypoint = DeformTrainerConfig()

render_sdf_entrypoint = SDFRendererConfig(
    load_checkpoint=Path('assets/weights/armadillo.pt'),
)


Defaults = {
    'train-sdf': train_sdf_entrypoint,
    'deform-sdf': deform_sdf_entrypoint,
    'render-sdf': render_sdf_entrypoint
}

Commands = tyro.conf.SuppressFixed[tyro.conf.FlagConversionOff[
    tyro.extras.subcommand_type_from_defaults(defaults=Defaults)
]]
