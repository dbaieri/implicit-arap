import tyro

from pathlib import Path

from iarap.data.mesh import MeshDataConfig
from iarap.model.nn.loss import DeformationLossConfig, IGRConfig
from iarap.model.neural_rtf import NeuralRTFConfig
from iarap.model.neural_sdf import NeuralSDFConfig
from iarap.render.sdf_renderer import SDFRendererConfig
from iarap.train.deform_trainer import DeformTrainerConfig
from iarap.train.optim import AdamConfig, MultiStepSchedulerConfig
from iarap.train.sdf_trainer import SDFTrainerConfig



train_sdf_entrypoint = SDFTrainerConfig(
    num_steps=10000,
    data=MeshDataConfig(
        file=Path('assets\\mesh\\armadillo.ply'),
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
    num_steps=1000,
    pretrained_shape=Path('./assets/weights/sdf/armadillo.pt'),
    handles_spec=Path('assets/constraints/armadillo/arm_front.yaml'),
    delaunay_sample=30,
    zero_samples=1000,
    space_samples=1000,
    attempts_per_step=10000,
    near_surface_threshold=0.01,
    domain_bounds=(-1, 1),
    num_projections=5,
    plane_coords_scale=0.001,
    device='cuda',
    shape_model=NeuralSDFConfig(),
    rotation_model=NeuralRTFConfig(),
    loss=DeformationLossConfig(
        moving_handle_loss_w=1000,  # 1000, 
        static_handle_loss_w=1000,  # 1000, 
        arap_loss_w=100
    ),
    optimizer=AdamConfig(
        lr=1e-3
    ),
    scheduler=MultiStepSchedulerConfig()
)

render_sdf_entrypoint = SDFRendererConfig(
    load_shape=Path('assets/weights/sdf/armadillo.pt'),
    # load_deformation=Path('wandb/run-20240304_160841-bh9r4jzt/files/checkpoints/neural_rotation.pt'),
    # load_deformation=Path('wandb/run-20240314_103509-58hdqszn/files/checkpoints/neural_rotation.pt'),
    # load_deformation=Path('wandb/buddha_bust_rotate_L/files/checkpoints/neural_rotation.pt'),
    load_deformation=Path('wandb/run-20240402_172857-232tw1bj/files/checkpoints/neural_rotation.pt'),
    chunk=300000,
    resolution=512
)


Defaults = {
    'train-sdf': train_sdf_entrypoint,
    'deform-sdf': deform_sdf_entrypoint,
    'render-sdf': render_sdf_entrypoint
}

Commands = tyro.conf.SuppressFixed[tyro.conf.FlagConversionOff[
    tyro.extras.subcommand_type_from_defaults(defaults=Defaults)
]]
