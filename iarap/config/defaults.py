import tyro

from pathlib import Path

from iarap.data.mesh import MeshDataConfig
from iarap.model.nn.invertible import InvertibleMLP3DConfig
from iarap.model.nn.loss import DeformationLossConfig, IGRConfig
from iarap.model.neural_rtf import NeuralRTFConfig
from iarap.model.neural_sdf import NeuralSDFConfig
from iarap.model.nn.mlp import MLPConfig
from iarap.render.animate import AnimatorConfig
from iarap.render.sdf_renderer import SDFRendererConfig
from iarap.train.deform_trainer import DeformTrainerConfig, MCDeformTrainerConfig
from iarap.train.optim import AdamConfig, MultiStepSchedulerConfig
from iarap.train.sdf_trainer import SDFTrainerConfig



train_sdf_entrypoint = SDFTrainerConfig(
    num_steps=10000,
    data=MeshDataConfig(
        file=Path('assets\\mesh\\TrollRghtHand.stl'),
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
    pretrained_shape=Path('./assets/weights/sdf/hand.pt'),
    handles_spec=Path('assets/constraints/hand/close.yaml'),
    delaunay_sample=30,
    zero_samples=1000,
    space_samples=1000,
    attempts_per_step=10000,
    near_surface_threshold=0.01,
    domain_bounds=(-1, 1),
    num_projections=5,
    plane_coords_scale=0.03,  # 
    device='cuda',
    shape_model=NeuralSDFConfig(),
    rotation_model=NeuralRTFConfig(
        network=MLPConfig(
            in_dim=3,
            num_layers=8,
            layer_width=256,
            out_dim=6,
            skip_connections=(4,),
            activation='Softplus',
            act_defaults={'beta': 100},
            num_frequencies=6,
            encoding_with_input=True,
            geometric_init=True
        )
    ),
    loss=DeformationLossConfig(
        moving_handle_loss_w=1000,  # 1000, 
        static_handle_loss_w=1000,  # 1000, 
        arap_loss_w=10
    ),
    optimizer=AdamConfig(
        lr=1e-3
    ),
    scheduler=MultiStepSchedulerConfig()
)
'''
deform_sdf_entrypoint = MCDeformTrainerConfig(    
    num_steps=1000,
    pretrained_shape=Path('./assets/weights/sdf/dino.pt'),
    handles_spec=Path('assets/constraints/dino/arap_snout_experiment.yaml'),
    mc_resolution=128,
    mc_level_bounds=(-0.1, 0.2),
    domain_bounds=(-1, 1),   
    device='cuda',
    chunk=300000,
    shape_model=NeuralSDFConfig(),
    rotation_model=NeuralRTFConfig(),
    loss=DeformationLossConfig(
        moving_handle_loss_w=1000,  # 1000, 
        static_handle_loss_w=1000,  # 1000, 
        arap_loss_w=10
    ),
    optimizer=AdamConfig(
        lr=1e-3
    ),
    scheduler=MultiStepSchedulerConfig()
)
'''
render_sdf_entrypoint = SDFRendererConfig(
    load_shape=Path('assets/weights/sdf/hand.pt'),
    # load_shape=Path('assets/weights/sdf/buddha.pt'),
    shape_type='sdf',
    # load_deformation=Path('wandb/run-20240304_160841-bh9r4jzt/files/checkpoints/neural_rotation.pt'),
    # load_deformation=Path('wandb/run-20240314_103509-58hdqszn/files/checkpoints/neural_rotation.pt'),
    # load_deformation=Path('wandb/buddha_bust_rotate_L/files/checkpoints/neural_rotation.pt'),
    load_deformation=Path('wandb/offline-run-20240429_194827-i7iwdf88/files/checkpoints/neural_rotation.pt'),
    chunk=300000,
    resolution=512
)

animate_entrypoint = AnimatorConfig(
    load_shape=Path('assets/weights/sdf/cubes.pt'),
    load_deformation=Path('wandb/cubes-bending-mlp/files/checkpoints/neural_rotation.pt'),
    chunk=300000,
    resolution=512,
    ffmpeg_path='ffmpeg',
    camera_origin=(0.0, 0.0, 2.5)
)


Defaults = {
    'train-sdf': train_sdf_entrypoint,
    'deform-sdf': deform_sdf_entrypoint,
    'render-sdf': render_sdf_entrypoint,
    'animate': animate_entrypoint
}

Commands = tyro.conf.SuppressFixed[tyro.conf.FlagConversionOff[
    tyro.extras.subcommand_type_from_defaults(defaults=Defaults)
]]
