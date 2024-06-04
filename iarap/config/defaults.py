import tyro

from pathlib import Path

from iarap.data.mesh import MeshDataConfig
from iarap.model.nn.invertible import InvertibleMLP3DConfig, InvertibleRtMLPConfig
from iarap.model.nn.loss import DeformationLossConfig, IGRConfig
from iarap.model.neural_rtf import NeuralRTFConfig
from iarap.model.neural_sdf import NeuralSDFConfig
from iarap.model.nn.mlp import MLPConfig
from iarap.render.animate import AnimatorConfig
from iarap.render.sdf_renderer import SDFRendererConfig
from iarap.train.deform_trainer import DeformTrainerConfig, MCDeformTrainerConfig
from iarap.train.optim import AdamConfig, MultiStepSchedulerConfig
from iarap.train.sdf_trainer import SDFTrainerConfig



reconstruct_entrypoint = SDFTrainerConfig(
    num_steps=10000,
    data=MeshDataConfig(
        file=Path('assets\\mesh\\chomper.stl'),
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

deform_mesh_entrypoint = DeformTrainerConfig(    
    num_steps=1000,
    pretrained_shape=Path('./assets/weights/sdf/holed_sculpture.pt'),
    handles_spec=Path('assets/constraints/holed_sculpture/bending.yaml'),
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

deform_sdf_entrypoint = DeformTrainerConfig(    
    num_steps=1000,
    pretrained_shape=Path('./assets/weights/sdf/armadillo.pt'),
    handles_spec=Path('assets/constraints/armadillo/overlap.yaml'),
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
        network=InvertibleMLP3DConfig()  # InvertibleRtMLPConfig()
    ),
    loss=DeformationLossConfig(
        moving_handle_loss_w=1000,  # 1000, 
        static_handle_loss_w=1000,  # 1000, 
        arap_loss_w=1000
    ),
    optimizer=AdamConfig(
        lr=1e-3
    ),
    scheduler=MultiStepSchedulerConfig()
)


'''deform_mesh_entrypoint = MCDeformTrainerConfig(    
    num_steps=1000,
    pretrained_shape=Path('./assets/weights/sdf/buddha.pt'),
    handles_spec=Path('assets/constraints/buddha/bust_rotation_L.yaml'),
    mc_resolution=256,
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
)'''


render_sdf_entrypoint = SDFRendererConfig(
    load_shape=Path('assets/weights/sdf/dragon.pt'),
    shape_type='sdf',
    deform_mode='implicit',
    deformation_model=NeuralRTFConfig(
        network=InvertibleMLP3DConfig()
    ),
    # load_deformation=Path('wandb/buddha-head_rotate-mlp/files/checkpoints/neural_rotation.pt'),
    chunk=300000,
    resolution=512
)

render_mesh_entrypoint = SDFRendererConfig(
    load_shape=Path('assets/mesh/holed_sculpture.stl'),
    shape_type='mesh',
    deformation_model=NeuralRTFConfig(
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
    # load_deformation=Path('wandb/armadillo-overlap-mlp/files/checkpoints/neural_rotation.pt'),
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
    'reconstruct': reconstruct_entrypoint,
    'deform-sdf': deform_sdf_entrypoint,
    'deform-mesh': deform_mesh_entrypoint,
    'render-sdf': render_sdf_entrypoint,
    'render-mesh': render_mesh_entrypoint,
    'animate': animate_entrypoint
}

Commands = tyro.conf.SuppressFixed[tyro.conf.FlagConversionOff[
    tyro.extras.subcommand_type_from_defaults(defaults=Defaults)
]]
