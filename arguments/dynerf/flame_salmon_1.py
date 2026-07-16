_base_ = './default.py'

ModelHiddenParams = dict(
    no_do=False,
    no_dshs=False,
    h2m_voxel_conf=dict(
        film_trunk_width=128,
        vertex_dhe_conf=dict(
            prime=200003,
            bucket_size=50000,
            num_hidden_layers=2,
        ),
    ),
)

OptimizationParams = dict(
    iterations=20000,
    position_lr_max_steps=20000,
    deformation_lr_final=1.6e-05,
    grid_lr_final=2e-06,
    opacity_head_lr_mult=0.2,
)
