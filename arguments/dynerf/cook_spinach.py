_base_ = './default_20k.py'

# Longer optimization schedule with a three-level CDF grid.
ModelHiddenParams = dict(
    net_width=192,
    h2m_voxel_conf=dict(
        # Three levels provide a compact, stable scene-specific configuration.
        resolutions=[(32, 32, 32), (64, 64, 64), (128, 128, 128)],
        emb_dim_per_level=16,
        final_out_dim=32,
        film_trunk_width=128,
        vertex_dhe_conf=dict(
            num_hash_functions=128,
            emb_dim=16,
            prime=[200003, 1000003, 3000007],
            bucket_size=[20000, 120000, 320000],
            hidden_dim=64,
            num_hidden_layers=4,
        ),
        combine_mode='concat',
    ),
)

OptimizationParams = dict(
    iterations=26000,
    position_lr_max_steps=26000,
    densify_until_iter=22000,
    grid_lr_final=8e-06,
    deformation_lr_final=1.5e-05,
    lambda_smooth_init=0.0,
    lambda_smooth_final=0.05,
    lambda_smooth_start_iter=3000,
    lambda_smooth_end_iter=16000,
)
