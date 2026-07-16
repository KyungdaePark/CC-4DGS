ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [64, 64, 64, 150]
    },
    multires = [1,2],
    defor_depth = 6,
    net_width = 192, # 192 -> 160
    plane_tv_weight = 0.0002, # 2e-4
    l1_time_planes =  0.0001, # 1e-4
    no_do=False,
    no_dshs=False,
    no_ds=False,
    
    h2m_voxel_conf = {
        'resolutions': [(32, 32, 32), (64, 64, 64), (128, 128, 128), (256, 256, 256), (512, 512, 512)],
        'emb_dim_per_level': 16,
        'final_out_dim': 32,
        'film_trunk_width': 256,
        'vertex_dhe_conf': {
            'num_hash_functions': 128,
            'emb_dim': 16,
            'prime': [200003, 400009, 1000003, 2000003, 3000017],
            'bucket_size': [60000, 360000, 960000, 1800000, 2800000],
            'hidden_dim': 64,
            'num_hidden_layers': 4,
        },
        'combine_mode': 'concat',
    },  
        
    h2m_time_conf = {
        'freq_L': 24,
        'hidden': 256,
        'out_dim': 256,
    },
        
    h2m_render_mode = False

)
OptimizationParams = dict(
    dataloader=True,
    iterations = 20000,
    position_lr_max_steps = 20_000,
    batch_size=2,
    lambda_dssim = 0.2,
    lambda_smooth_init = 0.0,
    lambda_smooth_final = 0.05,
    lambda_smooth_start_iter = 3000,
    lambda_smooth_end_iter = 16000,
    coarse_iterations = 5000,
    densify_until_iter = 18000,
    opacity_reset_interval = 60000, # 4dgs
    opacity_threshold_coarse = 0.005,
    opacity_threshold_fine_init = 0.005,
    opacity_threshold_fine_after = 0.005,

    
    
    # deformation_lr_init = 0.00016,
    # deformation_lr_final = 0.00002, # 1.6e-5 -> 2e-5

    # grid_lr_init = 0.0002,
    # grid_lr_final = 0.00002, # 2e-6 -> 2e-5    

    deformation_lr_init = 0.00016,
    deformation_lr_final = 0.00002,

    grid_lr_init = 0.0002,
    grid_lr_final = 0.00002,    

    dynamic_thro = 7.0,
    
    # pruning_interval = 2000
)
