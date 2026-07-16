import torch
from utils.graphics_utils import apply_rotation, batch_quaternion_multiply

def _compute_smooth_regularization_71(self,
                                    xyz, 
                                    scales_emb, 
                                    rotations_emb, 
                                    opacity_emb, 
                                    shs_emb,
                                    time,
                                    out_pts, 
                                    out_scales, 
                                    out_rot, 
                                    out_opacity, 
                                    out_shs,
                                    smooth_cfg,
                                    orig_cache=None):
    """
    Compute smooth regularization loss on the deformation outputs.
    The idea is: for a small perturbation in the input (both spatial and temporal),
    the deformation network should produce similar outputs.

    This function re-computes the hidden representation and each branch's deformation
    for the original and perturbed inputs, then calculates an L2 loss for each branch,
    weighted by cfg, and returns the averaged loss.
    """
    # Retrieve perturbation magnitudes from config (or use defaults)
    spatial_eps = smooth_cfg.get("spatial_eps", 5e-3)
    temporal_eps = smooth_cfg.get("temporal_eps", 5e-3)
    # temporal_eps = 0.0
    
    choice = None
    if smooth_cfg['downsample_ratio'] < 1.0:
        # == Choose Random Points == 
        device = xyz.device
        N_all = xyz.shape[0]
        subN = int(max(1, N_all * smooth_cfg['downsample_ratio']))
        choice = torch.randperm(N_all, device=device)[:subN]

        # (a) Subset + detach
        xyz   = xyz[choice,:3].clone().detach()
        scales_emb    = scales_emb[choice, :3].clone().detach()
        rotations_emb   = rotations_emb[choice, :4].clone().detach()
        time     = time[choice,:1].clone().detach()   # [subN,1]
    
    # === Compute original deformation outputs ===
    if orig_cache is None:
        hidden_orig = self.query_time(xyz, scales_emb, rotations_emb, None, time)
        dx_orig = self.pos_deform(hidden_orig) if not self.args.no_dx else None
        ds_orig = self.scales_deform(hidden_orig) if not self.args.no_ds else None
        dr_orig = self.rotations_deform(hidden_orig) if not self.args.no_dr else None
        do_orig = self.opacity_deform(hidden_orig) if not self.args.no_do else None
        dshs_orig = (
            self.shs_deform(hidden_orig).reshape(xyz.shape[0], 16, 3)
            if not self.args.no_dshs else None
        )
    else:
        dx_orig = orig_cache.get("dx") if orig_cache else None
        ds_orig = orig_cache.get("ds") if orig_cache else None
        dr_orig = orig_cache.get("dr") if orig_cache else None
        do_orig = orig_cache.get("do") if orig_cache else None
        dshs_orig = orig_cache.get("dshs") if orig_cache else None
        if not self.args.no_dx and dx_orig is None:
            raise ValueError("orig_cache missing 'dx' for smooth regularization.")
        if not self.args.no_ds and ds_orig is None:
            raise ValueError("orig_cache missing 'ds' for smooth regularization.")
        if not self.args.no_dr and dr_orig is None:
            raise ValueError("orig_cache missing 'dr' for smooth regularization.")
        if not self.args.no_do and do_orig is None:
            raise ValueError("orig_cache missing 'do' for smooth regularization.")
        if not self.args.no_dshs and dshs_orig is None:
            raise ValueError("orig_cache missing 'dshs' for smooth regularization.")
        if choice is not None:
            if dx_orig is not None:
                dx_orig = dx_orig.index_select(0, choice)
            if ds_orig is not None:
                ds_orig = ds_orig.index_select(0, choice)
            if dr_orig is not None:
                dr_orig = dr_orig.index_select(0, choice)
            if do_orig is not None:
                do_orig = do_orig.index_select(0, choice)
            if dshs_orig is not None:
                dshs_orig = dshs_orig.index_select(0, choice)

    # === Generate perturbed inputs ===
    # Perturb the spatial (rays_pts_emb) and temporal (time_emb) inputs.
    xyz = self.normalize_aabb(xyz, self.grid.get_aabb)
    perturbed_rays = xyz + (torch.rand_like(xyz) * 2 - 1) * spatial_eps
    perturbed_time = time + (torch.rand_like(time) * 2 - 1) * temporal_eps

    # Compute the hidden representation for the perturbed inputs.
    hidden_pert = self.query_time(perturbed_rays, scales_emb, rotations_emb, None, perturbed_time, need_normalize=False)
    # hidden_pert = self.feature_out(hidden_pert)
    
    dx_pert = self.pos_deform(hidden_pert) if dx_orig is not None else None
    ds_pert = self.scales_deform(hidden_pert) if ds_orig is not None else None
    dr_pert = self.rotations_deform(hidden_pert) if dr_orig is not None else None
    do_pert = self.opacity_deform(hidden_pert) if do_orig is not None else None
    dshs_pert = (self.shs_deform(hidden_pert)
                 .reshape(xyz.shape[0], 16, 3)) if dshs_orig is not None else None

    # === Compute L2 differences per branch ===
    loss = 0.0
    count = 0
    if dx_orig is not None:
        loss_dx = torch.mean((dx_orig - dx_pert)**2)
        loss += smooth_cfg.get("w_dx", 1.0) * loss_dx
        count += 1
    if ds_orig is not None:
        loss_ds = torch.mean((ds_orig - ds_pert)**2)
        loss += smooth_cfg.get("w_ds", 1.0) * loss_ds
        count += 1
    if dr_orig is not None:
        loss_dr = torch.mean((dr_orig - dr_pert)**2)
        loss += smooth_cfg.get("w_dr", 1.0) * loss_dr
        count += 1
    if do_orig is not None:
        loss_do = torch.mean((do_orig - do_pert)**2)
        loss += smooth_cfg.get("w_do", 1.0) * loss_do
        count += 1
    if dshs_orig is not None:
        loss_dshs = torch.mean((dshs_orig - dshs_pert)**2)
        loss += smooth_cfg.get("w_dshs", 1.0) * loss_dshs
        count += 1
    assert count > 0, "No deformation branches enabled for smooth regularization."
    # Return the averaged loss across the branches used.
    return loss / (count if count > 0 else 1.0)

def _compute_smooth_regularization_72(self,
                                    xyz, 
                                    scales_emb, 
                                    rotations_emb, 
                                    opacity_emb, 
                                    shs_emb,
                                    time,
                                    out_pts, 
                                    out_scales, 
                                    out_rot, 
                                    out_opacity, 
                                    out_shs,
                                    smooth_cfg):
    """
    Compute smooth regularization loss on the deformation outputs.
    The idea is: for a small perturbation in the input (both spatial and temporal),
    the deformation network should produce similar outputs.

    This function re-computes the hidden representation and each branch's deformation
    for the original and perturbed inputs, then calculates an L2 loss for each branch,
    weighted by cfg, and returns the averaged loss.
    """
    # Retrieve perturbation magnitudes from config (or use defaults)
    spatial_eps = smooth_cfg.get("spatial_eps", 5e-3)
    temporal_eps = smooth_cfg.get("temporal_eps", 5e-3)
    # temporal_eps = 0.0
    
    if smooth_cfg['downsample_ratio'] < 1.0:
        # == Choose Random Points == 
        device = xyz.device
        N_all = xyz.shape[0]
        subN = int(max(1, N_all * smooth_cfg['downsample_ratio']))
        choice = torch.randperm(N_all, device=device)[:subN]

        # (a) Subset + detach
        xyz   = xyz[choice,:3].clone().detach()
        scales_emb    = scales_emb[choice, :3].clone().detach()
        rotations_emb   = rotations_emb[choice, :4].clone().detach()
        time     = time[choice,:1].clone().detach()   # [subN,1]
    
    # === Compute original deformation outputs ===
    # For consistency, re-run query_time and feature_out on the original input.
    hidden_orig = self.query_time(xyz, scales_emb, rotations_emb, None, time)
    
    dx_orig = self.pos_deform(hidden_orig) if not self.args.no_dx else None
    ds_orig = self.scales_deform(hidden_orig) if not self.args.no_ds else None
    dr_orig = self.rotations_deform(hidden_orig) if not self.args.no_dr else None
    do_orig = self.opacity_deform(hidden_orig) if not self.args.no_do else None
    dshs_orig = (self.shs_deform(hidden_orig)
                 .reshape(xyz.shape[0], 16, 3)) if not self.args.no_dshs else None

    # === Generate perturbed inputs ===
    # Perturb the spatial (rays_pts_emb) and temporal (time_emb) inputs.
    xyz = self.normalize_aabb(xyz, self.grid.get_aabb)
    perturbed_rays = xyz + (torch.rand_like(xyz) * 2 - 1) * spatial_eps
    perturbed_time = time + (torch.rand_like(time) * 2 - 1) * temporal_eps

    # Compute the hidden representation for the perturbed inputs.
    hidden_pert = self.query_time(perturbed_rays, scales_emb, rotations_emb, None, perturbed_time, need_normalize=False)
    # hidden_pert = self.feature_out(hidden_pert)
    
    dx_pert = self.pos_deform(hidden_pert) if dx_orig is not None else None
    ds_pert = self.scales_deform(hidden_pert) if ds_orig is not None else None
    dr_pert = self.rotations_deform(hidden_pert) if dr_orig is not None else None
    do_pert = self.opacity_deform(hidden_pert) if do_orig is not None else None
    dshs_pert = (self.shs_deform(hidden_pert)
                 .reshape(xyz.shape[0], 16, 3)) if dshs_orig is not None else None

    # === Compute L2 differences per branch ===
    loss = 0.0
    count = 0
    if dx_orig is not None:
        loss_dx = torch.mean((dx_orig - dx_pert)**2)
        loss += smooth_cfg.get("w_dx", 1.0) * loss_dx
        count += 1
    if ds_orig is not None:
        loss_ds = torch.mean((ds_orig - ds_pert)**2)
        loss += smooth_cfg.get("w_ds", 0.3) * loss_ds
        count += 1
    if dr_orig is not None:
        loss_dr = torch.mean((dr_orig - dr_pert)**2)
        loss += smooth_cfg.get("w_dr", 0.3) * loss_dr
        count += 1
    if do_orig is not None:
        loss_do = torch.mean((do_orig - do_pert)**2)
        loss += smooth_cfg.get("w_do", 0.1) * loss_do
        count += 1
    if dshs_orig is not None:
        loss_dshs = torch.mean((dshs_orig - dshs_pert)**2)
        loss += smooth_cfg.get("w_dshs", 0.1) * loss_dshs
        count += 1
    assert count > 0, "No deformation branches enabled for smooth regularization."
    # Return the averaged loss across the branches used.
    return loss / (count if count > 0 else 1.0)
