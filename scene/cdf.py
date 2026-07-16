import torch
import torch.nn as nn
import math

def normalize_aabb(pts, aabb):
    norm = (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0
    return (norm + 1) / 2  # [0,1]

class TemporalFeatureEncoder(nn.Module):
    def __init__(self, freq_L=8, hidden=64, out_dim=32):
        super().__init__()

        self.freq_L = freq_L
        in_dim = 1 + 2*freq_L
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def freq_encode_1d(self, t):
        # t: [N]
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        N = t.shape[0]
        freqs = 2 ** torch.arange(self.freq_L, device=t.device)
        t_proj = t.unsqueeze(-1) * (2 * math.pi * freqs.view(1, 1, self.freq_L))
        sin_enc = torch.sin(t_proj).view(N, -1)
        cos_enc = torch.cos(t_proj).view(N, -1)
        out = torch.cat([t, sin_enc, cos_enc], dim=-1)  # [N, 1+2L]
        return out

    def forward(self, t):
        enc = self.freq_encode_1d(t)  # [N, 1+2L]
        out = self.mlp(enc)           # [N, out_dim]
        return out


class MultiResolutionVoxelHashField(nn.Module):
    def __init__(self, 
                 resolutions=[(32,32,32), (64,64,64)],
                 emb_dim_per_level=16,
                 final_out_dim=32,
                 vertex_dhe_conf=None,
                 combine_mode='weighted',
                 render_mode = False):
        super().__init__()
        self.levels = len(resolutions)
        self.emb_dim_per_level = emb_dim_per_level
        self.combine_mode = combine_mode
        self.grids = nn.ModuleList()
        prime_conf = None
        bucket_conf = None
        base_conf = dict(vertex_dhe_conf) if vertex_dhe_conf is not None else {}
        if 'prime' in base_conf and isinstance(base_conf['prime'], (list, tuple)):
            prime_conf = [int(v) for v in base_conf.pop('prime')]
            if len(prime_conf) != self.levels:
                raise ValueError(f"prime list length {len(prime_conf)} does not match number of resolutions {self.levels}")
        elif 'prime' in base_conf:
            base_conf['prime'] = int(base_conf['prime'])
        if 'bucket_size' in base_conf and isinstance(base_conf['bucket_size'], (list, tuple)):
            bucket_conf = [int(v) for v in base_conf.pop('bucket_size')]
            if len(bucket_conf) != self.levels:
                raise ValueError(f"bucket_size list length {len(bucket_conf)} does not match number of resolutions {self.levels}")
        elif 'bucket_size' in base_conf:
            base_conf['bucket_size'] = int(base_conf['bucket_size'])

        for idx, res in enumerate(resolutions):
            gridconf = dict(base_conf)
            gridconf['emb_dim'] = emb_dim_per_level
            if prime_conf is not None:
                gridconf['prime'] = int(prime_conf[idx])
            if bucket_conf is not None:
                gridconf['bucket_size'] = int(bucket_conf[idx])
            if 'prime' in gridconf and 'bucket_size' in gridconf:
                if gridconf['prime'] <= gridconf['bucket_size']:
                    raise ValueError(f"prime must exceed bucket_size for level {idx}: prime={gridconf['prime']}, bucket_size={gridconf['bucket_size']}")
            self.grids.append(CDFGridLevel(res, gridconf, render_mode=render_mode))
            
        if combine_mode == 'concat':
            in_dim = self.levels * emb_dim_per_level
            self.level_logits = None
        elif combine_mode == 'weighted':
            in_dim = emb_dim_per_level
            self.level_logits = nn.Parameter(torch.zeros(self.levels))
        else:
            raise ValueError(f"Unsupported combine_mode: {combine_mode}")
        self.final_mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Mish(),
            nn.Linear(in_dim, final_out_dim)
        )

    def forward(self, pts_3d):
        # pts_3d: [N,3]
        emb_list = []
        for grid in self.grids:
            e = grid(pts_3d)  # [N, emb_dim_per_level]
            emb_list.append(e)
        if self.combine_mode == 'concat':
            combined = torch.cat(emb_list, dim=-1)  # [N, L*d]
        else:  # weighted
            weights = torch.softmax(self.level_logits, dim=0)
            combined = torch.zeros_like(emb_list[0])
            for w, e in zip(weights, emb_list):
                combined = combined + w * e
        if self.combine_mode == "weighted":
            combined = self.final_mlp(combined)
        return combined

class CDFGridLevel(nn.Module):
    """
    Single-level
    """
    def __init__(self, resolution, vertex_dhe_conf, render_mode=False):
        super().__init__()
        self.res = resolution
        self.unique_mode = True
        self.render_mode = render_mode
        self.cached_pts = None     
        self.cached_scaled = None  
        self.cached_idx_floor = None
        self.cached_frac = None
        self.cached_corners = None
        self.cached_unique_coords = None
        self.cached_inv = None
        self.vertex_dhe = HashFeatureDecoder(**vertex_dhe_conf, render_mode=render_mode)
    
    def forward(self, pts_3d):
        device = pts_3d.device
        if self.render_mode and self.cached_pts is not None:
            if torch.equal(pts_3d, self.cached_pts):
                corner_emb = self.cached_corner_emb
                out_emb = self.cached_out_emb
                return out_emb

        N = pts_3d.shape[0]
        res = torch.tensor(self.res, device=device, dtype=torch.float)
        scaled = pts_3d * res
        idx_floor = torch.floor(scaled).long()   # [N,3]
        frac = scaled - idx_floor.float()        # [N,3]


        corners = []
        for dx in [0,1]:
            for dy in [0,1]:
                for dz in [0,1]:
                    off = torch.tensor([dx, dy, dz], device=device).unsqueeze(0)
                    idx_c = idx_floor + off
                    # idx_c = idx_c.float() / res
                    corners.append(idx_c)
        corners = torch.stack(corners, dim=1)  # [N,8,3]

        c_reshape = corners.view(-1, 3).float()
        if self.unique_mode:
            if self.render_mode:
                if self.cached_unique_coords is None or self.cached_inv is None:
                    # Compute unique coordinates once and reuse them while rendering.
                    unique_coords, inv = torch.unique(c_reshape, dim=0, return_inverse=True)
                    self.cached_unique_coords = unique_coords
                    self.cached_inv = inv
                else:
                    unique_coords = self.cached_unique_coords
                    inv = self.cached_inv
                unique_emb = self.vertex_dhe(unique_coords)  # [U, emb_dim]
                corner_emb = unique_emb[inv]  # [N*8, emb_dim]
            else:
                unique_coords, inv = torch.unique(c_reshape, dim=0, return_inverse=True)
                unique_emb = self.vertex_dhe(unique_coords)
                corner_emb = unique_emb[inv]
        else:
            corner_emb = self.vertex_dhe(c_reshape)  # [N*8, emb_dim]
        emb_dim = corner_emb.shape[-1]
        corner_emb = corner_emb.view(N, 8, emb_dim)

        mask = torch.tensor([[0,0,0],
                             [0,0,1],
                             [0,1,0],
                             [0,1,1],
                             [1,0,0],
                             [1,0,1],
                             [1,1,0],
                             [1,1,1]], device=device, dtype=torch.float)
        frac_exp = frac.unsqueeze(1)
        weights = mask * frac_exp + (1 - mask) * (1 - frac_exp)
        weights = torch.prod(weights, dim=2)  # [N,8]

        out_emb = torch.sum(corner_emb * weights.unsqueeze(-1), dim=1)  # [N, emb_dim]

        if self.render_mode:
            self.cached_pts = pts_3d.clone()
            self.cached_scaled = scaled.clone()
            self.cached_idx_floor = idx_floor.clone()
            self.cached_frac = frac.clone()
            self.cached_corners = corners.clone()
            self.cached_corner_emb = corner_emb.clone()
            self.cached_out_emb = out_emb.clone()
        
        return out_emb


class HashFeatureDecoder(nn.Module):
    """
    universal hash -> MLP
    """
    def __init__(
        self,
        num_hash_functions=128,
        emb_dim=16,
        prime=999983,
        bucket_size=1e5,
        hidden_dim=64,
        num_hidden_layers=2,
        render_mode=False,
    ):
        super().__init__()
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be >= 1")
        total_layers = num_hidden_layers + 1
        self.num_hash_functions = num_hash_functions
        self.emb_dim = emb_dim
        self.prime = int(prime)
        self.bucket_size = int(bucket_size)
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers

        self.register_buffer('A', torch.randint(1, self.prime, (num_hash_functions,3), dtype=torch.long))
        self.register_buffer('b', torch.randint(0, self.prime, (num_hash_functions,), dtype=torch.long))

        layers = []
        layers.append(nn.LayerNorm(num_hash_functions))
        layers.append(nn.Linear(num_hash_functions, hidden_dim))
        layers.append(nn.Mish())
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Mish())
        layers.append(nn.Linear(hidden_dim, emb_dim))
        self.mlp = nn.Sequential(*layers)

    def _dense_hash_enc(self, coords_3d):
        dot = coords_3d @ self.A.float().T
        h = (dot+self.b)%self.prime
        h = h%self.bucket_size
        enc = (h-1)/(self.bucket_size-1)
        enc = enc*2 -1
        return enc
    
    def forward(self, coords_3d):
        enc = self._dense_hash_enc(coords_3d)
        out = self.mlp(enc)
        return out


class ComputationalDeformationField(nn.Module):
    def __init__(self, voxel_conf=None, time_conf=None, final_out_dim=32, render_mode=False, use_film=None):
        super().__init__()
        if voxel_conf is None and time_conf is None:
            voxel_conf = {
                'resolutions': [(32,32,32), (64,64,64), (128,128,128)],
                'emb_dim_per_level': 16,  
                'final_out_dim': 32,     
                'vertex_dhe_conf': {
                    'num_hash_functions': 128,  
                    'emb_dim': 16,              
                    'prime': 200003,
                    'bucket_size': 50000,
                    'hidden_dim': 64
                },
                'combine_mode': 'concat'
            }
            time_conf = {
                'freq_L': 8,
                'hidden': 64,
                'out_dim': 32
            }
        film_trunk_width = voxel_conf.get('film_trunk_width', None)
        config_use_film = voxel_conf.get('use_film', None)
        if use_film is None:
            if config_use_film is not None:
                use_film = bool(config_use_film)
            else:
                use_film = film_trunk_width is not None

        voxel_kwargs = {
            key: voxel_conf[key]
            for key in ('resolutions', 'emb_dim_per_level', 'final_out_dim', 'vertex_dhe_conf', 'combine_mode')
            if key in voxel_conf
        }
        self.voxel_3d = MultiResolutionVoxelHashField(**voxel_kwargs, render_mode=render_mode)
        self.time_mlp = TemporalFeatureEncoder(**time_conf)
        self.render_mode = render_mode
        self.use_film = use_film

        if voxel_conf.get('combine_mode', 'concat') == 'concat':
            self.voxel_output_dim = len(voxel_conf['resolutions']) * voxel_conf['emb_dim_per_level']
        else:
            self.voxel_output_dim = voxel_conf.get('final_out_dim', final_out_dim)
        self.time_output_dim = time_conf['out_dim']

        if self.use_film:
            self.trunk_width = film_trunk_width or (self.voxel_output_dim + self.time_output_dim)

            self.spatial_to_trunk = nn.Linear(self.voxel_output_dim, self.trunk_width)
            self.film_generator = nn.Linear(self.time_output_dim, self.trunk_width * 2)
            with torch.no_grad():
                self.film_generator.weight.zero_()
                self.film_generator.bias[:self.trunk_width].fill_(1.0)
                self.film_generator.bias[self.trunk_width:].zero_()
            self.output_dim = self.trunk_width
        else:
            self.output_dim = self.voxel_output_dim + self.time_output_dim

        # combine => final
        input_dim = self.output_dim
        self.final_mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, final_out_dim)
        )

        self.cached_xyz = None
        self.cached_emb_xyz = None
        self.cached_hidden_voxel = None
        self.aabb = None

    def forward(self, xyzt):
        xyz = xyzt[:, :3]
        t = xyzt[:, 3]
        
        if self.render_mode and self.cached_emb_xyz is not None:
            emb_xyz = self.cached_emb_xyz
        else:
            emb_xyz = self.voxel_3d(xyz)  # [N, voxel_out_dim]
            if self.render_mode:
                self.cached_xyz = xyz.clone()
                self.cached_emb_xyz = emb_xyz.clone()
                self.cached_hidden_voxel = None
            else:
                self.cached_xyz = None
                self.cached_emb_xyz = None
                self.cached_hidden_voxel = None
        
        emb_t = self.time_mlp(t)  # [N, time_out_dim]
        if self.use_film:
            if self.render_mode and self.cached_hidden_voxel is not None:
                hidden_voxel = self.cached_hidden_voxel
            else:
                hidden_voxel = self.spatial_to_trunk(emb_xyz)
                if self.render_mode:
                    self.cached_hidden_voxel = hidden_voxel.clone()
            film_params = self.film_generator(emb_t)
            gamma, beta = film_params.chunk(2, dim=-1)
            modulated = hidden_voxel * gamma + beta
            return modulated
        else:
            cat = torch.cat([emb_xyz, emb_t], dim=-1)
            return cat
    
    def set_aabb(self, xyz_max, xyz_min):
        xyz_max = torch.tensor(xyz_max, dtype=torch.float32)
        xyz_min = torch.tensor(xyz_min, dtype=torch.float32)
        aabb_tensor = torch.stack([xyz_max, xyz_min], dim=0)
        self.aabb = nn.Parameter(aabb_tensor, requires_grad=False)
        
    @property
    def get_aabb(self):
        return self.aabb[0], self.aabb[1]


# Backward-compatible aliases
TimeMLP = TemporalFeatureEncoder
MultiResolutionVoxelHashDHE_3D = MultiResolutionVoxelHashField
VoxelGridHashDHE_3D = CDFGridLevel
VertexHashMLP3D = HashFeatureDecoder
MultiResVoxel3D_TimeModel = ComputationalDeformationField
