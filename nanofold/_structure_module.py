import torch
import math
import torch.nn as nn


from openfold.utils.rigid_utils import Rigid

from nanofold._layers import InvariantPointAttention, ZeroLinear, TransitionModule, get_transformer_layer, TorsionAngleModule, SinusoidalEmbedding, AdaLN

class StructureNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_layers = config.n_layers
        self.s_norm = nn.LayerNorm(config.c_s)
        self.z_norm = nn.LayerNorm(config.c_z)
        self.s_init_lin = nn.Linear(config.c_s, config.c_s)
        
        # Need a time embedder here for AdaLN to use
        # Assuming you allocate config.c_time (e.g. 256) for time embedding
        self.time_embedder = SinusoidalEmbedding(dim=config.c_time) 

        self.trunk = nn.ModuleDict()
        for l in range(self.n_layers):
            self.trunk[f"{l}_ipa"] = InvariantPointAttention(config.c_hidden, config.c_s, config.c_z, config.n_heads, config.n_query_points, config.n_point_values, False)
            
            self.trunk[f"{l}_adaln"] = AdaLN(config.c_s, config.c_time) 
            self.trunk[f"{l}_ln"] = nn.LayerNorm(config.c_s)
            self.trunk[f"{l}_ln_drop"] = nn.Dropout(0.1)
            self.trunk[f"{l}_skip_con"] = ZeroLinear(config.c_s, config.c_s_skip)
            self.trunk[f"{l}_trmf"] = get_transformer_layer(config.c_s + config.c_s_skip, config.n_heads_attn, 1)
            self.trunk[f"{l}_trmf_proj"] = ZeroLinear(config.c_s + config.c_s_skip, config.c_s)
            self.trunk[f"{l}_transition"] = TransitionModule(config.c_s)
            self.trunk[f"{l}_backbone"] = nn.Linear(config.c_s, 6)

        self.torsion_module = TorsionAngleModule(config.c_s)

    def forward(self, s_embed, z_embed, T: Rigid, mask, time):
        node_mask = mask.bool()
        
        # Embed time for the AdaLN layers
        t_emb = self.time_embedder(time) 

        s_initial = self.s_init_lin(self.s_norm(s_embed))
        z_initial = self.z_norm(z_embed)

        s = s_initial
        z = z_initial

        for l in range(self.n_layers):
            s = self.trunk[f"{l}_adaln"](s, t_emb) 
            
            s_update = self.trunk[f"{l}_ipa"](s, z, T, node_mask)
            s_update *= node_mask[..., None]

            s = self.trunk[f"{l}_ln"](self.trunk[f"{l}_ln_drop"](s + s_update))

            res_s =  self.trunk[f"{l}_skip_con"](s_initial)
            res_embed = torch.cat([s, res_s], dim=-1)

            trmf_out = self.trunk[f"{l}_trmf"](res_embed, src_key_padding_mask=~node_mask)
            
            s = s + self.trunk[f"{l}_trmf_proj"](trmf_out)
            
            s = self.trunk[f"{l}_transition"](s)
            s *= node_mask[..., None]

            ds = self.trunk[f"{l}_backbone"](s)
            ds *= node_mask[..., None]
            T = T.compose_q_update_vec(ds)

        psi = self.torsion_module(s, s_initial)

        return {
            "frames": T,
            "psi": psi,
            "velocities": ds
        }



class SequenceEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        esm_dim = config.esm_dim
        self.c_s = config.c_s
        self.c_z = config.c_z
        
        # We will use c_s // 2 for the sinusoidal dims to keep sizes manageable
        self.idx_dim = self.c_s // 2  

        self.sinusoid_embedder = SinusoidalEmbedding(dim=self.idx_dim)
        
        # 1. Single Node Embedder (ESM + Time + Absolute Idx)
        node_in_dim = esm_dim + self.idx_dim + self.idx_dim
        self.node_embedder = nn.Sequential(
            nn.Linear(node_in_dim, self.c_s),
            nn.ReLU(),
            nn.Linear(self.c_s, self.c_s),
            nn.LayerNorm(self.c_s)
        )

        # 2. Pair Edge Embedder (Cross Concat Time + Relative Idx + Spatial Dist)
        # Cross concat takes 1D Time (idx_dim) -> yields 2 * idx_dim
        edge_in_dim = (self.idx_dim * 2) + self.idx_dim + 16
        self.edge_embedder = nn.Sequential(
            nn.Linear(edge_in_dim, self.c_z),
            nn.ReLU(),
            nn.Linear(self.c_z, self.c_z),
            nn.LayerNorm(self.c_z)
        )
        
        # Small projection for the current 3D geometry
        self.spatial_proj = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU()
        )

    def _cross_concat(self, feats_1d, num_res):
        # feats_1d: (B, N, D)
        # Returns: (B, N, N, 2D)
        feats_i = feats_1d.unsqueeze(2).expand(-1, -1, num_res, -1)
        feats_j = feats_1d.unsqueeze(1).expand(-1, num_res, -1, -1)
        return torch.cat([feats_i, feats_j], dim=-1)

    def forward(self, esm_repr, t, current_ca_pos):
        # current_ca_pos: (B, N, 3) from the Rigid frames
        B, N, _ = esm_repr.shape
        device = esm_repr.device

        # --- 1. Index & Time Embeddings ---
        seq_idx = torch.arange(N, device=device).float().unsqueeze(0).expand(B, -1) # (B, N)
        rel_seq_offset = seq_idx.unsqueeze(-1) - seq_idx.unsqueeze(1) # (B, N, N)
        
        t_embed = self.sinusoid_embedder(t) # (B, idx_dim)
        t_embed_1d = t_embed.unsqueeze(1).expand(-1, N, -1) # (B, N, idx_dim)
        
        abs_idx_embed = self.sinusoid_embedder(seq_idx) # (B, N, idx_dim)
        rel_idx_embed = self.sinusoid_embedder(rel_seq_offset) # (B, N, N, idx_dim)

        # --- 2. Node Representation (s) ---
        node_feats = torch.cat([esm_repr, t_embed_1d, abs_idx_embed], dim=-1)
        s = self.node_embedder(node_feats.float())

        # --- 3. Pair Representation (z) ---
        # Cross-concat time so pair features know what time it is
        pair_time = self._cross_concat(t_embed_1d, N) # (B, N, N, idx_dim * 2)

        # Current 3D Spatial geometry (X_t)
        dist_sq = torch.sum((current_ca_pos.unsqueeze(2) - current_ca_pos.unsqueeze(1)) ** 2, dim=-1)
        dist = torch.sqrt(dist_sq + 1e-8).unsqueeze(-1) 
        spatial_feats = self.spatial_proj(dist) # (B, N, N, 16)

        # Concatenate everything for the pair track
        pair_feats = torch.cat([pair_time, rel_idx_embed, spatial_feats], dim=-1)
        z = self.edge_embedder(pair_feats.float())

        return s, z