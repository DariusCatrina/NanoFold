import math

import torch
import torch.nn as nn

from openfold.utils.rigid_utils import Rigid


class InvariantPointAttention(nn.Module):
    def __init__(self, c_hidden, c_s, c_z, n_heads, n_query_points, n_point_values, use_bias=False):
        super().__init__()
        self.c_hidden= c_hidden
        self.n_heads = n_heads
        self.n_query_points = n_query_points
        self.n_point_values = n_point_values


        self.qkv_linear = nn.Linear(c_s, c_hidden*n_heads*3, bias=use_bias)
        self.qk_points_linear = nn.Linear(c_s, n_heads*n_query_points*3*2)
        self.v_points_linear =  nn.Linear(c_s, n_heads*n_point_values*3)
        self.b_linear = nn.Linear(c_z, n_heads)

        self.w_C = math.sqrt(2/(9*n_query_points))
        self.w_L = math.sqrt(1/3)

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()
        self.gamma = nn.Parameter(torch.full((n_heads,), 0.541324854612918))

        concat_dim = (
            (self.n_heads * c_z) +                           # output_tilde
            (self.n_heads * self.c_hidden) +                 # output
            (self.n_heads * self.n_point_values * 3) +       # o_vector points
            (self.n_heads * self.n_point_values)             # o_vector norms (no 3D dim)
        )

        self.final_lin = nn.Linear(concat_dim, c_s)

    def forward(self, s, z, T : Rigid, mask):
        batch_size, num_residues, c_s = s.size()
        batch_size, num_residues, num_residues, c_z = z.size()

        qkv = self.qkv_linear(s) #(batch_size, num_residues, c_hidden * n_heads * 3)
        qkv = qkv.view(batch_size, num_residues, 3, self.n_heads, self.c_hidden) #(batch_size, num_residues, 3, n_heads, hidden)
        q, k, v = torch.unbind(qkv, dim=2) # (batch_size, num_residues, n_heads, hidden)

        term1 = torch.einsum('b i h c, b j h c -> b h i j', q, k) / math.sqrt(self.c_hidden) # (batch_size, n_heads, num_residues, num_residues)

        b = self.b_linear(z) #(batch_size, num_residues, num_residues, n_heads)
        term2 = b.permute(0, 3, 1, 2)

        qk_points = self.qk_points_linear(s) #(batch_size, num_residues, n_heads * n_query_points *3 *2)
        qk_points = qk_points.view(batch_size, num_residues, 2, self.n_heads, self.n_query_points, 3) # (batch_size, num_residues, 2, n_heads, n_query_points, 3)
        q_points, k_points = torch.unbind(qk_points, dim=2) # (batch_size, num_residues, n_heads, n_query_points, 3)

        rigids_expanded = T[:, :, None, None] # (B, N) -> (B, N, 1, 1)
        q_global = rigids_expanded.apply(q_points)
        k_global = rigids_expanded.apply(k_points)
        
        # q becomes: (B, N, 1, n_heads, n_query_points, 3) -> Represents Residue i
        # k becomes: (B, 1, N, n_heads, n_query_points, 3) -> Represents Residue j
        q_expanded = q_global[:, :, None, :, :, :]
        k_expanded = k_global[:, None, :, :, :, :]

        diff = (q_expanded - k_expanded)**2
        diff = diff.sum(dim=(-2,-1)).permute(0, 3, 1, 2) #(B, N, N, n_heads) -> (B,n_heads, N, N)
        gamma = self.softplus(self.gamma).view(1, -1, 1, 1) #(1, n_heads, 1, 1)
        term3 = ((gamma* self.w_C)/ 2)* diff 


        attention_logits = self.w_L * (term1 + term2 - term3) # (B,n_heads, N, N)
        # mask: B,N where 1 real 0 padding
        if mask != None:
            mask_expanded = mask.view(batch_size, 1, 1, num_residues) # B,1,1,N, masking the keys/columns
            attention_logits = attention_logits.masked_fill(~mask_expanded, -1e9)

        attention_weights = self.softmax(attention_logits) # (B,n_heads, N, N)


        output_tilde = torch.einsum('b h i j, b i j c -> b i h c', attention_weights, z) #  (B,N, n_heads, c_z)
        output = torch.einsum('b h i j, b j h c  -> b i h c', attention_weights, v) #  (B,N, n_heads, c_z)
        
        v_points = self.v_points_linear(s)
        v_points = v_points.view(batch_size, num_residues, self.n_heads, self.n_point_values, 3) # B, N, n_heads, n_point_values,3
        v_global = rigids_expanded.apply(v_points)
        o_global = torch.einsum('b h i j, b j h v x -> b i h v x',attention_weights, v_global)# (B, N, n_heads, n_point_values, 3)
        o_vector = rigids_expanded.invert_apply(o_global)
        o_vector_norm = torch.sqrt(torch.sum(o_vector**2, dim=-1) + 1e-8)

        out_tilde_flat = output_tilde.flatten(start_dim=2)  # Shape: (B, N, n_heads * c_z)
        out_flat = output.flatten(start_dim=2)              # Shape: (B, N, n_heads * c_hidden)
        o_vector_flat = o_vector.flatten(start_dim=2)         # Shape: (B, N, n_heads * n_point_values * 3)
        o_vector_norm_flat = o_vector_norm.flatten(start_dim=2) # Shape: (B, N, n_heads * n_point_values)

        # 3. Concatenate along the final feature dimension
        concat_features = torch.cat([
            out_tilde_flat, 
            out_flat, 
            o_vector_flat, 
            o_vector_norm_flat
        ], dim=-1)

        return self.final_lin(concat_features)



class TransitionModule(nn.Module):
    def __init__(self, c_s):
        super().__init__()
        self.module = nn.Sequential(
            nn.Linear(c_s, c_s),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
            nn.ReLU(),
        )
        self.ln = nn.LayerNorm(c_s)
        self.dropout = nn.Dropout(0.1)


    def forward(self, s):
        s = s + self.module(s)
        s = self.ln(self.dropout(s))

        return s


class TorsionAngleModule(nn.Module):
    def __init__(self, c_s):
        super().__init__()
        self.s_linear = nn.Linear(c_s, c_s)
        self.s_init_linear = nn.Linear(c_s, c_s)

        self.module_1 = nn.Sequential(
            nn.Linear(c_s, c_s),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
            nn.ReLU(),
        )
        self.module_2 = nn.Sequential(
            nn.Linear(c_s, c_s),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
            nn.ReLU(),
        )
        self.final_lin = nn.Linear(c_s, 2)
        self.final_relu = nn.ReLU()
        self.eps=1e-8

    def forward(self, s, s_initial):
        a = self.s_linear(s) + self.s_init_linear(s_initial)
        a += self.module_1(a)
        a += self.module_2(a)

        alpha = self.final_lin(self.final_relu(a))

        norm_denom = torch.sqrt(
            torch.clamp(
                torch.sum(alpha**2, dim=-1, keepdim=True),
                min=self.eps,
            )
        )
        alpha = alpha / norm_denom

        return alpha

class ZeroLinear(nn.Linear):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super(ZeroLinear, self).__init__(in_dim, out_dim, bias=bias)
        with torch.no_grad():
            self.weight.fill_(0.0)
            if bias: 
                self.bias.fill_(0.0)

def get_transformer_layer(d_model, n_heads, n_layers):
    tfmr_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model,
                batch_first=True,
                dropout=0.0,
                norm_first=True,
            )

    return torch.nn.TransformerEncoder(tfmr_layer, n_layers)



class SinusoidalEmbedding(nn.Module):
    """Used for both Time (t) and Sequence Indices"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # x shape: (B) for time, or (B, N) for absolute, or (B, N, N) for relative
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x.unsqueeze(-1) * emb.view(1, -1)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)




class AdaLN(nn.Module):
    def __init__(self, c_s, c_time):
        super().__init__()
        # Disable learnable gamma/beta because we will predict them from time
        self.norm = nn.LayerNorm(c_s, elementwise_affine=False)
        self.silu = nn.SiLU()
        
        # Maps time to [scale, shift]
        self.linear = nn.Linear(c_time, c_s * 2)
        

        with torch.no_grad():
            nn.init.zeros_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)

    def forward(self, x, t_emb):
        # x is (B, N, c_s)
        # t_emb is (B, c_time)
        
        # Predict scale and shift
        emb = self.linear(self.silu(t_emb))
        scale, shift = emb.chunk(2, dim=-1) # Each is (B, c_s)
        
        # Add sequence dimension for broadcasting -> (B, 1, c_s)
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)
        
        # Apply normalization: y = norm(x) * (1 + scale) + shift
        return self.norm(x) * (1 + scale) + shift