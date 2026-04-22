from dataclasses import dataclass


import esm
import torch
import torch.nn as nn

from nanofold._structure_module import StructureNetwork, SequenceEmbedder
from openfold.utils.rigid_utils import Rigid

@dataclass
class NanoFoldConfig:
    # --- ESM Sequence Embedder ---
    esm_dim: int = 1280         # ESM2-650M layer 33 output dim
    esm_layer: int = 33
    
    # --- Core Dimensions ---
    c_s: int = 384              # Single (node) representation dimension
    c_z: int = 128              # Pair (edge) representation dimension
    c_time: int = 256           # Continuous time (t) embedding dimension
    c_s_skip: int = 64          # Dimension for the initial sequence skip connection
    c_hidden:int = 256
    
    # --- Structure Module (Trunk) ---
    n_layers: int = 4           # Number of IPA + Transformer blocks (Use 12-24 for full scale, 4-8 for Toy/Nano)
    n_heads_attn:int = 4
    
    # --- Invariant Point Attention (IPA) ---
    c_hidden: int = 16          # Hidden channel dimension per head
    n_heads: int = 8           # Number of attention heads
    n_query_points: int = 4     # Number of query 3D points per head
    n_point_values: int = 8     # Number of value 3D points per head



class NanoFold(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.structure_module = StructureNetwork(config)
        self.sequence_embedder = SequenceEmbedder(config)

        self.esm_model, self.esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.esm_layer = config.esm_layer

        self.esm_model.eval() # Disable dropout
        for param in self.esm_model.parameters():
            param.requires_grad = False # Freeze weights
            
        self.batch_converter = self.esm_alphabet.get_batch_converter()
        self.cond_embedder = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, config.c_z))

    def get_esm_representations(self, raw_sequences, device):
        """
        Takes raw string sequences, handles tokenization, padding, 
        and extracts the exact (B, N, D) embeddings without gradients.
        """
        # 1. Prepare data for ESM batch converter
        # format: [(protein_id, sequence), ...]
        batch_data = [(str(i), seq) for i, seq in enumerate(raw_sequences)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(batch_data)
        batch_tokens = batch_tokens.to(device)

        # 2. Extract sequence lengths to build our boolean mask
        seq_lengths = torch.tensor([len(seq) for seq in raw_sequences], device=device)
        B = len(raw_sequences)
        max_len = seq_lengths.max().item()

        # Build the boolean node_mask (B, max_len)
        mask = torch.arange(max_len, device=device).unsqueeze(0) < seq_lengths.unsqueeze(1)

        # 3. Forward pass through ESM without tracking gradients!
        with torch.no_grad():
            results = self.esm_model(batch_tokens, repr_layers=[self.esm_layer], return_contacts=False)
            raw_esm_repr = results["representations"][self.esm_layer] # Shape: (B, max_len + 2, 1280)

        # 4. Crop <cls> and <eos> tokens
        # We create a clean tensor of exactly (B, max_len, 1280)
        esm_repr = torch.zeros(B, max_len, 1280, device=device)
        for i, length in enumerate(seq_lengths):
            # Tokens are [CLS, AA_1, ..., AA_L, EOS, PAD...]
            # We want indices 1 to length + 1
            esm_repr[i, :length] = raw_esm_repr[i, 1 : length + 1]

        return esm_repr, mask

    def forward(self, raw_sequences: list[str], t, noisy_rigids: Rigid, self_cond_frames=None):
        """
        Args:
            raw_sequences: List of amino acid strings e.g., ["MKTV...", "GATT..."]
            t: (B) The continuous time scalar between 0 and 1.
            noisy_rigids: Rigid (B, N) The current noisy 3D frames X_t.
        """
        device = t.device
        
        # 1. Run frozen ESM and dynamically generate the sequence mask
        # Ensures no gradients flow backward into ESM, saving memory!
        esm_repr, mask = self.get_esm_representations(raw_sequences, device)

        # 2. Embed inputs (Sequence + Time + Current Geometry)
        s, z = self.sequence_embedder(esm_repr, t, noisy_rigids.get_trans())
        
        if self_cond_frames is not None:
            # Extract the Alpha Carbon coordinates from the guess
            guess_coords = self_cond_frames.get_trans() # (B, L, 3)
            
            # Calculate pairwise Euclidean distances: (B, L, 1, 3) - (B, 1, L, 3) -> (B, L, L)
            dists = torch.norm(
                guess_coords.unsqueeze(2) - guess_coords.unsqueeze(1), 
                dim=-1, 
                keepdim=True
            ) # (B, L, L, 1)
            
            # Project distances to the pair dimension and add to the pair_rep
            cond_bias = self.cond_embedder(dists) # (B, L, L, pair_dim)
            z += cond_bias

        # 3. Predict the final folded structure
        out = self.structure_module(
            s_embed=s,
            z_embed=z,
            T=noisy_rigids, 
            mask=mask,
            time=t
        )

        return out