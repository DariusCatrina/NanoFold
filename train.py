import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import wandb # <--- Imported wandb

# Import your architecture components
from nanofold import NanoFold, NanoFoldConfig
from dataset import AlphaFoldPeptideDataset, collate_peptides

# Import SE(3) Flow Matcher and Rigid utils from the foldflow repo
from foldflow.models.se3_fm import SE3FlowMatcher 
from openfold.utils.rigid_utils import Rigid, Rotation


import torch
import torch.nn.functional as F

class SimpleSE3FlowMatcher:
    @staticmethod
    def sample_noise(B, N, device):
        # 1. Translation Noise ~ N(0, 1)
        # You can scale this by a factor (e.g. 10.0) if you want wider starting noise
        T_0 = torch.randn(B, N, 3, device=device)
        
        # 2. Rotation Noise ~ Haar Uniform via QR decomposition
        H = torch.randn(B, N, 3, 3, device=device)
        Q, R = torch.linalg.qr(H)
        
        # Enforce positive diagonals on R to make Q Haar distributed
        D = torch.diagonal(R, dim1=-2, dim2=-1).sign()
        Q = Q * D.unsqueeze(-2)
        
        # Ensure det(Q) == 1 (Valid Orthogonal Rotation Matrix)
        det = torch.linalg.det(Q)
        mask = (det < 0)
        Q[mask, :, 2] *= -1.0
        
        return T_0, Q

    @staticmethod
    def interpolate_translation(T_0, T_1, t):
        # Straight-line Euclidean interpolation
        return (1 - t) * T_0 + t * T_1

    @staticmethod
    def interpolate_rotation(R_0, R_1, t):
        # Geodesic interpolation on SO(3) using Rodrigues' formula
        R_rel = R_1 @ R_0.transpose(-1, -2)
        trace = R_rel.diagonal(dim1=-2, dim2=-1).sum(-1)
        
        # Clamp to avoid NaN issues with arccos
        cos_theta = torch.clamp((trace - 1.0) / 2.0, -1.0 + 1e-6, 1.0 - 1e-6)
        theta = torch.acos(cos_theta).unsqueeze(-1).unsqueeze(-1) 
        
        sin_theta = torch.sin(theta)
        
        # Safe division masking for very small angles
        safe_sin = torch.where(sin_theta < 1e-4, torch.ones_like(sin_theta), sin_theta)
        K = (R_rel - R_rel.transpose(-1, -2)) / (2.0 * safe_sin)
        K = torch.where(sin_theta < 1e-4, torch.zeros_like(K), K)
        
        I = torch.eye(3, device=R_0.device).view(1, 1, 3, 3)
        
        R_rel_t = I + torch.sin(t * theta) * K + (1.0 - torch.cos(t * theta)) * (K @ K)
        
        return R_rel_t @ R_0

def train_nanofold():
    # ---------------------------------------------------------
    # 0. Initialize W&B
    # ---------------------------------------------------------
    wandb.init(
        project="nanofold-flow-matching", 
        name="se3-flow-self-cond-fape", # Give your run a descriptive name
        config={
            "learning_rate": 1e-4,
            "batch_size": 32,
            "epochs": 100,
            "max_len": 128,
            "dataset": "small_peptides_pdb",
            "architecture": "NanoFold",
            "inference_scaling": 1.0
        },
        entity="darius-catrina"
    )

    # ---------------------------------------------------------
    # 1. Setup Environment & Models
    # ---------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = NanoFoldConfig()
    model = NanoFold(config).to(device)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=wandb.config.learning_rate)
    
    se3_conf = OmegaConf.create({
        "flow_trans": True, "flow_rot": True, "ot_fn": "exact", "reg": 0.05,
        "ot_plan": False, "stochastic_paths": False,
        "r3": {"min_b": 0.01, "max_b": 1.0, "coordinate_scaling": 0.1, "g": 0.1, "min_sigma": 0.01, "max_sigma": 1.5, "inference_scaling": 1.0},
        "so3": {"min_b": 0.01, "max_b": 1.0, "g": 0.1, "min_sigma": 0.01, "max_sigma": 1.5, "inference_scaling": 1.0}
    })
    
    se3_matcher = SE3FlowMatcher(se3_conf)
    
    # ---------------------------------------------------------
    # 2. Dataset & DataLoader
    # ---------------------------------------------------------
    PDB_DIR = "/work/dgc26/small_peptides_pdb"
    dataset = AlphaFoldPeptideDataset(pdb_dir=PDB_DIR, max_len=wandb.config.max_len)
    
    dataloader = DataLoader(
        dataset, batch_size=wandb.config.batch_size, shuffle=True, 
        collate_fn=collate_peptides, num_workers=4, pin_memory=True
    )
    
    model.train()
    
    # ---------------------------------------------------------
    # 3. Training Loop
    # ---------------------------------------------------------
    global_step = 0

    for epoch in range(wandb.config.epochs):
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            raw_sequences = batch["sequences"]
            coords = batch["coords"].to(device) 
            
            B, max_len = coords.shape[:2]
            seq_lengths = torch.tensor([len(s) for s in raw_sequences], device=device)
            mask = torch.arange(max_len, device=device).unsqueeze(0) < seq_lengths.unsqueeze(1)
            mask_bool = mask.bool()
            
            # ---------------------------------------------------------
            # Center the Protein at the Origin to fix Translation spikes
            # ---------------------------------------------------------
            ca_coords = coords[:, :, 1, :] 
            valid_ca = ca_coords * mask[..., None] 
            
            center = valid_ca.sum(dim=1) / seq_lengths.unsqueeze(-1) 
            coords = coords - center.view(B, 1, 1, 3)
            
            native_frames = Rigid.from_3_points(
                p_neg_x_axis=coords[:, :, 2, :], origin=coords[:, :, 1, :], p_xy_plane=coords[:, :, 0, :]    
            )
            
            # ---------------------------------------------------------
            # Step A: Sample Time & Pure Noise (X_0)
            # ---------------------------------------------------------
            t = torch.rand(B, device=device) * (1.0 - 2e-3) + 1e-3
            
            n_total = B * max_len
            noise_dict = se3_matcher.sample_ref(n_samples=n_total)
            noise_rigids_flat = noise_dict["rigids_t"]
            
            noise_rot_mats = noise_rigids_flat.get_rots().get_rot_mats().view(B, max_len, 3, 3).to(device)
            noise_trans = noise_rigids_flat.get_trans().view(B, max_len, 3).to(device)

            # ---------------------------------------------------------
            # Step B: Interpolate True Noisy Frames (X_t)
            # ---------------------------------------------------------
            t_trans = t.view(B, 1, 1)
            t_rot = t.view(B, 1, 1, 1)


            trans_t = SimpleSE3FlowMatcher.interpolate_translation(
                noise_trans, native_frames.get_trans(), t_trans
            )
            rot_t = SimpleSE3FlowMatcher.interpolate_rotation(
                noise_rot_mats, native_frames.get_rots().get_rot_mats(), t_rot
            )
            
            noisy_frames = Rigid(rots=Rotation(rot_mats=rot_t), trans=trans_t)
            
            # ---------------------------------------------------------
            # Step C: Model Forward Pass -> Predicts \hat{X}_1
            # ---------------------------------------------------------
            if torch.rand(1).item() < 0.5:
                with torch.no_grad():
                    guess_out = model(raw_sequences, t, noisy_frames, self_cond_frames=None)
                    # Manually detach both components and clone to break the reference
                    g_rot = guess_out["frames"].get_rots().get_rot_mats().detach().clone()
                    g_trans = guess_out["frames"].get_trans().detach().clone()
                    guess_frames = Rigid(rots=Rotation(rot_mats=g_rot), trans=g_trans)
                
                # Explicitly clear the output dictionary to free the GPU
                del guess_out
                    
            else:
                # 2. Null condition (first step of inference always starts here)
                guess_frames = None

            # 3. Main Forward Pass (Gradient tracked)
            # The model now receives its own previous guess to refine!
            out = model(raw_sequences, t, noisy_frames, self_cond_frames=guess_frames)
            pred_native_frames = out["frames"]

            # ---------------------------------------------------------
            # Step D: Calculate TRUE & PREDICTED Vector Fields
            # ---------------------------------------------------------
            true_trans = native_frames.get_trans()
            pred_trans = pred_native_frames.get_trans()
            
            true_rot = native_frames.get_rots().get_rot_mats()
            pred_rot = pred_native_frames.get_rots().get_rot_mats()

            # 1. Global MSE Losses
            loss_trans = F.mse_loss(pred_trans[mask_bool], true_trans[mask_bool])
            loss_rot = F.mse_loss(pred_rot[mask_bool], true_rot[mask_bool])
            
            # 2. Pairwise Distance Loss (Internal Geometry / "Lightweight FAPE")
            # Calculate the distance between every CA atom and every other CA atom
            # (B, L, 1, 3) - (B, 1, L, 3) -> (B, L, L, 3) -> norm -> (B, L, L)
            pred_dists = torch.norm(pred_trans.unsqueeze(2) - pred_trans.unsqueeze(1), dim=-1)
            true_dists = torch.norm(true_trans.unsqueeze(2) - true_trans.unsqueeze(1), dim=-1)
            
            # Create a 2D mask for valid interacting pairs
            mask_2d = mask.unsqueeze(1) & mask.unsqueeze(2) # (B, L, L)
            
            # We use Smooth L1 (Huber) loss here instead of MSE. 
            # This prevents massive outlier distances at the start of training 
            # from completely exploding the gradients.
            loss_dist = F.smooth_l1_loss(pred_dists[mask_2d], true_dists[mask_2d])

            # ---------------------------------------------------------
            # 3. Loss Balancing
            # ---------------------------------------------------------
            # We explicitly down-weight translation so it doesn't bully rotation,
            # and give a healthy weight to the internal geometry (distance loss).
            
            W_TRANS = 0.05  # Scales Angstrom MSE down to ~1.0
            W_ROT = 1.0     # Keep rotation weight as standard
            W_DIST = 0.5    # Strong weight for internal fold geometry

            loss = (W_TRANS * loss_trans) + (W_ROT * loss_rot) + (W_DIST * loss_dist)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if guess_frames is not None:
                del guess_frames
            
            # ---------------------------------------------------------
            # NEW: Log Metrics to WandB
            # ---------------------------------------------------------
            wandb.log({
                "train/total_loss": loss.item(),
                "train/trans_loss": loss_trans.item(),
                "train/rot_loss": loss_rot.item(),
                "train/dist_loss": loss_dist.item() ,
                "epoch": epoch,
            }, step=global_step)
            
            global_step += 1
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Total: {loss.item():.4f} | "
                      f"Trans: {loss_trans.item():.4f} | Rot: {loss_rot.item():.4f}")

    # Close the wandb run when finished
    wandb.finish()

if __name__ == "__main__":
    train_nanofold()