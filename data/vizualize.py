import io
import torch
import numpy as np

from openfold.utils.rigid_utils import Rigid
from openfold.np import protein, residue_constants

from openfold.utils.feats import (
    torsion_angles_to_frames,
    frames_and_literature_positions_to_atom14_pos,
    atom14_to_atom37
)
from openfold.data import data_transforms




def extract_representation_from_pdb(pdb_path):
    with open(pdb_path, 'r') as f:
        prot = protein.from_pdb_string(f.read())
        
    aatype = torch.tensor(prot.aatype, dtype=torch.long)
    atom37 = torch.tensor(prot.atom_positions, dtype=torch.float32)
    atom37_mask = torch.tensor(prot.atom_mask, dtype=torch.float32)
    residue_index = torch.tensor(prot.residue_index, dtype=torch.long)
    

    N_idx = residue_constants.atom_order['N']
    CA_idx = residue_constants.atom_order['CA']
    C_idx = residue_constants.atom_order['C']
    O_idx = residue_constants.atom_order['O']
    
    atom4 = atom37[:, [N_idx, CA_idx, C_idx, O_idx], :]
    ca_mask = atom37_mask[:, CA_idx]
    
    valid_ca = atom4[ca_mask == 1.0, 1, :]
    if valid_ca.shape[0] > 0:
        centroid = valid_ca.mean(dim=0)
        atom4 = atom4 - centroid
        atom37 = atom37 - centroid 


    batch = data_transforms.atom37_to_torsion_angles()({
        "aatype": aatype,
        "all_atom_positions": atom37,
        "all_atom_mask": atom37_mask,
    })
    torsion_angles = batch["torsion_angles_sin_cos"]

    # Use make_transform_from_reference for 100% native AF2 backbone extraction
    frames = Rigid.make_transform_from_reference(
        n_xyz=atom4[:, 0, :],
        ca_xyz=atom4[:, 1, :],
        c_xyz=atom4[:, 2, :]
    )
    # origin = atom4[:, 1, :]

    # # 2. Define the XY plane anchor as N
    # p_xy_plane = atom4[:, 0, :]

    # # 3. Create a phantom point exactly opposite to C, across the CA atom
    # # Equation: p_neg_x_axis = CA - (C - CA) = 2*CA - C
    # p_neg_x_axis = (2 * atom4[:, 1, :]) - atom4[:, 2, :]

    # # 4. Generate the frames
    # frames = Rigid.from_3_points(
    #     p_neg_x_axis=p_neg_x_axis, 
    #     origin=origin,       
    #     p_xy_plane=p_xy_plane,   
    #     eps=1e-8
    # )
    frames_7d = frames.to_tensor_7()

    return frames_7d, atom4, torsion_angles, aatype

def reconstruct_pdb_from_representation(frames_7d, torsion_angles, aatype, output_pdb_path):
    device = frames_7d.device
    dtype = frames_7d.dtype
    N = frames_7d.shape[0]

    # 1. Load idealized OpenFold constants
    default_frames = torch.tensor(residue_constants.restype_rigid_group_default_frame, dtype=dtype, device=device)
    group_idx = torch.tensor(residue_constants.restype_atom14_to_rigid_group, dtype=torch.long, device=device)
    atom14_mask = torch.tensor(residue_constants.restype_atom14_mask, dtype=dtype, device=device)
    lit_positions = torch.tensor(residue_constants.restype_atom14_rigid_group_positions, dtype=dtype, device=device)

    # 2. Rebuild structural frames
    frames = Rigid.from_tensor_7(frames_7d)
    all_frames_to_global = torsion_angles_to_frames(frames, torsion_angles, aatype, default_frames)

    # 3. Generate Atom14 positions
    atom14_pos = frames_and_literature_positions_to_atom14_pos(
        all_frames_to_global, aatype, default_frames, group_idx, atom14_mask, lit_positions
    )

    # 4. Map back to Atom37
    restype_atom37_to_atom14 = torch.tensor(residue_constants._make_restype_atom37_to_atom14(), dtype=torch.long, device=device)
    restype_atom37_mask = torch.tensor(residue_constants.restype_atom37_mask, dtype=dtype, device=device)
    
    minimal_batch = {
    "residx_atom37_to_atom14": restype_atom37_to_atom14[aatype],
    "atom37_atom_exists": restype_atom37_mask[aatype]
    }

    atom37_pos = atom14_to_atom37(atom14_pos, minimal_batch)
    atom37_mask = restype_atom37_mask[aatype]

    # 5. Export to PDB
    prot = protein.Protein(
        atom_positions=atom37_pos.detach().cpu().numpy(),
        aatype=aatype.detach().cpu().numpy(),
        atom_mask=atom37_mask.detach().cpu().numpy(),
        residue_index=np.arange(1, N + 1),
        b_factors=np.zeros_like(atom37_mask.cpu().numpy()),
        chain_index=np.zeros(N, dtype=np.int32)
    )

    pdb_str = protein.to_pdb(prot)
    with open(output_pdb_path, 'w') as f:
        f.write(pdb_str)

    return pdb_str

if __name__ == '__main__':
    frames_7d, atom4, all_torsions, aatype = extract_representation_from_pdb('./data/AF-A0A009HBH5-F1-model_v6.pdb')
    reconstruct_pdb_from_representation(frames_7d, all_torsions, aatype, output_pdb_path='./data/AF-A0A009HBH5-F1-model_v6_generated.pdb')
     
