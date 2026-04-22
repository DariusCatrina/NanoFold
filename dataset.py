import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from openfold.utils.rigid_utils import Rigid

# Standard 3-letter to 1-letter amino acid mapping
RESIDUE_MAP = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

class AlphaFoldPeptideDataset(Dataset):
    def __init__(self, pdb_dir, max_len=50):
        """
        Args:
            pdb_dir (str): Path to the directory containing 80k PDB files.
            max_len (int): Maximum sequence length. Longer proteins are filtered out 
                           or cropped to save GPU memory during ESM embedding.
        """
        self.pdb_dir = pdb_dir
        self.max_len = max_len
        
        # Grab all .pdb files in the directory
        print(f"Scanning directory {pdb_dir}...")
        self.files = glob.glob(os.path.join(pdb_dir, "*.pdb"))
        print(f"Found {len(self.files)} PDB files.")

    def __len__(self):
        return len(self.files)

    def _parse_pdb(self, filepath):
        """Fast custom parser for AlphaFold PDBs (Backbone only: N, CA, C)"""
        seq = []
        coords = []
        current_res_id = None
        current_res_coords = {}
        prev_res_name = ""

        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    atom_name = line[12:16].strip()
                    res_name = line[17:20].strip()
                    res_id = line[22:26].strip()

                    if atom_name in ['N', 'CA', 'C']:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])

                        if res_id != current_res_id:
                            if current_res_id is not None and len(current_res_coords) == 3:
                                seq.append(RESIDUE_MAP.get(prev_res_name, 'X'))
                                coords.append([current_res_coords['N'], current_res_coords['CA'], current_res_coords['C']])
                            
                            current_res_id = res_id
                            prev_res_name = res_name
                            current_res_coords = {}

                        current_res_coords[atom_name] = [x, y, z]

            # Catch the very last residue
            if current_res_id is not None and len(current_res_coords) == 3:
                seq.append(RESIDUE_MAP.get(prev_res_name, 'X'))
                coords.append([current_res_coords['N'], current_res_coords['CA'], current_res_coords['C']])

        sequence_str = "".join(seq)
        coords_tensor = torch.tensor(coords, dtype=torch.float32) # Shape: (L, 3, 3)
        
        return sequence_str, coords_tensor

    def __getitem__(self, idx):
        filepath = self.files[idx]
        seq, coords = self._parse_pdb(filepath)
        
        # Optional: Crop if it exceeds max_len to prevent ESM OOM
        if len(seq) > self.max_len:
            seq = seq[:self.max_len]
            coords = coords[:self.max_len]
            
        return {"sequence": seq, "coords": coords}


def collate_peptides(batch):
    """
    Pads the batch dynamically to the longest sequence in the current batch.
    """
    sequences = [item["sequence"] for item in batch]
    coords_list = [item["coords"] for item in batch]
    
    B = len(sequences)
    max_len = max(len(s) for s in sequences)
    
    # Initialize padded coordinates tensor (B, max_len, 3, 3)
    padded_coords = torch.zeros(B, max_len, 3, 3, dtype=torch.float32)
    
    for i, coords in enumerate(coords_list):
        L = coords.shape[0]
        padded_coords[i, :L] = coords
        
    return {
        "sequences": sequences,
        "coords": padded_coords
    }