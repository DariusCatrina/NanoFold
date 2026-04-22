# NanoFold: Peptide Backbone Structure Generation on SE(3) via Conditional Flow Matching

NanoFold is a generative modeling framework for static peptide structure prediction. Developed as a didactical exploration into state-of-the-art computational biology architectures, it generates the 3D spatial coordinates of short peptide backbones on the $\text{SE}(3)$ manifold using continuous-time conditional flow matching.

Rather than modeling the kinetic folding trajectory, this project maps amino acid sequences directly to stable 3D conformations. It conditions the generation process on **ESM2** language model embeddings and heavily leverages architectural components inspired by AlphaFold2, including custom from-scratch implementations of **Invariant Point Attention (IPA)** and **Triangle Multiplicative Operations**, stabilized via **Adaptive Layer Normalization (AdaLN)** and **Self-Conditioning**.

## Repository Structure

* **`nanofold/`**: The core generative architecture.
    * `_layers.py`: Contains custom implementations of complex geometric operations (IPA, Triangle Multiplicative updates) and AdaLN for time-step injection.
    * `_structure_module.py`: The SE(3) structure generation module adapted for flow matching.
* **`openfold/`**: Utility modules, MSA handling, residue constants, and geometric processing tools ported from [OpenFold](https://github.com/aqlaboratory/openfold) to maintain strict physical validity during feature processing.
* **`data/`**: Visualization tools and the dataset download pipeline.
* **`dataset.py`**: PyTorch dataset implementation for loading and processing PDB files and sequence data.
* **`train.py`**: Main entry point for training the flow matching ODE.

## Requirements

Ensure you have Python 3.10+ installed along with PyTorch. Additional dependencies typically include standard scientific computing libraries (`numpy`, `einops`, `biopython`, etc.) as well as the pre-trained ESM2 model weights via `transformers`.

## Data Preparation

Due to the immense computational overhead of SE(3) geometric operations, the model is trained on a highly curated dataset of 20,000 short peptides (length $\le$ 50 amino acids) with high confidence scores (pLDDT > 80%) sourced from the AlphaFold Protein Structure Database (AFDB).

### 1. The AFDB Clusters Dataset
To understand the full scope of the available data or to generate your own filtered lists, you will need the AlphaFold DB clusters file. 
* **File:** `1-AFDBClusters-entryId_repId_taxId.tsv.gz`
* **Source:** This file can be found and downloaded from the [Foldseek website](https://search.foldseek.com/).

### 2. Downloading Target PDB Files
We have provided a pre-filtered list of target sequences in `data/download/small_peptides.tsv`. To download the actual `.pdb` structural files for training, use the provided shell script:

```bash
cd data/download

# Make the script executable
chmod +x download_pdb.sh

# Run the download script, passing the TSV file as an argument
./download_pdb.sh small_peptides.tsv