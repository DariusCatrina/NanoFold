#!/bin/bash

#SBATCH --job-name=mbe_finetune          
#SBATCH --output=runs/nanoflow_%j.out   
#SBATCH --error=runs/nanoflow_%j.err    
#SBATCH --nodes=1                       
#SBATCH --ntasks=1                       
#SBATCH --cpus-per-task=4                
#SBATCH --mem=100G                       
#SBATCH --gres=gpu:1                   
#SBATCH --partition=cellbio-dgx     


eval "$(mamba shell hook --shell bash)"
mamba activate multiflow

python train.py
