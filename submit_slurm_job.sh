#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodelist=ablette,anchois      # Names of the computers to use as nodes for the training
#SBATCH --nodes=2            # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1   # This needs to match Trainer(devices=...)
#SBATCH --mem=0
#SBATCH --time=0-00:01:00       # Max time allowed

# activate conda env
source activate env_diff/bin/activate

# run script from above
srun python encdec.py