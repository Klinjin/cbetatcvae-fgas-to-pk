#!/bin/bash -l
#SBATCH -A m4031
#SBATCH -C gpu
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH -J beta_tcvae
#SBATCH -o experiments/sweeps/beta_tcvae/slurm-%j.out
# Joint beta-TCVAE over snap74 (LRG) + snap82 (BGS) and all number densities
# (see scripts/joint_beta_tcvae.py), alpha = beta = gamma = lambda = 1.
module load python
conda activate /pscratch/sd/l/lindajin/envs/fgas-ml
export FGAS_SCRATCH_ROOT=/pscratch/sd/l/lindajin/fgas-to-pk-ml
cd /pscratch/sd/l/lindajin/fgas-to-pk-ml
set -e
python -u scripts/joint_beta_tcvae.py --dims --train --report --diagnostics
