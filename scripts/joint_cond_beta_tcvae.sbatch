#!/bin/bash -l
#SBATCH -A m4031
#SBATCH -C gpu
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH -J cond_beta_tcvae
#SBATCH -o experiments/sweeps/cond_beta_tcvae/slurm-%j.out
# Joint conditional-prior beta-TCVAE (p(z|x,y_hat,ctx)) over snap74 (LRG) + snap82 (BGS)
# (see scripts/joint_cond_beta_tcvae.py), alpha = beta = gamma = lambda = 1.
module load python
conda activate /pscratch/sd/l/lindajin/envs/fgas-ml
export FGAS_SCRATCH_ROOT=/pscratch/sd/l/lindajin/fgas-to-pk-ml
cd /pscratch/sd/l/lindajin/fgas-to-pk-ml
set -e
python -u scripts/joint_cond_beta_tcvae.py --dims --train --report --diagnostics
