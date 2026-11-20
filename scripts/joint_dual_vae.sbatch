#!/bin/bash -l
#SBATCH -A m4031
#SBATCH -C gpu
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH -J joint_dual_vae
#SBATCH -o experiments/sweeps/joint_dual_vae_20261116/slurm-%j.out
# One dual_vae (Arm P) over snap74 (LRG) + snap82 (BGS) and all number densities
# (see scripts/joint_dual_vae.py). One seed per job (~20 min on an A100):
#     sbatch --export=ALL,SEED=0 scripts/joint_dual_vae.sbatch
module load python
conda activate /pscratch/sd/l/lindajin/envs/fgas-ml
export FGAS_SCRATCH_ROOT=/pscratch/sd/l/lindajin/fgas-to-pk-ml
cd /pscratch/sd/l/lindajin/fgas-to-pk-ml
set -e
python -u scripts/joint_dual_vae.py --train --seeds "${SEED:?set SEED}"
