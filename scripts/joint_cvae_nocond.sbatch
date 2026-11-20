#!/bin/bash -l
#SBATCH -A m4031
#SBATCH -C gpu
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH -J joint_cvae_nocond
#SBATCH -o experiments/sweeps/joint_cvae_nocond_20261117/slurm-%j.out
# Joint cVAE with no X_cond and no X_params (see scripts/joint_cvae_nocond.py).
module load python
conda activate /pscratch/sd/l/lindajin/envs/fgas-ml
export FGAS_SCRATCH_ROOT=/pscratch/sd/l/lindajin/fgas-to-pk-ml
cd /pscratch/sd/l/lindajin/fgas-to-pk-ml
set -e
python -u scripts/joint_cvae_nocond.py --train
