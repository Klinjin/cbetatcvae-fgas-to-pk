#!/bin/bash -l
#SBATCH -A m4031
#SBATCH -C gpu
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH -J cosmo_ablation
#SBATCH -o experiments/sweeps/cosmo_ablation_20261001/slurm-%j.out
# Cosmological-parameter ablation, arcmin9 k<=10 CVAE (see scripts/ablate_cosmo_params.py).
module load python
conda activate /pscratch/sd/l/lindajin/envs/fgas-ml
export FGAS_SCRATCH_ROOT=/pscratch/sd/l/lindajin/fgas-to-pk-ml
cd /pscratch/sd/l/lindajin/fgas-to-pk-ml
python -u scripts/ablate_cosmo_params.py --train --report
