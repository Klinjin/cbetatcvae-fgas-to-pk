#!/bin/bash -l
#SBATCH -A m4031
#SBATCH -C gpu
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH -J joint_cvae_fgas_std
#SBATCH -o experiments/sweeps/joint_cvae_fgas_std_20261117/slurm-%j.out
# Joint cVAE with [f_gas, log10 sigma_fgas] profile input (see scripts/joint_cvae_fgas_std.py).
module load python
conda activate /pscratch/sd/l/lindajin/envs/fgas-ml
export FGAS_SCRATCH_ROOT=/pscratch/sd/l/lindajin/fgas-to-pk-ml
cd /pscratch/sd/l/lindajin/fgas-to-pk-ml
set -e
python -u scripts/joint_cvae_fgas_std.py --train --report --diagnostics
