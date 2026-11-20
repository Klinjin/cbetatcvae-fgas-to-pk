#!/bin/bash -l
#SBATCH -A m4031
#SBATCH -C gpu
#SBATCH --qos=regular
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH -J k3_sweep
#SBATCH -o experiments/sweeps/k3_20261001/slurm-%j.out
# Profile-only CVAE: k-window sweep + HP tuning for SP(k=3) (see scripts/k3_sweep.py).
module load python
conda activate /pscratch/sd/l/lindajin/envs/fgas-ml
export FGAS_SCRATCH_ROOT=/pscratch/sd/l/lindajin/fgas-to-pk-ml
cd /pscratch/sd/l/lindajin/fgas-to-pk-ml
python -u scripts/k3_sweep.py --out experiments/sweeps/k3_20261001 --workers-per-gpu 4 --n-gpus 4
