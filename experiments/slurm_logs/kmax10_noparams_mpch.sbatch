#!/bin/bash
#SBATCH -A m4031
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:20:00
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -J kmax10_noparams_mpch
#SBATCH -o /pscratch/sd/l/lindajin/fgas-to-pk-ml/experiments/slurm_logs/kmax10_noparams_mpch_%j.out
module load python
conda activate /pscratch/sd/l/lindajin/envs/fgas-ml
cd /pscratch/sd/l/lindajin/fgas-to-pk-ml
export FGAS_SCRATCH_ROOT=/pscratch/sd/l/lindajin/fgas-to-pk-ml
python scripts/kmax10_noparams.py --design mpch --train --seed 2
