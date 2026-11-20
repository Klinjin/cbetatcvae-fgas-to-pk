#!/bin/bash
#SBATCH -A m4031
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:20:00
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -J kmax10_cosmo
#SBATCH -o /pscratch/sd/l/lindajin/fgas-to-pk-ml/experiments/slurm_logs/kmax10_cosmo_%j.out
module load python
conda activate /pscratch/sd/l/lindajin/envs/fgas-ml
cd /pscratch/sd/l/lindajin/fgas-to-pk-ml
export FGAS_SCRATCH_ROOT=/pscratch/sd/l/lindajin/fgas-to-pk-ml
python scripts/kmax10_noparams.py --design arcmin --cosmo --train --seed 2
python scripts/kmax10_noparams.py --design mpch --cosmo --train --seed 2
