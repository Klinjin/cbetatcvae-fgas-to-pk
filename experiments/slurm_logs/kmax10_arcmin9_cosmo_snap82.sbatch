#!/bin/bash
#SBATCH -A m4031
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:20:00
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -J kmax10_arcmin9_cosmo_snap82
#SBATCH -o /pscratch/sd/l/lindajin/fgas-to-pk-ml/experiments/slurm_logs/kmax10_arcmin9_cosmo_snap82_%j.out
module load python
conda activate /pscratch/sd/l/lindajin/envs/fgas-ml
cd /pscratch/sd/l/lindajin/fgas-to-pk-ml
export FGAS_SCRATCH_ROOT=/pscratch/sd/l/lindajin/fgas-to-pk-ml
g=arcmin9_cosmo_kmax10_snap82
python scripts/kmax10_noparams.py --design arcmin9 --cosmo --snap 82 --train --seed 2
F=figures/$(date +%Y-%m)/$(date +%m-%d)
for d in experiments/runs/$g/*; do
    cid=$(basename $d | cut -d_ -f3)
    mv $F/*__${cid}__cvae__* $F/$g/ && echo "moved figures for $cid -> $F/$g"
done
