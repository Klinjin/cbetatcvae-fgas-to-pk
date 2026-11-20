#!/bin/bash
#SBATCH -A m4031
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:20:00
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -J ablate_only_Om_Ob_snap82
#SBATCH -o /pscratch/sd/l/lindajin/fgas-to-pk-ml/experiments/slurm_logs/ablate_only_Om_Ob_snap82_%j.out
module load python
conda activate /pscratch/sd/l/lindajin/envs/fgas-ml
cd /pscratch/sd/l/lindajin/fgas-to-pk-ml
export FGAS_SCRATCH_ROOT=/pscratch/sd/l/lindajin/fgas-to-pk-ml
g='arcmin9_cosmo-only-Omega0+OmegaBaryon_kmax10_snap82'
python -u scripts/ablate_cosmo_params.py --snap 82 --train --arms only-Omega0+OmegaBaryon
F=figures/$(date +%Y-%m)/$(date +%m-%d)
mkdir -p "$F/$g"
for d in experiments/runs/$g/*; do
    cid=$(basename $d | cut -d_ -f3)
    mv $F/*__${cid}__cvae__* "$F/$g/" && echo "moved figures for $cid -> $F/$g"
done
