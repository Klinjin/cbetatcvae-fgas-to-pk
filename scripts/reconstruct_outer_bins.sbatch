#!/bin/bash -l
#SBATCH -A m4031
#SBATCH -C gpu
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH -J outer_bins
#SBATCH -o experiments/outer_bins/slurm-%j.out
# 5-fold CV + final Student-t GP fit of the outer-bin reconstructor for up to
# four (snapshot:nd) cases, one per GPU (see scripts/reconstruct_outer_bins.py).
#   sbatch --export=ALL,TASKS="74:0 74:1 74:2 74:3" scripts/reconstruct_outer_bins.sbatch
#   sbatch --export=ALL,TASKS="74:4 82:0 82:1 82:2" scripts/reconstruct_outer_bins.sbatch
#   sbatch --export=ALL,TASKS="82:3 82:4" scripts/reconstruct_outer_bins.sbatch
module load python
conda activate /pscratch/sd/l/lindajin/envs/fgas-ml
export FGAS_SCRATCH_ROOT=/pscratch/sd/l/lindajin/fgas-to-pk-ml
cd /pscratch/sd/l/lindajin/fgas-to-pk-ml
g=0
for t in ${TASKS:?set TASKS, e.g. "74:0 74:1"}; do
  s=${t%%:*}; nd=${t##*:}
  CUDA_VISIBLE_DEVICES=$((g % 4)) python -u -W ignore scripts/reconstruct_outer_bins.py fit \
    --snapshot $s --nd $nd > experiments/outer_bins/fit_snap${s}_nd${nd}.log 2>&1 &
  g=$((g + 1))
done
wait
for t in $TASKS; do grep -h '^{' experiments/outer_bins/fit_snap${t%%:*}_nd${t##*:}.log; done
