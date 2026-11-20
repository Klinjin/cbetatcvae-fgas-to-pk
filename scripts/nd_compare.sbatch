#!/bin/bash -l
#SBATCH -A m4031
#SBATCH -C gpu
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH -J nd_compare
#SBATCH -o experiments/sweeps/nd_compare_20261007/slurm-%j.out
# Number-density comparison, arcmin9 k<=10 CVAE: nd_2 reference vs nd_4 (see scripts/nd_compare.py).
module load python
conda activate /pscratch/sd/l/lindajin/envs/fgas-ml
export FGAS_SCRATCH_ROOT=/pscratch/sd/l/lindajin/fgas-to-pk-ml
cd /pscratch/sd/l/lindajin/fgas-to-pk-ml
set -e
python -u scripts/nd_compare.py --nd 4 --train --report
