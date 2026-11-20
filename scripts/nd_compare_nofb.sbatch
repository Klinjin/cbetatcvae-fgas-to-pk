#!/bin/bash -l
#SBATCH -A m4031
#SBATCH -C gpu
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH -J nd_compare_nofb
#SBATCH -o experiments/sweeps/nd_compare_nofb_20261118/slurm-%j.out
# Number-density comparison, arcmin9 k<=10 CVAE: the nofb (f_gas not / f_b, no params) arms at the
# remaining nd bins 0, 1, 3 (nd_4 done in nd_compare_20261007, cosmo arms in nd_compare_20261023).
# Each nd: train the nofb arms of both snapshots (skipped if already recorded).
module load python
conda activate /pscratch/sd/l/lindajin/envs/fgas-ml
export FGAS_SCRATCH_ROOT=/pscratch/sd/l/lindajin/fgas-to-pk-ml
cd /pscratch/sd/l/lindajin/fgas-to-pk-ml
set -e
for nd in 0 1 3; do
    python -u scripts/nd_compare.py --nd $nd --inputs nofb --train
done
