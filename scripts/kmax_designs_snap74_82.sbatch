#!/bin/bash -l
#SBATCH -A m4031
#SBATCH -C gpu
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH -J kmax_designs
#SBATCH -o experiments/sweeps/kmax_designs_20261003/slurm-%j.out
# Train the k-window / radial-design / conditioning arms missing for the snap74 + snap82
# comparison (scripts/kmax10_noparams.py; seeds 0/1/2 each, split seed 10).
module load python
conda activate /pscratch/sd/l/lindajin/envs/fgas-ml
export FGAS_SCRATCH_ROOT=/pscratch/sd/l/lindajin/fgas-to-pk-ml
cd /pscratch/sd/l/lindajin/fgas-to-pk-ml
set -e
python -u scripts/kmax10_noparams.py --design mpch9 --snap 74 --train
python -u scripts/kmax10_noparams.py --design mpch9 --snap 82 --train
python -u scripts/kmax10_noparams.py --design mpch --snap 82 --train
python -u scripts/kmax10_noparams.py --design arcmin --snap 82 --train
python -u scripts/kmax10_noparams.py --design arcmin --cosmo --snap 82 --train
python -u scripts/kmax10_noparams.py --design arcmin9 --kmax 3 --snap 82 --train
