#!/bin/bash -l
#SBATCH -A m4031
#SBATCH -C gpu
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH -J observed_pred_joint
#SBATCH -o experiments/sweeps/observed_pred_joint_models_20261117/slurm-%j.out
# Observed-f_gas predictions of the joint models + cross-model RMSE comparison.
#   sbatch scripts/observed_pred_joint_models.sbatch verify   # default-path reproduction checks
#   sbatch scripts/observed_pred_joint_models.sbatch separate # beam-corrected input: separate cVAEs at all 5 n
#   sbatch scripts/observed_pred_joint_models.sbatch separate_nofb # beam-corrected input: separate nofb cVAEs at all 5 n
#   sbatch scripts/observed_pred_joint_models.sbatch predict  # beam-corrected input: joint_cvae_fgas_std, beta_tcvae
#   sbatch scripts/observed_pred_joint_models.sbatch compare  # six-arm RMSE(k) comparison
module load python
conda activate /pscratch/sd/l/lindajin/envs/fgas-ml
export FGAS_SCRATCH_ROOT=/pscratch/sd/l/lindajin/fgas-to-pk-ml
cd /pscratch/sd/l/lindajin/fgas-to-pk-ml
set -e
STAGE=${1:?stage: verify | separate | separate_nofb | predict | compare}
OBS_DIR=/pscratch/sd/l/lindajin/DH_profile_kSZ_WL/data
VDIR=figures/2026-11/11-17/verify_default_paths
declare -A SNAP=([lrg_hscy3_combined]=74 [bgs_hscy3_combined]=82)

if [ "$STAGE" = verify ]; then
  mkdir -p $VDIR
  # 1) The generalised predictor's default path must reproduce the existing joint_cvae outputs.
  for key in lrg_hscy3_combined bgs_hscy3_combined; do
    python -u scripts/predict_observed_fgas_joint.py --obs $OBS_DIR/fgas_obs_$key.npz \
      --out $VDIR/spk_pred_from_fgas_obs_${key}_joint_cvae.npz --snap ${SNAP[$key]}
  done
  python -u - <<'EOF'
import json
import numpy as np
for key in ("lrg_hscy3_combined", "bgs_hscy3_combined"):
    old = np.load(f"/pscratch/sd/l/lindajin/DH_profile_kSZ_WL/data/spk_pred_from_fgas_obs_{key}_joint_cvae.npz")
    new = np.load(f"figures/2026-11/11-17/verify_default_paths/spk_pred_from_fgas_obs_{key}_joint_cvae.npz")
    assert sorted(old.files) == sorted(new.files), (key, set(old.files) ^ set(new.files))
    bad = [f for f in old.files if not (np.array_equal(old[f], new[f]) if old[f].dtype.kind in "US"
                                        else np.allclose(old[f], new[f]))]
    exact = [f for f in old.files if np.array_equal(old[f], new[f])]
    maxdiff = max(float(np.max(np.abs(old[f] - new[f]))) for f in old.files if old[f].dtype.kind == "f")
    print(f"{key}: {len(old.files)} keys, allclose fails: {bad}, bit-identical: {len(exact)}/{len(old.files)}, "
          f"max |diff| {maxdiff:.3g}")
    oj = json.load(open(f"/pscratch/sd/l/lindajin/DH_profile_kSZ_WL/data/spk_pred_from_fgas_obs_{key}_joint_cvae.json"))
    nj = json.load(open(f"figures/2026-11/11-17/verify_default_paths/spk_pred_from_fgas_obs_{key}_joint_cvae.json"))
    print(f"{key}: sidecar json equal: {oj == nj}")
EOF
  # 2) joint_dual_vae.py after the _separate_arm / _rmse_table refactor must reproduce the 11-16 table.
  python -u scripts/joint_dual_vae.py --report --out-dir $VDIR/joint_dual_vae_refactor
  python -u - <<'EOF'
import json
import numpy as np
old = json.load(open("figures/2026-11/11-16/joint_dual_vae/joint_dual_vae_rmse.json"))
new = json.load(open("figures/2026-11/11-17/verify_default_paths/joint_dual_vae_refactor/joint_dual_vae_rmse.json"))
assert old["setting"] == new["setting"]
for key, row in old["metrics"].items():
    for f, v in row.items():
        w = new["metrics"][key][f]
        ok = (np.allclose(np.asarray(v, float), np.asarray(w, float), rtol=1e-6, atol=1e-10)
              if f.endswith("per_seed") else
              (v == w if not isinstance(v, dict) else all(np.isclose(v[q], w[q], rtol=1e-6) for q in v)))
        assert ok, (key, f)
print("joint_dual_vae refactor: table reproduces figures/2026-11/11-16 json")
EOF
fi

# Beam-corrected DESI x ACT x HSC f_gas (fig16a BGS, fig16b LRG); no covariance -> diag(fgas_obs_err^2) only.
declare -A CSV=([lrg_hscy3_beamcorr]=fig16b.csv [bgs_hscy3_beamcorr]=fig16a.csv)
declare -A SNAPB=([lrg_hscy3_beamcorr]=74 [bgs_hscy3_beamcorr]=82)
src_note() { echo "/pscratch/sd/l/lindajin/SimulationStacker/tests/DESI_kSZ_data/henry/${CSV[$1]}, series 'DESI x ACT x HSC (beam-corrected)', columns fgas / fgas_err, theta_arcmin <= 6 (9 bins)"; }
refuse() { [ -e "$1" ] && { echo "$1 exists; refusing to overwrite"; exit 1; }; return 0; }

if [ "$STAGE" = separate ]; then
  # Separate per-(snapshot, n) cosmology-conditioned cVAEs at all 5 n (joint_cvae._separate_group).
  for key in lrg_hscy3_beamcorr bgs_hscy3_beamcorr; do
    s=${SNAPB[$key]}
    groups="nd_compare/arcmin9_cosmo_kmax10_snap${s}_nd0 nd_compare/arcmin9_cosmo_kmax10_snap${s}_nd1 arcmin9_cosmo_kmax10_snap${s}_nd2 nd_compare/arcmin9_cosmo_kmax10_snap${s}_nd3 nd_compare/arcmin9_cosmo_kmax10_snap${s}_nd4"
    out=$OBS_DIR/spk_pred_from_fgas_obs_${key}_separate_cvae_all_nd.npz
    refuse $out
    python -u scripts/predict_observed_fgas.py --obs $OBS_DIR/fgas_obs_$key.npz --out $out --fb-obs 0.158 \
      --groups $groups --obs-source "$(src_note $key)"
  done
fi

if [ "$STAGE" = separate_nofb ]; then
  # Separate per-(snapshot, n) nofb cVAEs (f_gas not / f_b, no params) at all 5 n; nd0/1/3 from
  # nd_compare_nofb.sbatch. These get fgas_obs x fb-obs (predict_observed_fgas.py).
  for key in lrg_hscy3_beamcorr bgs_hscy3_beamcorr; do
    s=${SNAPB[$key]}
    groups="nd_compare/arcmin9nofb_noparams_kmax10_snap${s}_nd0 nd_compare/arcmin9nofb_noparams_kmax10_snap${s}_nd1 arcmin9nofb_noparams_kmax10_snap${s}_nd2 nd_compare/arcmin9nofb_noparams_kmax10_snap${s}_nd3 nd_compare/arcmin9nofb_noparams_kmax10_snap${s}_nd4"
    out=$OBS_DIR/spk_pred_from_fgas_obs_${key}_separate_cvae_nofb_all_nd.npz
    refuse $out
    python -u scripts/predict_observed_fgas.py --obs $OBS_DIR/fgas_obs_$key.npz --out $out --fb-obs 0.158 \
      --groups $groups --obs-source "$(src_note $key)"
  done
fi

if [ "$STAGE" = predict ]; then
  # joint_cvae on the beam-corrected input already exists (..._beamcorr_joint_cvae.npz) and is reused.
  for key in lrg_hscy3_beamcorr bgs_hscy3_beamcorr; do
    for g in joint_cvae_fgas_std beta_tcvae; do
      out=$OBS_DIR/spk_pred_from_fgas_obs_${key}_${g}.npz
      refuse $out
      extra=""; [ $g = joint_cvae_fgas_std ] && extra="--fgas-std"
      python -u scripts/predict_observed_fgas_joint.py --obs $OBS_DIR/fgas_obs_$key.npz --out $out \
        --snap ${SNAPB[$key]} --group $g $extra --obs-source "$(src_note $key)"
    done
  done
fi

if [ "$STAGE" = compare ]; then
  python -u scripts/compare_joint_models.py
fi
