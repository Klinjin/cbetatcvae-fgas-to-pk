"""Head-to-head default vs selected (rand21) hyperparameters over 4 CV repetitions,
window [1, 3.1], keeping the out-of-fold SP(k~3) predictions for an outlier analysis.
Non-test sims only (test fold untouched)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "scripts")
import k3_sweep as K

D = Path("experiments/sweeps/k3_20261001")
sel = json.loads((D / "selection.json").read_text())
CFGS = {"default": K._default_hp(), sel["stage3_selected_tag"]: sel["selected_hp"]}
WIN = (1.0, 3.1)
REPS = (101, 202, 303, 404)
out = {}
for tag, hp in CFGS.items():
    for snap in K.SNAPS:
        td, masks, _ = K._data(snap, WIN)
        nt = ~masks["test"]
        i = int(np.argmin(np.abs(td.k - K.K_STAR)))
        for rep in REPS:
            pred = np.full(len(td.y), np.nan)
            for f in range(K.N_FOLDS):
                r = K._fit_fold({"window": list(WIN), "snap": snap, "fold": f, "cv_seed": rep, "hp": hp})
                pred[r["rows"]] = np.asarray(r["pred"])[:, i]
            y = np.asarray(td.y)[:, i]
            out[f"{tag}|{snap}|{rep}"] = {"sim_index": td.sim_index[nt].tolist(), "y": y[nt].tolist(), "pred": pred[nt].tolist()}
            e = pred[nt] - y[nt]
            print(tag, snap, rep, "rmse", round(float(np.sqrt(np.mean(e**2))), 4), "med|d|", round(float(np.median(np.abs(e / y[nt]))) * 100, 2), flush=True)
(D / "robustness_oof.json").write_text(json.dumps(out))
