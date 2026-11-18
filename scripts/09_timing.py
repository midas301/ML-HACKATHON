import os
import sys
import json
import time
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap import config
from pathsnap.pipeline import cached
from pathsnap.models import REGISTRY, SEED

cfg = config.load()
train = cached("train", cfg["train_dir"], cfg)
test = cached("student_test", cfg["student_test_dir"], cfg)
X = np.vstack([d["X"][d["row_valid"]] for d in train])
y = np.concatenate([d["y"][d["row_valid"]] for d in train])
Xt = np.vstack([d["X"] for d in test])
n_obs = sum(d["n_points"] for d in test)
best = json.load(open(os.path.join(cfg["artifacts_dir"], "best_lgbm_params.json")))
items = [(n, f, c) for n, (_, _, f, c) in REGISTRY.items()] + [("LightGBM (tuned)", lambda: LGBMClassifier(**best), None)]
rows = []
for name, factory, cap in items:
    Xa, ya = X, y
    if cap:
        idx = np.random.default_rng(SEED).choice(len(y), cap, replace=False)
        Xa, ya = X[idx], y[idx]
    m = factory()
    t0 = time.time(); m.fit(Xa, ya); ft = time.time() - t0
    t0 = time.time(); m.predict_proba(Xt); it = time.time() - t0
    rows.append({"model": name, "train_rows": len(ya), "train_seconds": ft, "inference_seconds": it, "inference_ms_per_1k_obs": 1e6 * it / n_obs})
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(os.path.join(cfg["results_dir"], "timing_clean.csv"), index=False)
