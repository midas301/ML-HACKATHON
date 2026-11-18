import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss
from lightgbm import LGBMClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap import config
from pathsnap.pipeline import cached
from pathsnap.models import LGBM_DEFAULT, SEED

cfg = config.load()
train = cached("train", cfg["train_dir"], cfg)
from pathsnap.direction import N_BASE
for _d in train:
    _d["X"] = _d["X"][:, :N_BASE]
rng = np.random.default_rng(SEED)
va_idx = set(rng.choice(len(train), 16, replace=False).tolist())


def rows(trips):
    return (np.vstack([d["X"][d["row_valid"]] for d in trips]), np.concatenate([d["y"][d["row_valid"]] for d in trips]))


Xtr, ytr = rows([d for i, d in enumerate(train) if i not in va_idx])
Xva, yva = rows([d for i, d in enumerate(train) if i in va_idx])
hist = {}

best_path = os.path.join(cfg["artifacts_dir"], "best_lgbm_params.json")
for tag, params in (("LightGBM_untuned", dict(LGBM_DEFAULT)),) + ((("LightGBM_tuned", json.load(open(best_path))),) if os.path.exists(best_path) else ()):
    m = LGBMClassifier(**params)
    m.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], eval_names=["train", "valid"], eval_metric="binary_logloss")
    ev = m.evals_result_
    hist[tag] = {"train": ev["train"]["binary_logloss"], "valid": ev["valid"]["binary_logloss"]}

sc = StandardScaler().fit(Xtr)
A, B = sc.transform(Xtr), sc.transform(Xva)
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=1024, learning_rate_init=1e-3, random_state=SEED)
tr_l, va_l = [], []
for ep in range(25):
    mlp.partial_fit(A, ytr, classes=[0, 1])
    tr_l.append(log_loss(ytr, mlp.predict_proba(A)[:, 1])); va_l.append(log_loss(yva, mlp.predict_proba(B)[:, 1]))
    print("mlp epoch", ep, tr_l[-1], va_l[-1], flush=True)
hist["MLP"] = {"train": tr_l, "valid": va_l}

with open(os.path.join(cfg["results_dir"], "training_histories.json"), "w") as f:
    json.dump({k: {a: [float(x) for x in b] for a, b in v.items()} for k, v in hist.items()} | {"valid_trips": sorted(train[i]["trip_id"] for i in va_idx)}, f)

fig, ax = plt.subplots(1, len(hist), figsize=(6 * len(hist), 4))
for a, (k, v) in zip(np.atleast_1d(ax), hist.items()):
    a.plot(v["train"], label="train"); a.plot(v["valid"], label="validation (16 held-out train trips)")
    a.set_title(f"{k}: log loss per {'epoch' if k == 'MLP' else 'boosting round'}"); a.set_xlabel("epoch" if k == "MLP" else "boosting round")
    a.set_ylabel("binary log loss"); a.legend()
plt.tight_layout(); plt.savefig(os.path.join(cfg["figures_dir"], "training_histories.png"), dpi=120); plt.close()
