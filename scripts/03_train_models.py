import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap import config
from pathsnap.pipeline import cached
from pathsnap.metrics import evaluate
from pathsnap.models import REGISTRY, SEED

cfg = config.load()
train = cached("train", cfg["train_dir"], cfg)
test = cached("student_test", cfg["student_test_dir"], cfg)
only = sys.argv[1:] or list(REGISTRY)
rng = np.random.default_rng(SEED)
out_dir = cfg["results_dir"]
os.makedirs(os.path.join(cfg["artifacts_dir"], "models"), exist_ok=True)


def fit_rows(trips, cap):
    X = np.vstack([d["X"][d["row_valid"]] for d in trips])
    y = np.concatenate([d["y"][d["row_valid"]] for d in trips])
    if cap and len(y) > cap:
        idx = np.random.default_rng(SEED).choice(len(y), cap, replace=False)
        X, y = X[idx], y[idx]
    return X, y


def predict(model, trips):
    return [model.predict_proba(d["X"])[:, 1] for d in trips]


gkf = GroupKFold(n_splits=5)
groups = np.arange(len(train))
for name in only:
    family, desc, factory, cap = REGISTRY[name]
    print("=" * 10, name, flush=True)
    folds = []
    for k, (tr_idx, va_idx) in enumerate(gkf.split(groups, groups=groups)):
        tr = [train[i] for i in tr_idx]; va = [train[i] for i in va_idx]
        X, y = fit_rows(tr, cap)
        m = factory(); t0 = time.time(); m.fit(X, y); ft = time.time() - t0
        r, _, _ = evaluate(va, predict(m, va))
        r["fold"] = k; r["fit_seconds"] = ft
        folds.append(r)
        print(name, "fold", k, {a: round(b, 4) for a, b in r.items()}, flush=True)
    X, y = fit_rows(train, cap)
    m = factory(); t0 = time.time(); m.fit(X, y); train_time = time.time() - t0
    t0 = time.time(); probs = predict(m, test); inf = time.time() - t0
    n_obs = sum(d["n_points"] for d in test)
    res, per_trip, _ = evaluate(test, probs, root=cfg["student_test_dir"])
    sample = [train[i] for i in rng.choice(len(train), 20, replace=False)]
    tr_res, _, _ = evaluate(sample, predict(m, sample))
    cv = pd.DataFrame(folds)
    summary = {
        "model": name, "family": family, "config": desc,
        "cv_top1_mean": cv["top1"].mean(), "cv_top1_std": cv["top1"].std(),
        "cv_macro_f1_mean": cv["macro_f1"].mean(), "cv_macro_f1_std": cv["macro_f1"].std(),
        "cv_log_loss_mean": cv["log_loss"].mean(),
        "train_top1": tr_res["top1"], "train_macro_f1": tr_res["macro_f1"], "train_log_loss": tr_res["log_loss"],
        **{"test_" + a: b for a, b in res.items()},
        "train_seconds": train_time, "inference_seconds": inf, "inference_ms_per_1k_obs": 1e6 * inf / n_obs,
    }
    cv.to_csv(os.path.join(out_dir, f"cv_{name}.csv"), index=False)
    pd.DataFrame(per_trip).to_csv(os.path.join(out_dir, f"student_test_per_trip_{name}.csv"), index=False)
    np.save(os.path.join(out_dir, f"probs_{name}.npy"), np.concatenate(probs))
    with open(os.path.join(out_dir, f"summary_{name}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    joblib.dump(m, os.path.join(cfg["artifacts_dir"], "models", f"{name}.joblib"), compress=3)
    print("SUMMARY", json.dumps({a: (round(b, 4) if isinstance(b, float) else b) for a, b in summary.items()}), flush=True)
