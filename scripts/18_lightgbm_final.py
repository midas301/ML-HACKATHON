import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap import config
from pathsnap.pipeline import cached
from pathsnap.metrics import evaluate
from pathsnap.models import LGBM_DEFAULT, SEED
from pathsnap.direction import ALL_FEATURES

cfg = config.load()
R, A, F = cfg["results_dir"], cfg["artifacts_dir"], cfg["figures_dir"]
train = cached("train", cfg["train_dir"], cfg)
test = cached("student_test", cfg["student_test_dir"], cfg)
tuned = json.load(open(os.path.join(A, "best_lgbm_params.json")))
groups = np.arange(len(train))
n_obs = sum(d["n_points"] for d in test)
rng = np.random.default_rng(SEED)
sample = [train[i] for i in rng.choice(len(train), 20, replace=False)]


def rows(trips):
    return np.vstack([d["X"][d["row_valid"]] for d in trips]), np.concatenate([d["y"][d["row_valid"]] for d in trips])


configs = [
    ("LightGBM_dir", "LightGBM + direction features (untuned)", dict(LGBM_DEFAULT)),
    ("LightGBM_dir_tuned", "LightGBM + direction features (tuned, final)", tuned),
]
for key, label, params in configs:
    folds, oof = [], {}
    for k, (tr_idx, va_idx) in enumerate(GroupKFold(n_splits=5).split(groups, groups=groups)):
        X, y = rows([train[i] for i in tr_idx])
        t0 = time.time(); m = LGBMClassifier(**params).fit(X, y); ft = time.time() - t0
        va = [train[i] for i in va_idx]
        probs = [m.predict_proba(d["X"])[:, 1] for d in va]
        r, _, _ = evaluate(va, probs)
        r["fold"] = k; r["fit_seconds"] = ft
        folds.append(r)
        for d, p in zip(va, probs):
            oof[d["trip_id"]] = p
        print(key, "fold", k, round(r["top1"], 4), flush=True)
    X, y = rows(train)
    t0 = time.time(); m = LGBMClassifier(**params).fit(X, y); train_time = time.time() - t0
    t0 = time.time(); probs = [m.predict_proba(d["X"])[:, 1] for d in test]; inf = time.time() - t0
    res, per_trip, _ = evaluate(test, probs, root=cfg["student_test_dir"])
    tr_res, _, _ = evaluate(sample, [m.predict_proba(d["X"])[:, 1] for d in sample])
    cv = pd.DataFrame(folds)
    summary = {
        "model": label, "family": "Boosting", "config": ("Optuna best params, 29 features" if key.endswith("tuned") else "300 trees, lr=0.05, 31 leaves, 29 features"),
        "n_features": len(ALL_FEATURES),
        "cv_top1_mean": cv.top1.mean(), "cv_top1_std": cv.top1.std(), "cv_macro_f1_mean": cv.macro_f1.mean(), "cv_macro_f1_std": cv.macro_f1.std(),
        "cv_log_loss_mean": cv.log_loss.mean(),
        "train_top1": tr_res["top1"], "train_macro_f1": tr_res["macro_f1"], "train_log_loss": tr_res["log_loss"],
        **{"test_" + a: b for a, b in res.items()},
        "train_seconds": train_time, "inference_seconds": inf, "inference_ms_per_1k_obs": 1e6 * inf / n_obs,
    }
    cv.to_csv(os.path.join(R, f"cv_{key}.csv"), index=False)
    pd.DataFrame(per_trip).to_csv(os.path.join(R, f"student_test_per_trip_{key}.csv"), index=False)
    np.save(os.path.join(R, f"probs_{key}.npy"), np.concatenate(probs))
    json.dump(summary, open(os.path.join(R, f"summary_{key}.json"), "w"), indent=2)
    joblib.dump(oof, os.path.join(A, f"oof_{key}.joblib"))
    joblib.dump(m, os.path.join(A, "models", f"{key}.joblib"), compress=3)
    if key == "LightGBM_dir_tuned":
        joblib.dump(m, os.path.join(A, "final_model.joblib"), compress=3)
        imp = pd.Series(m.booster_.feature_importance("gain"), index=ALL_FEATURES).sort_values()
        imp.to_csv(os.path.join(R, "feature_importance_gain.csv"))
        imp.plot.barh(figsize=(7, 8), color=["#c8102e" if f in ALL_FEATURES[22:] else "#14284b" for f in imp.index])
        plt.title("Final LightGBM feature importance (gain); red = direction features"); plt.tight_layout()
        plt.savefig(os.path.join(F, "feature_importance.png"), dpi=120); plt.close()
    print("SUMMARY", json.dumps({a: (round(b, 4) if isinstance(b, float) else b) for a, b in summary.items()}), flush=True)

va_ids = set(json.load(open(os.path.join(R, "training_histories.json")))["valid_trips"])
Xtr, ytr = rows([d for d in train if d["trip_id"] not in va_ids])
Xva, yva = rows([d for d in train if d["trip_id"] in va_ids])
old = json.load(open(os.path.join(R, "training_histories.json")))
hist = {}
for key, label, params in configs:
    m = LGBMClassifier(**params).fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], eval_names=["train", "valid"], eval_metric="binary_logloss")
    hist[label] = {"train": m.evals_result_["train"]["binary_logloss"], "valid": m.evals_result_["valid"]["binary_logloss"]}
hist["MLP (22 features)"] = old["MLP"]
json.dump({k: {a: [float(x) for x in b] for a, b in v.items()} for k, v in hist.items()} | {"valid_trips": sorted(va_ids)},
          open(os.path.join(R, "training_histories.json"), "w"))
fig, ax = plt.subplots(1, 3, figsize=(18, 4.3))
for a, (k, v) in zip(ax, hist.items()):
    a.plot(v["train"], label="train"); a.plot(v["valid"], label="validation (16 held-out train trips)")
    best = int(np.argmin(v["valid"]))
    a.axvline(best, color="0.6", ls=":", lw=1); a.annotate(f"min valid {min(v['valid']):.4f} @ {best}", (best, min(v["valid"])), xytext=(5, 15), textcoords="offset points", fontsize=8)
    unit = "epoch" if k.startswith("MLP") else "boosting round"
    a.set_title(f"{k}\nlog loss per {unit}", fontsize=10); a.set_xlabel(unit); a.set_ylabel("binary log loss"); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(F, "training_histories.png"), dpi=120); plt.close()
print("histories done", flush=True)
