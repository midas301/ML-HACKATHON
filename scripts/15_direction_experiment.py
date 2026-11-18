import os
import sys
import json
import time
import pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap import config
from pathsnap.pipeline import cached
from pathsnap.data import Trip
from pathsnap.metrics import evaluate
from pathsnap.hmm import RoadGraph, decode
from pathsnap.candidates import FEATURES, gps_kinematics
from pathsnap.direction import DIRECTION_FEATURES, N_BASE

cfg = config.load()
exp_dir = os.path.join(cfg["artifacts_dir"], "experiments")
os.makedirs(exp_dir, exist_ok=True)
params = json.load(open(os.path.join(cfg["artifacts_dir"], "best_lgbm_params.json")))
hmm_cfg = json.load(open(os.path.join(cfg["artifacts_dir"], "hmm_config.json")))


def augment(part, root):
    trips = cached(part, root, cfg)
    for d in trips:
        d["X_old"] = d["X"][:, :N_BASE]
        d["X_new"] = d["X"]
    return trips


t0 = time.time()
train = augment("train", cfg["train_dir"])
test = augment("student_test", cfg["student_test_dir"])
print(f"direction features built in {time.time() - t0:.0f}s", flush=True)
groups = np.arange(len(train))


def rows(trips, key):
    return np.vstack([d[key][d["row_valid"]] for d in trips]), np.concatenate([d["y"][d["row_valid"]] for d in trips])


def hmm_top1(trips, probs, root):
    hit = n = 0
    out = []
    for d, p in zip(trips, probs):
        tr = Trip(root, d["trip_id"])
        pred, _ = decode(RoadGraph(tr.arcs, tr.arc_len), tr.xy, tr.t, d["pid"], d["aid"], d["proj_t"] * tr.arc_len[d["aid"]], p, hmm_cfg)
        v = d["label_valid"]
        h = int((pred[v] == d["label_arc"][v]).sum())
        hit += h; n += int(v.sum())
        out.append((d, pred))
    return hit / n, out


results = {}
oof_store = {}
for key in ("X_old", "X_new"):
    folds = []
    oof = {}
    for tr_idx, va_idx in GroupKFold(n_splits=5).split(groups, groups=groups):
        X, y = rows([train[i] for i in tr_idx], key)
        m = LGBMClassifier(**params).fit(X, y)
        va = [train[i] for i in va_idx]
        probs = [m.predict_proba(d[key])[:, 1] for d in va]
        r, _, _ = evaluate(va, probs)
        folds.append(r)
        for d, p in zip(va, probs):
            oof[d["trip_id"]] = p
        print(key, "fold", len(folds), round(r["top1"], 4), flush=True)
    X, y = rows(train, key)
    m = LGBMClassifier(**params).fit(X, y)
    probs = [m.predict_proba(d[key])[:, 1] for d in test]
    r, per_trip, _ = evaluate(test, probs, root=cfg["student_test_dir"])
    h_test, decoded = hmm_top1(test, probs, cfg["student_test_dir"])
    tune_ids = sorted(np.random.default_rng(42).choice(sorted(oof), 40, replace=False).tolist())
    tune = [d for d in train if d["trip_id"] in tune_ids]
    h_oof, _ = hmm_top1(tune, [oof[d["trip_id"]] for d in tune], cfg["train_dir"])
    speed_rows = []
    for d, pred in decoded:
        tr = Trip(cfg["student_test_dir"], d["trip_id"])
        sp = gps_kinematics(tr)[0]
        v = d["label_valid"]
        for i in np.where(v)[0]:
            a, b = pred[i], d["label_arc"][i]
            twin = a >= 0 and a != b and tr.arcs[a, 0] == tr.arcs[b, 1] and tr.arcs[a, 1] == tr.arcs[b, 0]
            speed_rows.append((sp[i], float(a == b), float(twin)))
    S = pd.DataFrame(speed_rows, columns=["speed", "ok", "twin"])
    S["band"] = pd.cut(S.speed, [-1, 0.5, 2, 5, 10, 100], labels=["<0.5", "0.5-2", "2-5", "5-10", ">10"])
    cv = pd.DataFrame(folds)
    results[key] = {
        "n_features": int(X.shape[1]),
        "cv_top1_mean": cv.top1.mean(), "cv_top1_std": cv.top1.std(), "cv_macro_f1_mean": cv.macro_f1.mean(), "cv_log_loss_mean": cv.log_loss.mean(),
        "train_oof_hmm_top1_40trips": h_oof,
        "test_top1": r["top1"], "test_macro_f1": r["macro_f1"], "test_log_loss": r["log_loss"], "test_continuity": r["continuity"],
        "test_hmm_top1": h_test, "test_hmm_wrong_direction_errors": int(S.twin.sum()),
        "test_hmm_top1_by_speed": S.groupby("band", observed=True).ok.mean().round(4).to_dict(),
        "per_trip_top1": {p["trip_id"]: round(p["top1"], 4) for p in per_trip},
    }
    if key == "X_new":
        joblib.dump(m, os.path.join(exp_dir, "lgbm_tuned_direction.joblib"), compress=3)
        imp = pd.Series(m.booster_.feature_importance("gain"), index=FEATURES + DIRECTION_FEATURES).sort_values(ascending=False)
        results[key]["top_features_gain"] = imp.head(10).round(0).to_dict()
    print(key, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in results[key].items() if k != "per_trip_top1"}), flush=True)

with open(os.path.join(cfg["results_dir"], "feature_ablation_direction.json"), "w") as f:
    json.dump(results, f, indent=2, default=float)
