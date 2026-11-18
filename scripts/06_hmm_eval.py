import os
import sys
import json
import time
import itertools
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap import config
from pathsnap.pipeline import cached
from pathsnap.data import Trip
from pathsnap.hmm import RoadGraph, decode
from pathsnap.metrics import top1_choice, continuity, sequence_scores
from pathsnap.models import REGISTRY

cfg = config.load()
res_dir, fig_dir = cfg["results_dir"], cfg["figures_dir"]
train = cached("train", cfg["train_dir"], cfg)
test = cached("student_test", cfg["student_test_dir"], cfg)
oof = joblib.load(os.path.join(cfg["artifacts_dir"], "oof_lgbm_tuned.joblib"))
_trips = {}


def get_trip(root, tid):
    k = (root, tid)
    if k not in _trips:
        tr = Trip(root, tid, load_route=True)
        _trips[k] = (tr, RoadGraph(tr.arcs, tr.arc_len))
    return _trips[k]


def run(root, d, prob, params):
    tr, g = get_trip(root, d["trip_id"])
    clf, _ = top1_choice(d["pid"], prob, d["n_points"], d["aid"])
    t0 = time.time()
    hmm, breaks = decode(g, tr.xy, tr.t, d["pid"], d["aid"], d["proj_t"] * tr.arc_len[d["aid"]], prob, params)
    dt = time.time() - t0
    v = d["label_valid"]
    out = {"trip_id": d["trip_id"], "points": int(v.sum()), "breaks": breaks, "decode_seconds": dt}
    for tag, p in (("clf", clf), ("hmm", hmm)):
        out[f"{tag}_hits"] = int((p[v] == d["label_arc"][v]).sum())
        out[f"{tag}_top1"] = out[f"{tag}_hits"] / max(1, v.sum())
        out[f"{tag}_continuity"] = continuity(p, tr.arcs)
        out[f"{tag}_seq_precision"], out[f"{tag}_seq_recall"], out[f"{tag}_seq_f1"] = sequence_scores(p, tr.route)
    return out


def agg(rows):
    df = pd.DataFrame(rows)
    r = {}
    for tag in ("clf", "hmm"):
        r[f"{tag}_top1"] = df[f"{tag}_hits"].sum() / df.points.sum()
        for m in ("continuity", "seq_precision", "seq_recall", "seq_f1"):
            r[f"{tag}_{m}"] = df[f"{tag}_{m}"].mean()
    r["breaks"] = int(df.breaks.sum()); r["decode_seconds"] = float(df.decode_seconds.sum())
    return r, df


tune_ids = sorted(oof)
rng = np.random.default_rng(42)
tune_ids = sorted(rng.choice(tune_ids, 40, replace=False).tolist())
tune = [d for d in train if d["trip_id"] in tune_ids]
grid = []
for beta, w in itertools.product([5, 10, 20, 40, 80], [0.5, 1.0, 2.0]):
    params = {"beta": beta, "w_emission": w, "vmax": 50.0, "back_tol": 15.0, "break_penalty": 30.0}
    r, _ = agg([run(cfg["train_dir"], d, oof[d["trip_id"]], params) for d in tune])
    grid.append({"beta": beta, "w_emission": w, **r})
    print("grid", beta, w, round(r["clf_top1"], 4), round(r["hmm_top1"], 4), round(r["hmm_continuity"], 4), flush=True)
grid = pd.DataFrame(grid)
grid.to_csv(os.path.join(res_dir, "hmm_grid_train_oof.csv"), index=False)
b = grid.sort_values("hmm_top1", ascending=False).iloc[0]
best = {"beta": float(b.beta), "w_emission": float(b.w_emission), "vmax": 50.0, "back_tol": 15.0, "break_penalty": 30.0,
        "tuned_on": f"out-of-fold LightGBM probabilities, {len(tune)} train trips", "train_oof_hmm_top1": float(b.hmm_top1)}
with open(os.path.join(cfg["artifacts_dir"], "hmm_config.json"), "w") as f:
    json.dump(best, f, indent=2)
print("best", best, flush=True)

piv = grid.pivot(index="beta", columns="w_emission", values="hmm_top1")
plt.figure(figsize=(6, 4)); plt.imshow(piv.values, cmap="viridis", aspect="auto")
plt.xticks(range(len(piv.columns)), piv.columns); plt.yticks(range(len(piv.index)), piv.index)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        plt.text(j, i, f"{piv.values[i, j]:.3f}", ha="center", va="center", color="w")
plt.xlabel("emission weight"); plt.ylabel("beta (m)"); plt.title("HMM top-1 on train OOF probabilities")
plt.colorbar(); plt.tight_layout(); plt.savefig(os.path.join(fig_dir, "hmm_grid.png"), dpi=120); plt.close()

lens = [len(d["y"]) for d in test]
splits = np.cumsum(lens)[:-1]
models = [m for m in list(REGISTRY) + ["LightGBM_tuned"] if os.path.exists(os.path.join(res_dir, f"probs_{m}.npy"))]
comp = []
for name in models:
    probs = np.split(np.load(os.path.join(res_dir, f"probs_{name}.npy")), splits)
    r, df = agg([run(cfg["student_test_dir"], d, p, best) for d, p in zip(test, probs)])
    comp.append({"model": name, **r})
    df.to_csv(os.path.join(res_dir, f"hmm_per_trip_{name}.csv"), index=False)
    print("test", name, {k: round(v, 4) for k, v in r.items()}, flush=True)
comp = pd.DataFrame(comp)
comp.to_csv(os.path.join(res_dir, "hmm_vs_classifier_student_test.csv"), index=False)

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
x = np.arange(len(comp))
for a, m, t in zip(ax, ("top1", "continuity", "seq_f1"), ("Top-1 accuracy", "Route continuity", "Sequence (route-arc) F1")):
    a.bar(x - 0.2, comp[f"clf_{m}"], 0.4, label="classifier only"); a.bar(x + 0.2, comp[f"hmm_{m}"], 0.4, label="classifier + HMM")
    a.set_xticks(x); a.set_xticklabels(comp.model, rotation=45, ha="right"); a.set_title(t + " (student test)")
    a.set_ylim(max(0, min(comp[f"clf_{m}"].min(), comp[f"hmm_{m}"].min()) - 0.05), 1.0); a.legend()
plt.tight_layout(); plt.savefig(os.path.join(fig_dir, "hmm_vs_classifier.png"), dpi=120); plt.close()

df = pd.read_csv(os.path.join(res_dir, "hmm_per_trip_LightGBM_tuned.csv"))
plt.figure(figsize=(9, 4)); x = np.arange(len(df))
plt.bar(x - 0.2, df.clf_top1, 0.4, label="classifier only"); plt.bar(x + 0.2, df.hmm_top1, 0.4, label="classifier + HMM")
plt.xticks(x, df.trip_id, rotation=45); plt.ylim(0.6, 1); plt.ylabel("top-1"); plt.title("Per-trip top-1, tuned LightGBM (student test)"); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(fig_dir, "hmm_per_trip.png"), dpi=120); plt.close()
