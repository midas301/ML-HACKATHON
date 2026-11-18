import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap import config

cfg = config.load()
R, F = cfg["results_dir"], cfg["figures_dir"]
ab = json.load(open(os.path.join(R, "feature_ablation_direction.json")))
o, n = ab["X_old"], ab["X_new"]

fig, ax = plt.subplots(1, 3, figsize=(18, 4.6))
labels = ["CV top-1\n(5-fold, train)", "Student-test top-1\n(classifier)", "Student-test top-1\n(with HMM)", "CV macro F1"]
ov = [o["cv_top1_mean"], o["test_top1"], o["test_hmm_top1"], o["cv_macro_f1_mean"]]
nv = [n["cv_top1_mean"], n["test_top1"], n["test_hmm_top1"], n["cv_macro_f1_mean"]]
x = np.arange(len(labels))
ax[0].bar(x - 0.2, ov, 0.4, label=f"22 features", color="#9aa8bd"); ax[0].bar(x + 0.2, nv, 0.4, label="29 features (+ direction)", color="#c8102e")
for i, (a, b) in enumerate(zip(ov, nv)):
    ax[0].text(i + 0.2, b + 0.004, f"+{100 * (b - a):.1f}", ha="center", fontsize=9)
ax[0].set_xticks(x); ax[0].set_xticklabels(labels, fontsize=9); ax[0].set_ylim(0.8, 1); ax[0].legend(fontsize=8)
ax[0].set_title("Feature ablation: same tuned LightGBM, same folds, same HMM", fontsize=10)

bands = list(o["test_hmm_top1_by_speed"].keys())
x = np.arange(len(bands))
ax[1].bar(x - 0.2, [o["test_hmm_top1_by_speed"][b] for b in bands], 0.4, label="22 features", color="#9aa8bd")
ax[1].bar(x + 0.2, [n["test_hmm_top1_by_speed"][b] for b in bands], 0.4, label="29 features", color="#c8102e")
ax[1].set_xticks(x); ax[1].set_xticklabels([f"{b} m/s" for b in bands]); ax[1].set_ylim(0.4, 1.02); ax[1].legend(fontsize=8)
ax[1].set_xlabel("GPS speed band"); ax[1].set_ylabel("top-1 with HMM"); ax[1].set_title("Student-test accuracy by speed: the gain is at low speed", fontsize=10)

ax[2].bar([0, 1], [o["test_hmm_wrong_direction_errors"], n["test_hmm_wrong_direction_errors"]], color=["#9aa8bd", "#c8102e"])
for i, v in enumerate([o["test_hmm_wrong_direction_errors"], n["test_hmm_wrong_direction_errors"]]):
    ax[2].text(i, v + 10, str(v), ha="center")
ax[2].set_xticks([0, 1]); ax[2].set_xticklabels(["22 features", "29 features"]); ax[2].set_ylabel("points")
ax[2].set_title("Student-test points matched to the right road\nbut the wrong direction (with HMM)", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(F, "feature_ablation_direction.png"), dpi=120); plt.close()

trips = list(o["per_trip_top1"].keys())
x = np.arange(len(trips))
plt.figure(figsize=(10, 4))
plt.bar(x - 0.2, [o["per_trip_top1"][t] for t in trips], 0.4, label="22 features", color="#9aa8bd")
plt.bar(x + 0.2, [n["per_trip_top1"][t] for t in trips], 0.4, label="29 features", color="#c8102e")
plt.xticks(x, trips, rotation=45); plt.ylim(0.6, 1); plt.ylabel("top-1 (classifier)"); plt.legend()
plt.title("Per-trip student-test top-1, before and after direction features (no trip got worse)")
plt.tight_layout(); plt.savefig(os.path.join(F, "feature_ablation_per_trip.png"), dpi=120); plt.close()
print("ok")
