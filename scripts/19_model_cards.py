import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap import config
from pathsnap.pipeline import cached
from pathsnap.metrics import top1_choice
from pathsnap.registry_view import MODEL_KEYS, DISPLAY

cfg = config.load()
R, F = cfg["results_dir"], cfg["figures_dir"]
test = cached("student_test", cfg["student_test_dir"], cfg)
splits = np.cumsum([len(d["y"]) for d in test])[:-1]
for k in MODEL_KEYS:
    pf, cvf, hf = (os.path.join(R, f) for f in (f"probs_{k}.npy", f"cv_{k}.csv", f"hmm_per_trip_{k}.csv"))
    if not (os.path.exists(pf) and os.path.exists(hf)):
        continue
    probs = np.split(np.load(pf), splits)
    ys, ps, cs = [], [], []
    for d, p in zip(test, probs):
        _, c = top1_choice(d["pid"], p, d["n_points"], d["aid"])
        rv = d["row_valid"]
        ys.append(d["y"][rv]); ps.append(p[rv]); cs.append(c[rv])
    y, p, c = np.concatenate(ys), np.concatenate(ps), np.concatenate(cs)
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.3))
    if os.path.exists(cvf):
        cv = pd.read_csv(cvf)
        ax[0].bar(np.arange(len(cv)) + 1, cv.top1, color="#14284b")
        ax[0].axhline(cv.top1.mean(), color="#c8102e", ls="--", lw=1)
        for i, v in enumerate(cv.top1):
            ax[0].text(i + 1, v + 0.004, f"{v:.3f}", ha="center", fontsize=8)
        ax[0].set_ylim(max(0.6, cv.top1.min() - 0.05), 1); ax[0].set_xlabel("fold")
        ax[0].set_title(f"CV top-1 per fold (mean {cv.top1.mean():.4f}, sd {cv.top1.std():.3f})", fontsize=10)
    else:
        ax[0].text(0.5, 0.5, "no CV record", ha="center"); ax[0].axis("off")
    h = pd.read_csv(hf)
    xx = np.arange(len(h))
    ax[1].bar(xx - 0.2, h.clf_top1, 0.4, label="classifier", color="#9aa8bd"); ax[1].bar(xx + 0.2, h.hmm_top1, 0.4, label="with HMM", color="#14284b")
    ax[1].set_xticks(xx); ax[1].set_xticklabels([t[-2:] for t in h.trip_id.astype(str).str.zfill(8)], fontsize=8); ax[1].set_xlabel("student-test trip (last 2 digits)")
    ax[1].set_ylim(0.6, 1); ax[1].legend(fontsize=8); ax[1].set_title("Per-trip student-test top-1", fontsize=10)
    fy, fx = calibration_curve(y, np.clip(p, 0, 1), n_bins=10, strategy="uniform")
    ax[2].plot([0, 1], [0, 1], color="0.7", ls="--"); ax[2].plot(fx, fy, "o-", color="#c8102e")
    ax[2].set_xlabel("predicted probability"); ax[2].set_ylabel("observed fraction correct"); ax[2].set_title("Calibration on student test (probability quality)", fontsize=10)
    cm = confusion_matrix(y, c, normalize="true")
    ax[3].imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax[3].text(j, i, f"{cm[i, j]:.3f}", ha="center", va="center", color="white" if cm[i, j] > 0.5 else "black")
    ax[3].set_xticks([0, 1]); ax[3].set_xticklabels(["not ref", "ref arc"]); ax[3].set_yticks([0, 1]); ax[3].set_yticklabels(["not ref", "ref arc"])
    ax[3].set_xlabel("predicted (top-1 rule)"); ax[3].set_ylabel("true"); ax[3].set_title("Confusion matrix (row-normalised)", fontsize=10)
    plt.suptitle(DISPLAY[k], fontsize=13, weight="bold")
    plt.tight_layout(); plt.savefig(os.path.join(F, f"model_card_{k}.png"), dpi=110); plt.close()
    print("card", k, flush=True)
