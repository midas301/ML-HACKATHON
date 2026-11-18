import os
import sys
import json
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap import config
from pathsnap.pipeline import cached
from pathsnap.metrics import top1_choice
from pathsnap.models import REGISTRY

cfg = config.load()
res_dir, fig_dir = cfg["results_dir"], cfg["figures_dir"]
names = [m for m in REGISTRY if os.path.exists(os.path.join(res_dir, f"summary_{m}.json"))]
S = pd.DataFrame([json.load(open(os.path.join(res_dir, f"summary_{m}.json"))) for m in names])
bo = json.load(open(os.path.join(res_dir, "bayes_opt_summary.json")))
t = bo["tuned"]
S = pd.concat([S, pd.DataFrame([{"model": "LightGBM (tuned)", "family": "Boosting", "config": "Optuna TPE best params",
                                  **{k: v for k, v in t.items()}}])], ignore_index=True)
hmm = pd.read_csv(os.path.join(res_dir, "hmm_vs_classifier_student_test.csv")).set_index("model")
key = {"LightGBM (tuned)": "LightGBM_tuned"}
S["test_hmm_top1"] = [hmm.loc[key.get(m, m), "hmm_top1"] if key.get(m, m) in hmm.index else np.nan for m in S.model]
S["test_hmm_continuity"] = [hmm.loc[key.get(m, m), "hmm_continuity"] if key.get(m, m) in hmm.index else np.nan for m in S.model]
tc = os.path.join(res_dir, "timing_clean.csv")
if os.path.exists(tc):
    T = pd.read_csv(tc).set_index("model")
    S["train_seconds"] = [T.loc[m, "train_seconds"] for m in S.model]
    S["inference_ms_per_1k_obs"] = [T.loc[m, "inference_ms_per_1k_obs"] for m in S.model]
cols = ["model", "family", "config", "cv_macro_f1_mean", "cv_macro_f1_std", "cv_top1_mean", "cv_top1_std", "test_top1", "test_macro_f1",
        "test_macro_precision", "test_macro_recall", "test_log_loss", "test_continuity", "test_seq_f1", "test_hmm_top1", "test_hmm_continuity",
        "train_top1", "train_seconds", "inference_ms_per_1k_obs"]
S[cols].to_csv(os.path.join(res_dir, "model_comparison.csv"), index=False)
print(S[cols].drop(columns=["config"]).round(4).to_string())

x = np.arange(len(S))
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(x, S.cv_macro_f1_mean, yerr=S.cv_macro_f1_std, capsize=4, color="tab:blue")
ax.set_xticks(x); ax.set_xticklabels(S.model, rotation=30, ha="right"); ax.set_ylim(0.8, 1)
ax.set_ylabel("Macro F1"); ax.set_title("5-fold GroupKFold CV Macro F1 (train trips), mean ± std")
for i, v in enumerate(S.cv_macro_f1_mean):
    ax.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(fig_dir, "cmp_cv_macro_f1.png"), dpi=120); plt.close()

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(x - 0.2, S.test_top1, 0.4, label="Top-1 accuracy"); ax.bar(x + 0.2, S.test_macro_f1, 0.4, label="Macro F1")
ax.set_xticks(x); ax.set_xticklabels(S.model, rotation=30, ha="right"); ax.set_ylim(0.75, 1); ax.legend()
ax.set_title("Student-test Top-1 accuracy and Macro F1 (classifier only)")
plt.tight_layout(); plt.savefig(os.path.join(fig_dir, "cmp_test_top1_f1.png"), dpi=120); plt.close()

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(x, S.test_log_loss, color="tab:red"); ax.set_yscale("log")
ax.set_xticks(x); ax.set_xticklabels(S.model, rotation=30, ha="right"); ax.set_title("Student-test log loss (lower is better, log scale)")
for i, v in enumerate(S.test_log_loss):
    ax.text(i, v * 1.05, f"{v:.3f}", ha="center", fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(fig_dir, "cmp_log_loss.png"), dpi=120); plt.close()

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(S.inference_ms_per_1k_obs, S.test_top1)
for _, r in S.iterrows():
    ax.annotate(r.model, (r.inference_ms_per_1k_obs, r.test_top1), fontsize=8)
ax.set_xscale("log"); ax.set_xlabel("inference ms per 1000 GPS observations (log)"); ax.set_ylabel("student-test top-1")
ax.set_title("Accuracy vs inference cost")
plt.tight_layout(); plt.savefig(os.path.join(fig_dir, "cmp_speed_accuracy.png"), dpi=120); plt.close()

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(x - 0.2, S.train_top1, 0.4, label="train (20 sampled train trips)"); ax.bar(x + 0.2, S.cv_top1_mean, 0.4, label="CV held-out folds")
ax.set_xticks(x); ax.set_xticklabels(S.model, rotation=30, ha="right"); ax.set_ylim(0.75, 1); ax.legend(); ax.set_title("Train vs CV top-1 (over/underfitting check)")
plt.tight_layout(); plt.savefig(os.path.join(fig_dir, "cmp_train_vs_cv.png"), dpi=120); plt.close()

test = cached("student_test", cfg["student_test_dir"], cfg)
lens = [len(d["y"]) for d in test]
splits = np.cumsum(lens)[:-1]
plot_names = names + ["LightGBM_tuned"]
fig, axes = plt.subplots(2, (len(plot_names) + 1) // 2, figsize=(4 * ((len(plot_names) + 1) // 2), 8))
for a, name in zip(axes.ravel(), plot_names):
    probs = np.split(np.load(os.path.join(res_dir, f"probs_{name}.npy")), splits)
    ys, cs = [], []
    for d, p in zip(test, probs):
        _, c = top1_choice(d["pid"], p, d["n_points"], d["aid"])
        ys.append(d["y"][d["row_valid"]]); cs.append(c[d["row_valid"]])
    cm = confusion_matrix(np.concatenate(ys), np.concatenate(cs))
    ConfusionMatrixDisplay(cm, display_labels=["not ref", "ref arc"]).plot(ax=a, colorbar=False, values_format="d")
    a.set_title(name)
for a in axes.ravel()[len(plot_names):]:
    a.axis("off")
plt.suptitle("Candidate-level confusion matrices, top-1 decision rule (student test)")
plt.tight_layout(); plt.savefig(os.path.join(fig_dir, "cmp_confusion_matrices.png"), dpi=110); plt.close()
