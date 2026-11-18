import os
import sys
import json
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
from pathsnap.registry_view import MODEL_KEYS, DISPLAY, BASE_KEYS

cfg = config.load()
R, F = cfg["results_dir"], cfg["figures_dir"]
keys = [k for k in MODEL_KEYS if os.path.exists(os.path.join(R, f"summary_{k}.json"))]
S = pd.DataFrame([json.load(open(os.path.join(R, f"summary_{k}.json"))) | {"key": k} for k in keys])
S["name"] = [DISPLAY[k] for k in S.key]
S["n_features"] = S.get("n_features", pd.Series([22] * len(S))).fillna(22).astype(int)
hmm = pd.read_csv(os.path.join(R, "hmm_vs_classifier_student_test.csv")).set_index("model")
S["test_hmm_top1"] = [hmm.loc[k, "hmm_top1"] if k in hmm.index else np.nan for k in S.key]
S["test_hmm_continuity"] = [hmm.loc[k, "hmm_continuity"] if k in hmm.index else np.nan for k in S.key]
S["test_hmm_seq_f1"] = [hmm.loc[k, "hmm_seq_f1"] if k in hmm.index else np.nan for k in S.key]
tc_path = os.path.join(R, "timing_clean.csv")
if os.path.exists(tc_path):
    T = pd.read_csv(tc_path).set_index("model")
    alias = {"LogReg": "LogReg", "KNN": "KNN", "RandomForest": "RandomForest", "MLP": "MLP", "LightGBM": "LightGBM", "LightGBM_tuned22": "LightGBM (tuned)"}
    for i, k in enumerate(S.key):
        if alias.get(k) in T.index:
            S.loc[i, "train_seconds"] = T.loc[alias[k], "train_seconds"]
            S.loc[i, "inference_ms_per_1k_obs"] = T.loc[alias[k], "inference_ms_per_1k_obs"]
cols = ["key", "name", "family", "n_features", "config", "cv_macro_f1_mean", "cv_macro_f1_std", "cv_top1_mean", "cv_top1_std", "cv_log_loss_mean",
        "test_top1", "test_accuracy_at_0.5", "test_macro_precision", "test_macro_recall", "test_macro_f1", "test_log_loss", "test_continuity", "test_seq_f1",
        "test_hmm_top1", "test_hmm_continuity", "test_hmm_seq_f1", "train_top1", "train_seconds", "inference_ms_per_1k_obs"]
S[cols].to_csv(os.path.join(R, "model_comparison.csv"), index=False)
print(S[cols].drop(columns=["config", "name"]).round(4).to_string())

colors = ["#9aa8bd" if k in BASE_KEYS else ("#c8102e" if k == "LightGBM_dir_tuned" else "#14284b") for k in S.key]
x = np.arange(len(S))
lbl = [n.replace(" (", "\n(") for n in S.name]


def sep(a):
    a.axvline(len([k for k in S.key if k in BASE_KEYS]) - 0.5, color="0.5", ls="--", lw=1)


fig, ax = plt.subplots(figsize=(13, 4.6))
ax.bar(x, S.cv_macro_f1_mean, yerr=S.cv_macro_f1_std, capsize=4, color=colors)
for i, v in enumerate(S.cv_macro_f1_mean):
    ax.text(i, 0.856, f"{v:.3f}", ha="center", fontsize=9, color="white", weight="bold")
ax.set_xticks(x); ax.set_xticklabels(lbl, fontsize=8); ax.set_ylim(0.85, 1.0); sep(ax)
ax.set_ylabel("Macro F1"); ax.set_title("5-fold GroupKFold cross-validation Macro F1 (80 train trips), mean ± sd\nleft of dashed line: 5-model comparison on identical 22 features; right: LightGBM refinements")
plt.tight_layout(); plt.savefig(os.path.join(F, "cmp_cv_macro_f1.png"), dpi=120); plt.close()

fig, ax = plt.subplots(figsize=(13, 4.6))
ax.bar(x - 0.27, S.test_top1, 0.27, label="Top-1 accuracy (classifier)", color="#9aa8bd")
ax.bar(x, S.test_hmm_top1, 0.27, label="Top-1 accuracy (with HMM)", color="#14284b")
ax.bar(x + 0.27, S.test_macro_f1, 0.27, label="Macro F1", color="#2f5bff")
ax.set_xticks(x); ax.set_xticklabels(lbl, fontsize=8); ax.set_ylim(0.8, 1.0); ax.legend(fontsize=8, loc="lower right"); sep(ax)
ax.set_title("Student-test (10 trips) Top-1 accuracy and Macro F1")
plt.tight_layout(); plt.savefig(os.path.join(F, "cmp_test_top1_f1.png"), dpi=120); plt.close()

fig, ax = plt.subplots(figsize=(13, 4.6))
ax.bar(x, S.test_log_loss, color=colors); ax.set_yscale("log"); sep(ax)
for i, v in enumerate(S.test_log_loss):
    ax.text(i, v * 1.05, f"{v:.3f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(lbl, fontsize=8); ax.set_title("Student-test log loss (lower is better, log scale)")
plt.tight_layout(); plt.savefig(os.path.join(F, "cmp_log_loss.png"), dpi=120); plt.close()

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.scatter(S.inference_ms_per_1k_obs, S.test_hmm_top1, c=colors, s=60)
for _, r in S.iterrows():
    ax.annotate(r["name"], (r.inference_ms_per_1k_obs, r.test_hmm_top1), xytext=(5, 4), textcoords="offset points", fontsize=8)
ax.set_xscale("log"); ax.set_xlabel("classifier inference, ms per 1000 GPS points (log)"); ax.set_ylabel("student-test top-1 with HMM")
ax.set_title("Accuracy vs inference cost")
plt.tight_layout(); plt.savefig(os.path.join(F, "cmp_speed_accuracy.png"), dpi=120); plt.close()

fig, ax = plt.subplots(figsize=(13, 4.6))
ax.bar(x - 0.2, S.train_top1, 0.4, label="train (20 sampled train trips, in-sample)", color="#9aa8bd")
ax.bar(x + 0.2, S.cv_top1_mean, 0.4, label="CV held-out folds", color="#14284b")
ax.set_xticks(x); ax.set_xticklabels(lbl, fontsize=8); ax.set_ylim(0.75, 1.0); ax.legend(fontsize=8); sep(ax)
ax.set_title("Train vs cross-validation top-1 (over/underfitting check)")
plt.tight_layout(); plt.savefig(os.path.join(F, "cmp_train_vs_cv.png"), dpi=120); plt.close()

test = cached("student_test", cfg["student_test_dir"], cfg)
splits = np.cumsum([len(d["y"]) for d in test])[:-1]
plot_keys = [k for k in S.key if os.path.exists(os.path.join(R, f"probs_{k}.npy"))]
nc = 4
nr = int(np.ceil(len(plot_keys) / nc))
fig, axes = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.9 * nr))
for a, k in zip(axes.ravel(), plot_keys):
    probs = np.split(np.load(os.path.join(R, f"probs_{k}.npy")), splits)
    ys, cs = [], []
    for d, p in zip(test, probs):
        _, c = top1_choice(d["pid"], p, d["n_points"], d["aid"])
        ys.append(d["y"][d["row_valid"]]); cs.append(c[d["row_valid"]])
    cm = confusion_matrix(np.concatenate(ys), np.concatenate(cs))
    ConfusionMatrixDisplay(cm, display_labels=["not ref", "ref arc"]).plot(ax=a, colorbar=False, values_format="d")
    a.set_title(DISPLAY[k], fontsize=9)
for a in axes.ravel()[len(plot_keys):]:
    a.axis("off")
plt.suptitle("Candidate-level confusion matrices, top-1 decision rule (student test)")
plt.tight_layout(); plt.savefig(os.path.join(F, "cmp_confusion_matrices.png"), dpi=110); plt.close()
