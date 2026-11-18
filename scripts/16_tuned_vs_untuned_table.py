import os
import sys
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap import config
from pathsnap.models import LGBM_DEFAULT

cfg = config.load()
tuned = json.load(open(os.path.join(cfg["artifacts_dir"], "best_lgbm_params.json")))
untuned = dict(LGBM_DEFAULT)
untuned.setdefault("reg_alpha", 0.0)
untuned.setdefault("subsample_freq", 0)
bo = json.load(open(os.path.join(cfg["results_dir"], "bayes_opt_summary.json")))
bo["untuned"] = json.load(open(os.path.join(cfg["results_dir"], "summary_LightGBM_dir.json")))
bo["tuned"] = json.load(open(os.path.join(cfg["results_dir"], "summary_LightGBM_dir_tuned.json")))
hmm = pd.read_csv(os.path.join(cfg["results_dir"], "hmm_vs_classifier_student_test.csv")).set_index("model")
hmm = hmm.loc[["LightGBM_dir", "LightGBM_dir_tuned"]].rename(index={"LightGBM_dir": "LightGBM", "LightGBM_dir_tuned": "LightGBM_tuned"})
tc = pd.DataFrame({"train_seconds": [bo["untuned"]["train_seconds"], bo["tuned"]["train_seconds"]],
                   "inference_ms_per_1k_obs": [bo["untuned"]["inference_ms_per_1k_obs"], bo["tuned"]["inference_ms_per_1k_obs"]]},
                  index=["LightGBM", "LightGBM (tuned)"])

desc = {
    "n_estimators": "number of trees", "learning_rate": "step size per tree", "num_leaves": "max leaves per tree",
    "min_child_samples": "min rows per leaf", "subsample": "row fraction per tree", "subsample_freq": "row subsampling on (1) / off (0)",
    "colsample_bytree": "feature fraction per tree", "reg_lambda": "L2 regularisation", "reg_alpha": "L1 regularisation",
    "random_state": "seed (fixed)", "n_jobs": "CPU threads (fixed)", "verbose": "logging (fixed)",
}
order = ["n_estimators", "learning_rate", "num_leaves", "min_child_samples", "subsample", "subsample_freq",
         "colsample_bytree", "reg_lambda", "reg_alpha", "random_state", "n_jobs", "verbose"]


def fmt(v):
    if isinstance(v, float) and not float(v).is_integer():
        return f"{v:.4f}"
    return str(int(v)) if isinstance(v, (int, float)) else str(v)


prow = []
for k in order:
    a, b = untuned.get(k), tuned.get(k)
    note = "Optuna searched" if k in bo["search"]["best_params"] else ("set by us (enables subsample)" if k == "subsample_freq" else "not tuned")
    prow.append([k, desc[k], fmt(a), fmt(b), "changed" if a != b else "same", note])

u, t = bo["untuned"], bo["tuned"]
mrow = [
    ["CV top-1 accuracy (5-fold, train)", f"{u['cv_top1_mean']:.4f} ± {u['cv_top1_std']:.3f}", f"{t['cv_top1_mean']:.4f} ± {t['cv_top1_std']:.3f}", f"{100 * (t['cv_top1_mean'] - u['cv_top1_mean']):+.2f} pts"],
    ["CV macro F1", f"{u['cv_macro_f1_mean']:.4f}", f"{t['cv_macro_f1_mean']:.4f}", f"{100 * (t['cv_macro_f1_mean'] - u['cv_macro_f1_mean']):+.2f} pts"],
    ["CV log loss (lower better)", f"{u['cv_log_loss_mean']:.4f}", f"{t['cv_log_loss_mean']:.4f}", f"{100 * (t['cv_log_loss_mean'] / u['cv_log_loss_mean'] - 1):+.1f}%"],
    ["Student-test top-1 (classifier)", f"{u['test_top1']:.4f}", f"{t['test_top1']:.4f}", f"{100 * (t['test_top1'] - u['test_top1']):+.2f} pts"],
    ["Student-test top-1 (with HMM)", f"{hmm.loc['LightGBM', 'hmm_top1']:.4f}", f"{hmm.loc['LightGBM_tuned', 'hmm_top1']:.4f}", f"{100 * (hmm.loc['LightGBM_tuned', 'hmm_top1'] - hmm.loc['LightGBM', 'hmm_top1']):+.2f} pts"],
    ["Student-test log loss", f"{u['test_log_loss']:.4f}", f"{t['test_log_loss']:.4f}", f"{100 * (t['test_log_loss'] / u['test_log_loss'] - 1):+.1f}%"],
    ["Training time (s, all train data)", f"{tc.loc['LightGBM', 'train_seconds']:.1f}", f"{tc.loc['LightGBM (tuned)', 'train_seconds']:.1f}", f"{tc.loc['LightGBM (tuned)', 'train_seconds'] / tc.loc['LightGBM', 'train_seconds']:.1f}x"],
    ["Inference (ms per 1000 GPS points)", f"{tc.loc['LightGBM', 'inference_ms_per_1k_obs']:.0f}", f"{tc.loc['LightGBM (tuned)', 'inference_ms_per_1k_obs']:.0f}", f"{tc.loc['LightGBM (tuned)', 'inference_ms_per_1k_obs'] / tc.loc['LightGBM', 'inference_ms_per_1k_obs']:.1f}x"],
]

fig = plt.figure(figsize=(14, 9.6))
fig.suptitle("Final LightGBM (29 features): untuned vs tuned (Bayesian optimisation, Optuna TPE, 30 trials)", fontsize=15, y=0.985)
ax1 = fig.add_axes([0.03, 0.40, 0.94, 0.53]); ax1.axis("off")
ax1.set_title("Hyperparameters   (untuned: pathsnap/models.py LGBM_DEFAULT  |  tuned: artifacts/best_lgbm_params.json)", fontsize=11, loc="left")
tb = ax1.table(cellText=prow, colLabels=["Hyperparameter", "Meaning", "Untuned", "Tuned", "Diff", "How set"], loc="upper center", cellLoc="left",
               colWidths=[0.17, 0.25, 0.10, 0.10, 0.08, 0.24])
tb.auto_set_font_size(False); tb.set_fontsize(10.5); tb.scale(1, 1.75)
for (r, c), cell in tb.get_celld().items():
    cell.set_edgecolor("#bbbbbb")
    if r == 0:
        cell.set_facecolor("#2b2d42"); cell.get_text().set_color("white"); cell.get_text().set_weight("bold")
    elif prow[r - 1][4] == "changed":
        cell.set_facecolor("#fff3cd" if c not in (3,) else "#ffe08a")
        if c == 3:
            cell.get_text().set_weight("bold")
    else:
        cell.set_facecolor("#f4f4f4"); cell.get_text().set_color("#666666")

ax2 = fig.add_axes([0.03, 0.03, 0.94, 0.33]); ax2.axis("off")
ax2.set_title("What tuning bought   (results/bayes_opt_summary.json, results/hmm_vs_classifier_student_test.csv, results/timing_clean.csv)", fontsize=11, loc="left")
tb2 = ax2.table(cellText=mrow, colLabels=["Metric", "Untuned", "Tuned", "Change"], loc="upper center", cellLoc="left", colWidths=[0.38, 0.2, 0.2, 0.16])
tb2.auto_set_font_size(False); tb2.set_fontsize(10.5); tb2.scale(1, 1.75)
for (r, c), cell in tb2.get_celld().items():
    cell.set_edgecolor("#bbbbbb")
    if r == 0:
        cell.set_facecolor("#2b2d42"); cell.get_text().set_color("white"); cell.get_text().set_weight("bold")
    elif c == 3:
        txt = mrow[r - 1][3]; name = mrow[r - 1][0]
        worse = ("time" in name.lower() or "inference" in name.lower()) or ("HMM" in name and txt.startswith("-"))
        tie = "pts" in txt and abs(float(txt.split()[0])) < 0.05
        better = not tie and ((("log loss" in name) and txt.startswith("-")) or (("top-1" in name or "F1" in name) and txt.startswith("+")))
        worse = worse and not tie
        cell.set_facecolor("#f8d7da" if worse else ("#d4edda" if better else "#f4f4f4"))
fig.text(0.03, 0.012, f"Search: {bo['search']['n_trials']} trials on the 22-feature set, objective = mean 3-fold GroupKFold top-1 on train trips only (best {bo['search']['best_value']:.4f}); best values reused for the 29-feature model. "
         "Yellow = changed by tuning. Green = better, red = worse, grey = no meaningful change (< 0.05 pts).", fontsize=9.5, color="#444444")
for ext in ("png", "pdf"):
    plt.savefig(os.path.join(cfg["figures_dir"], f"tuned_vs_untuned.{ext}"), dpi=150)
print("saved")
