import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
import optuna
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

cfg = config.load()
N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
train = cached("train", cfg["train_dir"], cfg)
test = cached("student_test", cfg["student_test_dir"], cfg)
groups = np.arange(len(train))
res_dir, fig_dir = cfg["results_dir"], cfg["figures_dir"]


def rows(trips):
    return (np.vstack([d["X"][d["row_valid"]] for d in trips]),
            np.concatenate([d["y"][d["row_valid"]] for d in trips]))


def cv(params, n_splits, keep_oof=False):
    scores, f1s, lls, oof = [], [], [], {}
    for tr_idx, va_idx in GroupKFold(n_splits=n_splits).split(groups, groups=groups):
        X, y = rows([train[i] for i in tr_idx])
        m = LGBMClassifier(**params).fit(X, y)
        va = [train[i] for i in va_idx]
        probs = [m.predict_proba(d["X"])[:, 1] for d in va]
        r, _, _ = evaluate(va, probs)
        scores.append(r["top1"]); f1s.append(r["macro_f1"]); lls.append(r["log_loss"])
        if keep_oof:
            for i, p in zip(va_idx, probs):
                oof[train[i]["trip_id"]] = p
    return np.array(scores), np.array(f1s), np.array(lls), oof


def objective(trial):
    params = dict(LGBM_DEFAULT)
    params.update(
        n_estimators=trial.suggest_int("n_estimators", 100, 800, log=True),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 200, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        subsample_freq=1,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    )
    t0 = time.time()
    s, f, l, _ = cv(params, 3)
    trial.set_user_attr("cv3_macro_f1", float(f.mean()))
    trial.set_user_attr("cv3_log_loss", float(l.mean()))
    trial.set_user_attr("seconds", time.time() - t0)
    return float(s.mean())


t_start = time.time()
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED), study_name="lgbm_top1")
study.enqueue_trial({k: LGBM_DEFAULT[k] for k in ("n_estimators", "learning_rate", "num_leaves", "min_child_samples", "subsample", "colsample_bytree")} | {"reg_lambda": 1e-3, "reg_alpha": 1e-3})
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
search_seconds = time.time() - t_start
trials = study.trials_dataframe()
trials.to_csv(os.path.join(res_dir, "bayes_opt_trials.csv"), index=False)

best = dict(LGBM_DEFAULT); best.update(study.best_params); best["subsample_freq"] = 1
with open(os.path.join(cfg["artifacts_dir"], "best_lgbm_params.json"), "w") as f:
    json.dump(best, f, indent=2)

out = {}
for tag, params in (("untuned", dict(LGBM_DEFAULT)), ("tuned", best)):
    s, f1, ll, oof = cv(params, 5, keep_oof=True)
    X, y = rows(train)
    t0 = time.time(); m = LGBMClassifier(**params).fit(X, y); tt = time.time() - t0
    t0 = time.time(); probs = [m.predict_proba(d["X"])[:, 1] for d in test]; inf = time.time() - t0
    r, per_trip, _ = evaluate(test, probs, root=cfg["student_test_dir"])
    n_obs = sum(d["n_points"] for d in test)
    out[tag] = {"cv_top1_mean": s.mean(), "cv_top1_std": s.std(ddof=1), "cv_macro_f1_mean": f1.mean(), "cv_macro_f1_std": f1.std(ddof=1),
                "cv_log_loss_mean": ll.mean(), **{"test_" + k: v for k, v in r.items()},
                "train_seconds": tt, "inference_seconds": inf, "inference_ms_per_1k_obs": 1e6 * inf / n_obs}
    joblib.dump(oof, os.path.join(cfg["artifacts_dir"], f"oof_lgbm_{tag}.joblib"))
    if tag == "tuned":
        joblib.dump(m, os.path.join(cfg["artifacts_dir"], "final_model.joblib"), compress=3)
        np.save(os.path.join(res_dir, "probs_LightGBM_tuned.npy"), np.concatenate(probs))
        pd.DataFrame(per_trip).to_csv(os.path.join(res_dir, "student_test_per_trip_LightGBM_tuned.csv"), index=False)
        imp = pd.Series(m.booster_.feature_importance("gain"), index=__import__("pathsnap.candidates", fromlist=["FEATURES"]).FEATURES).sort_values()
        imp.to_csv(os.path.join(res_dir, "feature_importance_gain.csv"))
        imp.plot.barh(figsize=(7, 7)); plt.title("Tuned LightGBM feature importance (gain)"); plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "feature_importance.png"), dpi=120); plt.close()
    print(tag, json.dumps({k: round(v, 4) for k, v in out[tag].items()}), flush=True)

out["search"] = {"n_trials": N_TRIALS, "cv_folds_in_search": 3, "objective": "mean GroupKFold(3) observation-level top-1 accuracy",
                 "sampler": "Optuna TPE (seed 42)", "best_value": study.best_value, "best_params": study.best_params,
                 "search_seconds": search_seconds}
with open(os.path.join(res_dir, "bayes_opt_summary.json"), "w") as f:
    json.dump(out, f, indent=2)

v = trials["value"].values
plt.figure(figsize=(8, 4))
plt.plot(v, "o", alpha=0.5, label="trial"); plt.plot(np.maximum.accumulate(v), "r-", label="best so far")
plt.xlabel("trial"); plt.ylabel("CV(3) top-1 accuracy"); plt.title("Bayesian optimisation history (TPE)"); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(fig_dir, "bayes_opt_history.png"), dpi=120); plt.close()
try:
    imp = optuna.importance.get_param_importances(study)
    pd.Series(imp).sort_values().plot.barh(figsize=(6, 4)); plt.title("Hyperparameter importance (fANOVA)")
    plt.tight_layout(); plt.savefig(os.path.join(fig_dir, "bayes_opt_param_importance.png"), dpi=120); plt.close()
    pd.Series(imp).to_csv(os.path.join(res_dir, "bayes_opt_param_importance.csv"))
except Exception as e:
    print("importance failed", e)
