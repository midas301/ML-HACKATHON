import os
import sys
import json
import time
import glob
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap import config

cfg = config.load()
R = cfg["results_dir"]
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)

from pathsnap.registry_view import MODEL_KEYS, DISPLAY
rows = []
for k in MODEL_KEYS:
    f = os.path.join(R, f"summary_{k}.json")
    if os.path.exists(f):
        rows.append(json.load(open(f)) | {"model": DISPLAY[k], "key": k})
hmm_path = os.path.join(R, "hmm_vs_classifier_student_test.csv")
hmm = pd.read_csv(hmm_path).set_index("model") if os.path.exists(hmm_path) else None

S = pd.DataFrame(rows)
table = pd.DataFrame({
    "model": S["model"],
    "CV top-1": S["cv_top1_mean"],
    "CV macro F1": S["cv_macro_f1_mean"],
    "test top-1 accuracy": S["test_top1"],
    "test accuracy@0.5": S["test_accuracy_at_0.5"],
    "test macro precision": S["test_macro_precision"],
    "test macro recall": S["test_macro_recall"],
    "test macro F1": S["test_macro_f1"],
    "test log loss": S["test_log_loss"],
})
if hmm is not None:
    table["test top-1 + HMM"] = [hmm.loc[k, "hmm_top1"] if k in hmm.index else np.nan for k in S["key"]]
    table["continuity + HMM"] = [hmm.loc[k, "hmm_continuity"] if k in hmm.index else np.nan for k in S["key"]]
print("\nSAVED RESULTS (from results/, student-test = 10 student_test trips)\n")
print(table.round(4).to_string(index=False))
print("""
definitions:
  top-1 accuracy      : per GPS point, highest-probability candidate == reference arc
  accuracy@0.5        : per candidate row, prob >= 0.5 vs label (inflated: ~85% of rows are easy negatives)
  macro precision/recall/F1 : candidate rows, positive = top-1 candidate of each point, averaged over both classes
  log loss            : binary cross-entropy on candidate rows (lower is better)""")

if "--live" in sys.argv:
    from pathsnap.pipeline import cached
    from pathsnap.metrics import evaluate
    print("\nLIVE RE-EVALUATION of artifacts/final_model.joblib on student_test (builds cache if missing) ...")
    test = cached("student_test", cfg["student_test_dir"], cfg)
    model = joblib.load(os.path.join(cfg["artifacts_dir"], "final_model.joblib"))
    t0 = time.time()
    probs = [model.predict_proba(d["X"][:, :model.n_features_in_])[:, 1] for d in test]
    dt = time.time() - t0
    res, per_trip, _ = evaluate(test, probs, root=cfg["student_test_dir"])
    for k, v in res.items():
        print(f"  {k:22s}: {v:.4f}")
    print(f"  inference time        : {dt:.2f} s for {sum(d['n_points'] for d in test)} GPS points")
    print("\n per trip:")
    print(pd.DataFrame(per_trip).round(4).to_string(index=False))

if "--live-all" in sys.argv:
    from pathsnap.pipeline import cached
    from pathsnap.metrics import evaluate
    print("\nLIVE RE-EVALUATION of every saved model on student_test (KNN is slow, ~30 s) ...")
    test = cached("student_test", cfg["student_test_dir"], cfg)
    n_obs = sum(d["n_points"] for d in test)
    items = [(DISPLAY[k], os.path.join(cfg["artifacts_dir"], "models", f"{k}.joblib")) for k in MODEL_KEYS if os.path.exists(os.path.join(cfg["artifacts_dir"], "models", f"{k}.joblib"))]
    out = []
    for name, f in items:
        model = joblib.load(f)
        t0 = time.time()
        probs = [model.predict_proba(d["X"][:, :model.n_features_in_])[:, 1] for d in test]
        dt = time.time() - t0
        res, _, _ = evaluate(test, probs, root=cfg["student_test_dir"])
        out.append({"model": name, "top-1 accuracy": res["top1"], "macro precision": res["macro_precision"],
                    "macro recall": res["macro_recall"], "macro F1": res["macro_f1"], "log loss": res["log_loss"],
                    "continuity": res["continuity"], "seconds": round(dt, 2)})
        print(f"  done {name}", flush=True)
    print()
    print(pd.DataFrame(out).sort_values("top-1 accuracy", ascending=False).round(4).to_string(index=False))
    print(f"\n(classifier only, {n_obs} GPS points in 10 student_test trips; HMM adds ~3 points, see 'test top-1 + HMM' above)")
