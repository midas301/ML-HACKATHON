import os
import sys
import time
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap import config
from pathsnap.pipeline import cached

cfg = config.load()
summary = {}
for part, root in (("train", cfg["train_dir"]), ("student_test", cfg["student_test_dir"])):
    t0 = time.time()
    trips = cached(part, root, cfg)
    pts = sum(d["n_points"] for d in trips)
    lv = np.concatenate([d["label_valid"] for d in trips])
    hit = []
    nearest = []
    for d in trips:
        lab = d["label_arc"]
        for i in np.where(d["label_valid"])[0]:
            m = d["pid"] == i
            hit.append(d["y"][m].any())
        first = np.unique(d["pid"], return_index=True)
        ok = d["label_valid"][first[0]]
        nearest.extend(d["y"][first[1]][ok].tolist())
    summary[part] = {
        "trips": len(trips), "points": int(pts), "rows": int(sum(len(d["y"]) for d in trips)),
        "label_valid_frac": float(lv.mean()), "candidate_miss_rate": float(1 - np.mean(hit)),
        "nearest_arc_top1": float(np.mean(nearest)), "positive_rate": float(np.mean(np.concatenate([d["y"] for d in trips]))),
        "seconds": round(time.time() - t0, 1),
    }
    print(part, summary[part], flush=True)
os.makedirs(cfg["results_dir"], exist_ok=True)
with open(os.path.join(cfg["results_dir"], "dataset_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
