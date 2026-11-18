import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULTS = {
    "data_dir": os.path.join(ROOT, "data"),
    "train_dir": os.path.join(ROOT, "data", "train"),
    "student_test_dir": os.path.join(ROOT, "data", "student_test"),
    "eval_dir": os.path.join(ROOT, "data", "evaluator_test"),
    "cache_dir": os.path.join(ROOT, "artifacts", "cache"),
    "artifacts_dir": os.path.join(ROOT, "artifacts"),
    "results_dir": os.path.join(ROOT, "results"),
    "figures_dir": os.path.join(ROOT, "figures"),
    "maps_dir": os.path.join(ROOT, "maps"),
    "radius": 50.0,
    "k": 8,
    "label_max_dist": 50.0,
    "seed": 42,
}


def load():
    cfg = dict(DEFAULTS)
    p = os.path.join(ROOT, "config.json")
    if os.path.exists(p):
        with open(p) as f:
            cfg.update(json.load(f))
    for key in ("data_dir", "train_dir", "student_test_dir", "eval_dir", "cache_dir", "artifacts_dir", "results_dir", "figures_dir", "maps_dir"):
        if not os.path.isabs(cfg[key]):
            cfg[key] = os.path.join(ROOT, cfg[key])
    return cfg
