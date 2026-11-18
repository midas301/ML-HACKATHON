import os
import sys
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap import config
from pathsnap.data import list_trips
from pathsnap.direction import ALL_FEATURES as FEATURES
from pathsnap.pipeline import process_trip

cfg = config.load()
part = sys.argv[1] if len(sys.argv) > 1 else "train"
root = cfg["train_dir"] if part == "train" else cfg["student_test_dir"]
if not os.path.isdir(root):
    sys.exit(f"data folder not found: {root}\nput the trip folders there or edit config.json")
trips = list_trips(root)
tid = sys.argv[2] if len(sys.argv) > 2 else trips[0]
n_points = int(sys.argv[3]) if len(sys.argv) > 3 else 3
path = os.path.join(cfg["cache_dir"], f"{part}_{tid}.pkl")
if os.path.exists(path):
    d = pickle.load(open(path, "rb"))
else:
    print(f"no cache for {part}/{tid}, building it now ...")
    _, d = process_trip(root, tid, cfg, True)
    os.makedirs(cfg["cache_dir"], exist_ok=True)
    pickle.dump(d, open(path, "wb"))

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
print(f"\ntrip {tid} ({part})")
print(f"  GPS points            : {d['n_points']}")
print(f"  candidate rows        : {len(d['y'])}  (avg {len(d['y']) / d['n_points']:.1f} per point)")
print(f"  points with valid label: {int(d['label_valid'].sum())} ({100 * d['label_valid'].mean():.1f}%)")
hit = np.array([d['y'][d['pid'] == i].any() for i in np.where(d['label_valid'])[0]])
print(f"  candidate miss rate   : {100 * (1 - hit.mean()):.2f}%")
df = pd.DataFrame(d["X"], columns=FEATURES)
df.insert(0, "gps_point", d["pid"])
df.insert(1, "arc_id", d["aid"])
df["label"] = d["y"]
show = df[df.gps_point < n_points]
cols = ["gps_point", "arc_id", "label", "dist", "dist_rank", "dist_minus_min", "heading_diff", "speed", "proj_t", "prev_has", "win_dist", "along_delta"]
print(f"\ncandidate rows for the first {n_points} GPS points (label=1 is the reference arc), key features:")
print(show[cols].round(2).to_string(index=False))
print(f"\nall {len(FEATURES)} features for the same rows:")
print(show.round(2).to_string(index=False))
