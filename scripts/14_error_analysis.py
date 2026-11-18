import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap import config
from pathsnap.pipeline import cached
from pathsnap.data import Trip
from pathsnap.hmm import RoadGraph, decode
from pathsnap.metrics import top1_choice
from pathsnap.candidates import gps_kinematics

cfg = config.load()
test = cached("student_test", cfg["student_test_dir"], cfg)
import joblib
model = joblib.load(os.path.join(cfg["artifacts_dir"], "final_model.joblib"))
hcfg = json.load(open(os.path.join(cfg["artifacts_dir"], "hmm_config.json")))
rows = []
for d in test:
    tr = Trip(cfg["student_test_dir"], d["trip_id"], load_route=True)
    prob = model.predict_proba(d["X"])[:, 1]
    clf, _ = top1_choice(d["pid"], prob, d["n_points"], d["aid"])
    hmm, _ = decode(RoadGraph(tr.arcs, tr.arc_len), tr.xy, tr.t, d["pid"], d["aid"], d["proj_t"] * tr.arc_len[d["aid"]], prob, hcfg)
    lab, v = d["label_arc"], d["label_valid"]
    speed = gps_kinematics(tr)[0]
    for i in np.where(v)[0]:
        rows.append({"system": "speed", "trip": d["trip_id"], "speed": speed[i], "ok": hmm[i] == lab[i]})
    route = set(tr.route.tolist())
    for tag, p in (("classifier", clf), ("classifier+HMM", hmm)):
        for i in np.where(v & (p != lab))[0]:
            a, b = p[i], lab[i]
            if a < 0:
                kind = "no candidate"
            elif tr.arcs[a, 0] == tr.arcs[b, 1] and tr.arcs[a, 1] == tr.arcs[b, 0]:
                kind = "same road, opposite direction"
            elif len({*tr.arcs[a]} & {*tr.arcs[b]}) > 0:
                kind = "neighbouring arc (shares a junction)" + (" on the route" if a in route else "")
            elif a in route:
                kind = "another arc of the true route"
            else:
                kind = "different road"
            lab_len = tr.arc_len[b]
            rows.append({"system": tag, "trip": d["trip_id"], "kind": kind, "label_arc_len": lab_len})
    for tag, p in (("classifier", clf), ("classifier+HMM", hmm)):
        rows.append({"system": tag, "trip": d["trip_id"], "kind": "_total", "n": int(v.sum())})
df = pd.DataFrame(rows)
sp = df[df.system == "speed"].copy()
sp["ok"] = sp["ok"].astype(float)
df = df[df.system != "speed"]
sp["speed band (m/s)"] = pd.cut(sp.speed, [-1, 0.5, 2, 5, 10, 100], labels=["<0.5 stopped", "0.5-2", "2-5", "5-10", ">10"])
print("\nclassifier+HMM top-1 by GPS speed:")
print(sp.groupby("speed band (m/s)", observed=True).agg(points=("ok", "size"), top1=("ok", "mean")).round(3).to_string())
for tag in ("classifier", "classifier+HMM"):
    s = df[df.system == tag]
    total = s[s.kind == "_total"].n.sum()
    err = s[s.kind != "_total"]
    print(f"\n{tag}: {len(err)} wrong of {int(total)} points -> top-1 {1 - len(err) / total:.4f}")
    c = err.kind.value_counts()
    for k, n in c.items():
        print(f"  {k:48s} {n:5d}  {100 * n / len(err):5.1f}% of errors  {100 * n / total:5.2f}% of points")
    print(f"  median length of the true arc at error points: {err.label_arc_len.median():.1f} m (all arcs median ~25 m)")
