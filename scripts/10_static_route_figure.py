import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap import config
from pathsnap.inference import Predictor

cfg = config.load()
tid = sys.argv[1] if len(sys.argv) > 1 else "00000056"
root = cfg["student_test_dir"]
res = Predictor(cfg).predict(root, tid)
trip = res["trip"]
route = np.loadtxt(os.path.join(root, tid, tid + ".route"), dtype=np.int64, ndmin=1)
nl = trip.nodes_lonlat
lo, hi = trip.track_lonlat.min(0) - 0.002, trip.track_lonlat.max(0) + 0.002
mid = nl[trip.arcs].mean(1)
inside = np.where((mid[:, 0] > lo[0]) & (mid[:, 0] < hi[0]) & (mid[:, 1] > lo[1]) & (mid[:, 1] < hi[1]))[0]
fig, ax = plt.subplots(figsize=(10, 9))
for e in inside:
    p, q = nl[trip.arcs[e]]
    ax.plot([p[0], q[0]], [p[1], q[1]], color="0.85", lw=0.6, zorder=1)
for e in route:
    p, q = nl[trip.arcs[e]]
    ax.plot([p[0], q[0]], [p[1], q[1]], color="#2b9348", lw=6, alpha=0.5, zorder=2)
c = res["clf"]
for e in np.unique(c[c >= 0]):
    p, q = nl[trip.arcs[e]]
    ax.plot([p[0], q[0]], [p[1], q[1]], color="#f4a261", lw=2.5, zorder=3)
for e in res["route"]:
    p, q = nl[trip.arcs[e]]
    ax.plot([p[0], q[0]], [p[1], q[1]], color="#d00000", lw=1.5, zorder=4)
ax.scatter(trip.track_lonlat[::2, 0], trip.track_lonlat[::2, 1], s=3, color="#1f6feb", zorder=5)
ax.plot([], [], color="0.85", label="road network"); ax.plot([], [], color="#2b9348", lw=6, alpha=0.5, label="reference route")
ax.plot([], [], color="#f4a261", lw=2.5, label="classifier-only arcs"); ax.plot([], [], color="#d00000", label="predicted route (HMM)")
ax.scatter([], [], s=10, color="#1f6feb", label="raw GPS")
ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect(1 / np.cos(np.radians(trip.lat0)))
ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.legend(loc="best")
ax.set_title(f"Student-test trip {tid}: GPS, reference and predicted routes (static matplotlib view)")
plt.tight_layout(); plt.savefig(os.path.join(cfg["figures_dir"], f"route_static_{tid}.png"), dpi=120); plt.close()
