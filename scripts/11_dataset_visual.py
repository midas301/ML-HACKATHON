import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap import config
from pathsnap.data import Trip, list_trips, point_segment
from pathsnap.labels import align_route
from pathsnap.candidates import ArcIndex, generate

cfg = config.load()
tid = sys.argv[1] if len(sys.argv) > 1 else "00000048"
tr = Trip(cfg["train_dir"], tid, load_route=True)
lab, ld, _ = align_route(tr)
cands = generate(tr, ArcIndex(tr), radius=50.0, k=8)
xy, nxy = tr.xy, tr.nxy
rset = set(tr.route.tolist())

fig = plt.figure(figsize=(16, 13))
ax1 = fig.add_subplot(2, 2, 1)
lo, hi = xy.min(0) - 150, xy.max(0) + 150
mid = (tr.a_xy0 + tr.a_xy1) / 2
inside = np.where((mid[:, 0] > lo[0]) & (mid[:, 0] < hi[0]) & (mid[:, 1] > lo[1]) & (mid[:, 1] < hi[1]))[0]
for e in inside:
    ax1.plot(*np.c_[tr.a_xy0[e], tr.a_xy1[e]], color="0.82", lw=0.6, zorder=1)
for e in tr.route:
    ax1.plot(*np.c_[tr.a_xy0[e], tr.a_xy1[e]], color="#2b9348", lw=4, alpha=0.6, zorder=2)
ax1.scatter(xy[:, 0], xy[:, 1], s=2, color="#1f6feb", zorder=3)
ax1.plot(*xy[0], "o", color="black", ms=8, zorder=4); ax1.annotate("start", xy[0], xytext=(8, 8), textcoords="offset points")
ax1.plot(*xy[-1], "s", color="black", ms=8, zorder=4); ax1.annotate("end", xy[-1], xytext=(8, -14), textcoords="offset points")
i0 = int(np.argmax(np.array([len(c[0]) for c in cands]) * (np.arange(len(cands)) > 200)))
c0 = xy[i0]
W = 70
ax1.add_patch(plt.Rectangle(c0 - W, 2 * W, 2 * W, fill=False, ec="red", lw=1.5, zorder=5))
ax1.annotate("zoomed in panel (b)", c0 + [W, W], xytext=(6, 4), textcoords="offset points", color="red")
ax1.set_aspect("equal"); ax1.set_xlabel("east (m)"); ax1.set_ylabel("north (m)")
ax1.set_title(f"(a) Train trip {tid}: {len(xy)} GPS fixes (blue), reference route (green),\n{len(tr.arcs):,} network arcs (grey, only nearby ones drawn)")

ax2 = fig.add_subplot(2, 2, 2)
box = lambda p: np.all(np.abs(p - c0) < W * 1.15, axis=-1)
near = np.where(box(tr.a_xy0) | box(tr.a_xy1))[0]
for e in near:
    p, q = tr.a_xy0[e], tr.a_xy1[e]
    d = q - p; n = np.array([-d[1], d[0]]) / (np.hypot(*d) + 1e-9) * 1.2
    col, lw = ("#2b9348", 2.2) if e in rset else ("0.6", 0.9)
    ax2.add_patch(FancyArrowPatch(p + n, q + n, arrowstyle="-|>", mutation_scale=10, color=col, lw=lw, zorder=2))
nn = np.where(box(nxy))[0]
ax2.scatter(nxy[nn, 0], nxy[nn, 1], s=22, color="black", zorder=3)
for k in nn[:40]:
    ax2.annotate(str(k), nxy[k], xytext=(3, 3), textcoords="offset points", fontsize=7, color="0.3")
pts = np.where(box(xy))[0]
for i in pts:
    e = lab[i]
    p, q = tr.a_xy0[e], tr.a_xy1[e]
    _, t = point_segment(xy[i, 0], xy[i, 1], p[0], p[1], q[0], q[1])
    proj = p + t * (q - p)
    ax2.plot([xy[i, 0], proj[0]], [xy[i, 1], proj[1]], color="#9b5de5", lw=0.7, zorder=3)
ax2.scatter(xy[pts, 0], xy[pts, 1], s=18, color="#1f6feb", zorder=4)
for i in pts[::4]:
    ax2.annotate(f"t={tr.t[i]:.0f}", xy[i], xytext=(4, -9), textcoords="offset points", fontsize=7, color="#1f6feb")
ca = cands[i0][0]
for e in ca:
    ax2.plot(*np.c_[tr.a_xy0[e], tr.a_xy1[e]], color="#f4a261", lw=5, alpha=0.45, zorder=1)
ax2.add_patch(Circle(xy[i0], 50, fill=False, ls="--", ec="#e76f51", lw=1.2, zorder=5))
ax2.plot(*xy[i0], "*", color="#e76f51", ms=16, zorder=6)
ax2.set_xlim(c0[0] - W, c0[0] + W); ax2.set_ylim(c0[1] - W, c0[1] + W); ax2.set_aspect("equal")
ax2.plot([], [], color="0.6", label="arc (directed, arrow = travel direction)")
ax2.plot([], [], color="#2b9348", lw=2.2, label="arc on the reference route")
ax2.scatter([], [], color="black", s=22, label="node (number = row in .nodes)")
ax2.scatter([], [], color="#1f6feb", s=18, label="GPS fix (t = seconds)")
ax2.plot([], [], color="#9b5de5", label="derived label: fix -> its route arc")
ax2.plot([], [], color="#f4a261", lw=5, alpha=0.45, label=f"{len(ca)} candidate arcs within 50 m of the star")
ax2.legend(loc="lower left", fontsize=8, framealpha=0.9)
ax2.set_xlabel("east (m)"); ax2.set_ylabel("north (m)")
ax2.set_title(f"(b) Zoom: {2*W} m x {2*W} m window. Two-way roads appear as paired arcs;\nthe model must pick one candidate per fix")

inv = pd.read_csv(os.path.join(cfg["results_dir"], "eda_trip_inventory.csv"))
ax3 = fig.add_subplot(2, 2, 3)
all_ld, all_dt = [], []
for t in list_trips(cfg["train_dir"])[:40]:
    tr2 = Trip(cfg["train_dir"], t, load_route=True)
    _, l2, _ = align_route(tr2)
    all_ld.append(l2); all_dt.append(np.diff(tr2.t))
all_ld = np.concatenate(all_ld); all_dt = np.concatenate(all_dt)
ax3.hist(np.clip(all_ld, 0, 80), bins=80, color="#1f6feb")
ax3.axvline(50, color="red", ls="--"); ax3.text(51, ax3.get_ylim()[1] * 0.8, "50 m: beyond this,\nfix is unlabelled", color="red", fontsize=9)
ax3.set_xlabel("distance from GPS fix to its reference arc (m, clipped at 80)"); ax3.set_ylabel("fixes")
ins = ax3.inset_axes([0.5, 0.25, 0.45, 0.35])
v, c = np.unique(np.clip(all_dt, 0, 10), return_counts=True)
ins.bar(v, c, color="0.4"); ins.set_yscale("log"); ins.set_title("seconds between fixes (clip 10)", fontsize=8); ins.tick_params(labelsize=7)
ax3.set_title(f"(c) GPS noise: median {np.median(all_ld):.1f} m, 95th pct {np.percentile(all_ld, 95):.1f} m\n(40 train trips, {len(all_ld):,} fixes); inset: sampling is mostly 1 Hz")

ax4 = fig.add_subplot(2, 2, 4)
cx = (inv.lon_min + inv.lon_max) / 2; cy = (inv.lat_min + inv.lat_max) / 2
for part, col in (("train", "#1f6feb"), ("student_test", "#e76f51")):
    m = inv.partition == part
    ax4.scatter(cx[m], cy[m], s=inv.gps_points[m] / 60 + 8, color=col, alpha=0.6, label=f"{part} ({m.sum()} trips)")
ax4.set_xlim(-180, 180); ax4.set_ylim(-60, 75); ax4.axhline(0, color="0.85", lw=0.5); ax4.axvline(0, color="0.85", lw=0.5)
ax4.set_xlabel("longitude"); ax4.set_ylabel("latitude"); ax4.legend(loc="lower left")
ax4.set_title("(d) Where trips are: centroids worldwide, dot size = GPS fixes\n(each trip ships its own clipped road network)")
plt.tight_layout()
out = os.path.join(cfg["figures_dir"], "dataset_explained.png")
plt.savefig(out, dpi=110); print(out, i0, len(ca))
