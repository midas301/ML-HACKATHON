import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import folium
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap import config
from pathsnap.data import Trip, list_trips
from pathsnap.labels import align_route

cfg = config.load()
fig_dir, res_dir, map_dir = cfg["figures_dir"], cfg["results_dir"], cfg["maps_dir"]
for d in (fig_dir, res_dir, map_dir):
    os.makedirs(d, exist_ok=True)

rows = []
for part, root in (("train", cfg["train_dir"]), ("student_test", cfg["student_test_dir"])):
    for tid in list_trips(root):
        base = os.path.join(root, tid, tid)
        raw = np.loadtxt(base + ".track", ndmin=2)
        tr = Trip(root, tid, load_route=True)
        dt = np.diff(raw[:, 2])
        n_nodes, n_arcs = len(tr.nodes_lonlat), len(tr.arcs)
        g = coo_matrix((np.ones(n_arcs), (tr.arcs[:, 0], tr.arcs[:, 1])), shape=(n_nodes, n_nodes))
        n_weak, lab_w = connected_components(g, directed=True, connection="weak")
        n_strong, lab_s = connected_components(g, directed=True, connection="strong")
        deg = tr.out_deg + tr.in_deg
        step = np.hypot(*np.diff(tr.xy, axis=0).T)
        r = tr.route
        rcont = float(np.mean(tr.arcs[r[:-1], 1] == tr.arcs[r[1:], 0])) if len(r) > 1 else 1.0
        lab, ld, _ = align_route(tr)
        rows.append({
            "partition": part, "trip_id": tid,
            "track_bytes": os.path.getsize(base + ".track"), "nodes_bytes": os.path.getsize(base + ".nodes"),
            "arcs_bytes": os.path.getsize(base + ".arcs"), "route_bytes": os.path.getsize(base + ".route"),
            "gps_points": len(raw), "nan_values": int(np.isnan(raw).sum()), "dup_or_backward_ts": int((dt <= 0).sum()),
            "duration_s": float(raw[-1, 2] - raw[0, 2]), "dt_eq_1_frac": float(np.mean(dt == 1)), "dt_max": float(dt.max()),
            "gaps_gt_5s": int((dt > 5).sum()), "max_step_m": float(step.max()), "speed_p99_mps": float(np.percentile(step / np.maximum(np.diff(tr.t), 1), 99)),
            "path_length_km": float(step.sum() / 1000),
            "lon_min": raw[:, 0].min(), "lon_max": raw[:, 0].max(), "lat_min": raw[:, 1].min(), "lat_max": raw[:, 1].max(),
            "nodes": n_nodes, "arcs": n_arcs, "isolated_nodes": int((deg == 0).sum()),
            "weak_components": int(n_weak), "largest_weak_frac": float(np.bincount(lab_w).max() / n_nodes),
            "strong_components": int(n_strong), "largest_strong_frac": float(np.bincount(lab_s).max() / n_nodes),
            "bidirectional_arc_frac": float(np.isin(tr.arcs[:, 1] * (n_nodes + 1) + tr.arcs[:, 0], tr.arcs[:, 0] * (n_nodes + 1) + tr.arcs[:, 1]).mean()),
            "arc_len_median_m": float(np.median(tr.arc_len)),
            "route_arcs": len(r), "route_connected_frac": rcont, "route_unique_frac": len(set(r.tolist())) / len(r),
            "label_dist_median_m": float(np.median(ld)), "label_dist_p95_m": float(np.percentile(ld, 95)),
            "points_far_from_route_frac": float(np.mean(ld > cfg["label_max_dist"])),
        })
        print(part, tid, flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(res_dir, "eda_trip_inventory.csv"), index=False)
summary = df.groupby("partition").agg(
    trips=("trip_id", "count"), gps_points=("gps_points", "sum"), mean_points=("gps_points", "mean"),
    nan_values=("nan_values", "sum"), dup_ts=("dup_or_backward_ts", "sum"), mean_dt1=("dt_eq_1_frac", "mean"),
    max_gap=("dt_max", "max"), trips_with_gap_gt5=("gaps_gt_5s", lambda s: int((s > 0).sum())),
    mean_nodes=("nodes", "mean"), mean_arcs=("arcs", "mean"), isolated_nodes=("isolated_nodes", "sum"),
    mean_weak_components=("weak_components", "mean"), route_connected=("route_connected_frac", "mean"),
    far_from_route=("points_far_from_route_frac", "mean")).reset_index()
summary.to_csv(os.path.join(res_dir, "eda_summary.csv"), index=False)
print(summary.T)

fig, ax = plt.subplots(2, 2, figsize=(12, 8))
for part, c in (("train", "tab:blue"), ("student_test", "tab:orange")):
    s = df[df.partition == part]
    ax[0, 0].hist(s.gps_points, bins=25, alpha=0.6, color=c, label=part)
    ax[0, 1].hist(s.path_length_km, bins=25, alpha=0.6, color=c, label=part)
    ax[1, 0].hist(s.label_dist_median_m, bins=25, alpha=0.6, color=c, label=part)
    ax[1, 1].scatter(s.arcs, s.gps_points, alpha=0.7, color=c, label=part)
ax[0, 0].set_title("GPS points per trip"); ax[0, 1].set_title("Trajectory length (km)")
ax[1, 0].set_title("Median GPS-to-route distance per trip (m)"); ax[1, 1].set_title("Road arcs vs GPS points")
ax[1, 1].set_xscale("log")
for a in ax.ravel():
    a.legend()
plt.tight_layout(); plt.savefig(os.path.join(fig_dir, "eda_distributions.png"), dpi=120); plt.close()

all_dt = []
for part, root in (("train", cfg["train_dir"]),):
    for tid in list_trips(root):
        raw = np.loadtxt(os.path.join(root, tid, tid + ".track"), ndmin=2)
        all_dt.append(np.diff(raw[:, 2]))
all_dt = np.concatenate(all_dt)
plt.figure(figsize=(7, 4))
vals, cnts = np.unique(np.clip(all_dt, 0, 20), return_counts=True)
plt.bar(vals, cnts); plt.yscale("log"); plt.xlabel("seconds between fixes (clipped at 20)"); plt.ylabel("count (log)")
plt.title("Sampling interval distribution (train)")
plt.tight_layout(); plt.savefig(os.path.join(fig_dir, "eda_sampling_intervals.png"), dpi=120); plt.close()

miss = df[["partition", "nan_values", "dup_or_backward_ts", "gaps_gt_5s"]].groupby("partition").sum()
miss["points_far_from_route"] = df.assign(n=df.gps_points * df.points_far_from_route_frac).groupby("partition").n.sum().round()
miss.to_csv(os.path.join(res_dir, "eda_missing_invalid.csv"))
miss.plot.bar(figsize=(8, 4), logy=True); plt.title("Missing / invalid observation counts by partition")
plt.tight_layout(); plt.savefig(os.path.join(fig_dir, "eda_missing_invalid.png"), dpi=120); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(16, 5))
for a, tid in zip(ax, list_trips(cfg["train_dir"])[:3]):
    tr = Trip(cfg["train_dir"], tid, load_route=True)
    for e in np.random.default_rng(0).choice(len(tr.arcs), min(4000, len(tr.arcs)), replace=False):
        p, q = tr.nodes_lonlat[tr.arcs[e]]
        a.plot([p[0], q[0]], [p[1], q[1]], color="0.85", lw=0.5)
    for e in tr.route:
        p, q = tr.nodes_lonlat[tr.arcs[e]]
        a.plot([p[0], q[0]], [p[1], q[1]], color="green", lw=2)
    a.scatter(tr.track_lonlat[:, 0], tr.track_lonlat[:, 1], s=2, color="blue")
    a.set_title(f"train trip {tid}"); a.set_xlabel("lon"); a.set_ylabel("lat")
    pad = 0.005
    a.set_xlim(tr.track_lonlat[:, 0].min() - pad, tr.track_lonlat[:, 0].max() + pad)
    a.set_ylim(tr.track_lonlat[:, 1].min() - pad, tr.track_lonlat[:, 1].max() + pad)
plt.tight_layout(); plt.savefig(os.path.join(fig_dir, "eda_trajectories_network.png"), dpi=120); plt.close()

plt.figure(figsize=(7, 6))
for part, c in (("train", "tab:blue"), ("student_test", "tab:orange")):
    s = df[df.partition == part]
    plt.scatter((s.lon_min + s.lon_max) / 2, (s.lat_min + s.lat_max) / 2, s=15, color=c, label=part)
plt.legend(); plt.title("Trip centroids"); plt.xlabel("lon"); plt.ylabel("lat")
plt.tight_layout(); plt.savefig(os.path.join(fig_dir, "eda_trip_centroids.png"), dpi=120); plt.close()

tid = list_trips(cfg["train_dir"])[47]
tr = Trip(cfg["train_dir"], tid, load_route=True)
m = folium.Map(location=[tr.lat0, tr.lon0], zoom_start=15, tiles="OpenStreetMap")
lo, hi = tr.track_lonlat.min(0) - 0.003, tr.track_lonlat.max(0) + 0.003
mid = tr.nodes_lonlat[tr.arcs].mean(1)
inside = np.where((mid[:, 0] > lo[0]) & (mid[:, 0] < hi[0]) & (mid[:, 1] > lo[1]) & (mid[:, 1] < hi[1]))[0]
net = folium.FeatureGroup("Road network (arcs)")
for e in inside:
    p, q = tr.nodes_lonlat[tr.arcs[e]]
    folium.PolyLine([(p[1], p[0]), (q[1], q[0])], color="#999999", weight=1).add_to(net)
net.add_to(m)
ref = folium.FeatureGroup("Reference route")
folium.PolyLine([(tr.nodes_lonlat[tr.arcs[e, j], 1], tr.nodes_lonlat[tr.arcs[e, j], 0]) for e in tr.route for j in (0, 1)], color="green", weight=5).add_to(ref)
ref.add_to(m)
gps = folium.FeatureGroup("Raw GPS")
for lon, lat in tr.track_lonlat[::3]:
    folium.CircleMarker((lat, lon), radius=2, color="blue", fill=True).add_to(gps)
gps.add_to(m)
folium.LayerControl().add_to(m)
m.save(os.path.join(map_dir, f"eda_train_{tid}.html"))
with open(os.path.join(res_dir, "eda_notes.json"), "w") as f:
    json.dump({"folium_trip": tid}, f)
