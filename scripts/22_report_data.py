import os
import re
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap import config
from pathsnap.registry_view import MODEL_KEYS, DISPLAY
from pathsnap.direction import ALL_FEATURES

cfg = config.load()
R, A = cfg["results_dir"], cfg["artifacts_dir"]
J = lambda f: json.load(open(os.path.join(R, f)))
inv = pd.read_csv(os.path.join(R, "eda_trip_inventory.csv"))
D = {}
D["dataset"] = J("dataset_summary.json")
parts = {}
for p, g in inv.groupby("partition"):
    parts[p] = {
        "trips": int(len(g)), "points": int(g.gps_points.sum()), "points_min": int(g.gps_points.min()), "points_max": int(g.gps_points.max()),
        "points_mean": float(g.gps_points.mean()), "nodes_min": int(g.nodes.min()), "nodes_max": int(g.nodes.max()), "arcs_min": int(g.arcs.min()),
        "arcs_max": int(g.arcs.max()), "arcs_mean": float(g.arcs.mean()), "nan": int(g.nan_values.sum()), "dup": int(g.dup_or_backward_ts.sum()),
        "dt1": float(g.dt_eq_1_frac.mean()), "gap_trips": int((g.gaps_gt_5s > 0).sum()), "gaps": int(g.gaps_gt_5s.sum()), "max_gap": float(g.dt_max.max()),
        "max_gap_trip": str(g.loc[g.dt_max.idxmax(), "trip_id"]).zfill(8), "isolated": int(g.isolated_nodes.sum()),
        "weak_med": float(g.weak_components.median()), "weak_max": int(g.weak_components.max()), "largest_weak": float(g.largest_weak_frac.mean()),
        "strong_med": float(g.strong_components.median()), "bidir": float(g.bidirectional_arc_frac.mean()), "arclen_med": float(g.arc_len_median_m.median()),
        "route_conn": float(g.route_connected_frac.mean()), "route_arcs_mean": float(g.route_arcs.mean()), "len_km_mean": float(g.path_length_km.mean()),
        "len_km_max": float(g.path_length_km.max()), "dur_min_mean": float(g.duration_s.mean() / 60), "label_med": float(g.label_dist_median_m.median()),
        "far_mean": float(g.points_far_from_route_frac.mean()),
        "track_mb": float(g.track_bytes.sum() / 1e6), "nodes_mb": float(g.nodes_bytes.sum() / 1e6), "arcs_mb": float(g.arcs_bytes.sum() / 1e6), "route_kb": float(g.route_bytes.sum() / 1e3),
        "speed_p99_med": float(g.speed_p99_mps.median()),
    }
D["parts"] = parts
D["extent"] = {"lat_min": float(inv.lat_min.min()), "lat_max": float(inv.lat_max.max()), "lon_min": float(inv.lon_min.min()), "lon_max": float(inv.lon_max.max())}
far = inv.sort_values("points_far_from_route_frac", ascending=False).head(3)
D["far_trips"] = [{"trip": str(r.trip_id).zfill(8), "points": int(r.gps_points), "far": float(r.points_far_from_route_frac)} for r in far.itertuples()]
rows = []
mc = pd.read_csv(os.path.join(R, "model_comparison.csv"))
for k in MODEL_KEYS:
    r = mc[mc.key == k]
    if len(r):
        s = J(f"summary_{k}.json")
        d = r.iloc[0].to_dict() | {k2: s.get(k2) for k2 in ("test_macro_precision", "test_macro_recall", "test_accuracy_at_0.5", "train_log_loss", "train_macro_f1")}
        d["display"] = DISPLAY[k]
        rows.append({a: (None if isinstance(b, float) and np.isnan(b) else b) for a, b in d.items()})
D["models"] = rows
bo = J("bayes_opt_summary.json")
tr = pd.read_csv(os.path.join(R, "bayes_opt_trials.csv"))
D["bayes"] = {"search": bo["search"], "untuned22": bo["untuned"], "tuned22": bo["tuned"], "trial_min": float(tr.value.min()), "trial_max": float(tr.value.max()),
              "trial_std": float(tr.value.std()), "n_trials": int(len(tr)), "trial0": float(tr.value.iloc[0]),
              "importance": pd.read_csv(os.path.join(R, "bayes_opt_param_importance.csv"), index_col=0).iloc[:, 0].round(3).to_dict(),
              "best_params": json.load(open(os.path.join(A, "best_lgbm_params.json")))}
D["untuned29"] = J("summary_LightGBM_dir.json"); D["tuned29"] = J("summary_LightGBM_dir_tuned.json")
D["ablation"] = J("feature_ablation_direction.json")
D["partial29"] = {m: J(os.path.join("archive_partial_29features", f"summary_{m}.json")) for m in ("LogReg", "KNN")}
hm = pd.read_csv(os.path.join(R, "hmm_vs_classifier_student_test.csv"))
D["hmm"] = hm.round(6).to_dict(orient="records")
D["hmm_cfg"] = json.load(open(os.path.join(A, "hmm_config.json")))
g = pd.read_csv(os.path.join(R, "hmm_grid_train_oof.csv"))
D["hmm_grid"] = {"min": float(g.hmm_top1.min()), "max": float(g.hmm_top1.max()), "clf": float(g.clf_top1.iloc[0]), "n": int(len(g))}
pt = pd.read_csv(os.path.join(R, "hmm_per_trip_LightGBM_dir_tuned.csv"))
D["per_trip"] = pt[["trip_id", "points", "breaks", "clf_top1", "hmm_top1", "clf_continuity", "hmm_continuity", "hmm_seq_precision", "hmm_seq_recall", "hmm_seq_f1"]].assign(trip_id=lambda x: x.trip_id.astype(str).str.zfill(8)).round(4).to_dict(orient="records")
txt = open(os.path.join(R, "error_analysis.txt")).read()
D["speed"] = [{"band": m[0].strip(), "points": int(m[1]), "top1": float(m[2])} for m in re.findall(r"^(<0\.5 stopped|0\.5-2|2-5|5-10|>10)\s+(\d+)\s+([\d.]+)$", txt, re.M)]
err = {}
for block in txt.split("\n\n"):
    head = re.search(r"^(classifier\+HMM|classifier): (\d+) wrong of (\d+) points", block.strip(), re.M)
    if head:
        err[head.group(1)] = {"wrong": int(head.group(2)), "total": int(head.group(3)),
                              "kinds": [{"kind": m[0].strip(), "n": int(m[1]), "share": float(m[2])} for m in re.findall(r"^\s{2}(.+?)\s+(\d+)\s+([\d.]+)% of errors", block, re.M)]}
D["errors"] = err
h = J("training_histories.json")
D["hist"] = {k: {"valid_min": float(min(v["valid"])), "valid_argmin": int(np.argmin(v["valid"])), "valid_end": float(v["valid"][-1]), "train_end": float(v["train"][-1]), "train_start": float(v["train"][0]), "n": len(v["valid"])} for k, v in h.items() if k != "valid_trips"}
fi = pd.read_csv(os.path.join(R, "feature_importance_gain.csv"), index_col=0).iloc[:, 0].sort_values(ascending=False)
D["feat_imp"] = [{"feature": f, "gain": float(v), "share": float(v / fi.sum())} for f, v in fi.items()]
D["features"] = ALL_FEATURES
D["timing22"] = pd.read_csv(os.path.join(R, "timing_clean.csv")).to_dict(orient="records")
D["api_log"] = open("docs/api_log_sample.txt").read().splitlines()[:8]
json.dump(D, open(os.path.join(R, "report_data.json"), "w"), indent=1, default=float)
print("ok", len(json.dumps(D)))
print(json.dumps(D["errors"], indent=0)[:600]); print(D["speed"]); print(D["hist"]); print(D["far_trips"])
