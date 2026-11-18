import os
import json
import time
import joblib
import numpy as np
from .data import Trip
from .candidates import ArcIndex, generate, build_features
from .hmm import RoadGraph, decode, route_from_points
from .metrics import continuity, sequence_scores
from .labels import align_route


class Predictor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = joblib.load(os.path.join(cfg["artifacts_dir"], "final_model.joblib"))
        with open(os.path.join(cfg["artifacts_dir"], "hmm_config.json")) as f:
            self.hmm = json.load(f)

    def predict(self, root, trip_id):
        timings = {}
        t0 = time.time()
        trip = Trip(root, trip_id, load_route=False)
        timings["load_s"] = time.time() - t0
        t0 = time.time()
        cands = generate(trip, ArcIndex(trip), radius=self.cfg["radius"], k=self.cfg["k"])
        X, pid, aid = build_features(trip, cands, radius=self.cfg["radius"])
        timings["features_s"] = time.time() - t0
        t0 = time.time()
        prob = self.model.predict_proba(X)[:, 1]
        timings["classifier_s"] = time.time() - t0
        order = np.lexsort((-prob, pid))
        first = np.concatenate([[True], pid[order][1:] != pid[order][:-1]])
        clf = np.full(len(trip.xy), -1, dtype=np.int64)
        clf[pid[order[first]]] = aid[order[first]]
        t0 = time.time()
        graph = RoadGraph(trip.arcs, trip.arc_len)
        hmm, breaks = decode(graph, trip.xy, trip.t, pid, aid, X[:, 9] * trip.arc_len[aid], prob, self.hmm)
        route = route_from_points(graph, hmm)
        timings["hmm_s"] = time.time() - t0
        return {"trip": trip, "cands": cands, "clf": clf, "hmm": hmm, "route": route, "breaks": breaks, "timings": timings}


def server_side_scores(root, trip_id, trip, result, label_max_dist):
    ref = Trip(root, trip_id, load_route=True)
    if ref.route is None:
        return None
    lab, ld, _ = align_route(ref)
    v = ld <= label_max_dist
    out = {"scored_points": int(v.sum()), "total_points": int(len(v))}
    cand_hit = np.array([lab[i] in set(c[0].tolist()) for i, c in enumerate(result["cands"])])
    out["candidate_miss_rate"] = float(1 - cand_hit[v].mean()) if v.any() else None
    for tag in ("clf", "hmm"):
        p = result[tag]
        out[f"{tag}_top1_accuracy"] = float((p[v] == lab[v]).mean()) if v.any() else None
        out[f"{tag}_route_continuity"] = continuity(p, trip.arcs)
        pr, rc, f1 = sequence_scores(p, ref.route)
        out[f"{tag}_route_precision"], out[f"{tag}_route_recall"], out[f"{tag}_route_f1"] = pr, rc, f1
    rset = set(ref.route.tolist())
    pset = set(result["route"])
    inter = len(rset & pset)
    lens = trip.arc_len
    out["route_length_overlap_recall"] = float(lens[list(rset & pset)].sum() / max(1e-9, lens[list(rset)].sum())) if inter else 0.0
    out["route_length_overlap_precision"] = float(lens[list(rset & pset)].sum() / max(1e-9, lens[list(pset)].sum())) if inter else 0.0
    return out
