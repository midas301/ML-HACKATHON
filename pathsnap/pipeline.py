import os
import pickle
import numpy as np
from .data import Trip, list_trips
from .labels import align_route
from .candidates import ArcIndex, generate, build_features


def process_trip(root, trip_id, cfg, with_labels):
    trip = Trip(root, trip_id, load_route=with_labels)
    index = ArcIndex(trip)
    cands = generate(trip, index, radius=cfg["radius"], k=cfg["k"])
    X, pid, aid = build_features(trip, cands, radius=cfg["radius"])
    out = {"trip_id": trip_id, "X": X, "pid": pid, "aid": aid, "n_points": len(trip.xy),
           "dist": X[:, 0].copy(), "proj_t": X[:, 9].copy()}
    if with_labels and trip.route is not None:
        lab, ld, _ = align_route(trip)
        valid = ld <= cfg["label_max_dist"]
        out["label_arc"] = lab
        out["label_dist"] = ld
        out["label_valid"] = valid
        out["y"] = (aid == lab[pid]).astype(np.int8)
        out["row_valid"] = valid[pid]
    return trip, out


def cached(partition, root, cfg):
    os.makedirs(cfg["cache_dir"], exist_ok=True)
    res = []
    for tid in list_trips(root):
        p = os.path.join(cfg["cache_dir"], f"{partition}_{tid}.pkl")
        if os.path.exists(p):
            with open(p, "rb") as f:
                res.append(pickle.load(f))
            continue
        _, out = process_trip(root, tid, cfg, True)
        with open(p, "wb") as f:
            pickle.dump(out, f)
        res.append(out)
    return res


def stack(trips, only_valid=True):
    Xs, ys, gs, ps, ts = [], [], [], [], []
    for g, d in enumerate(trips):
        m = d["row_valid"] if only_valid else np.ones(len(d["y"]), bool)
        Xs.append(d["X"][m]); ys.append(d["y"][m]); gs.append(np.full(m.sum(), g)); ps.append(d["pid"][m])
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(gs), np.concatenate(ps)
