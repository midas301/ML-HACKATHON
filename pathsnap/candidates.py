import numpy as np
from scipy.spatial import cKDTree
from .data import point_segment

FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio", "heading_diff", "heading_cos",
    "heading_valid", "speed", "arc_len", "proj_t", "at_endpoint", "n_cand", "prev_has",
    "next_has", "prev_dist", "next_dist", "win_dist", "win_presence", "along_delta",
    "out_deg_to", "in_deg_from", "dt_prev",
]


class ArcIndex:
    def __init__(self, trip, step=8.0):
        reps = np.maximum(1, np.ceil(trip.arc_len / step).astype(int))
        ids = np.repeat(np.arange(len(trip.arcs)), reps + 1)
        offs = np.concatenate([np.linspace(0, 1, k + 1) for k in reps])
        self.pts = trip.a_xy0[ids] + offs[:, None] * (trip.a_xy1[ids] - trip.a_xy0[ids])
        self.ids = ids
        self.step = step
        self.tree = cKDTree(self.pts)


def generate(trip, index, radius=50.0, k=8, fallback=200.0):
    xy = trip.xy
    out = []
    hits = index.tree.query_ball_point(xy, r=radius + index.step)
    for i, h in enumerate(hits):
        arcs = np.unique(index.ids[h]) if len(h) else np.empty(0, dtype=np.int64)
        if len(arcs) == 0:
            _, nn = index.tree.query(xy[i], k=64, distance_upper_bound=fallback)
            nn = nn[nn < len(index.ids)]
            arcs = np.unique(index.ids[nn])
        if len(arcs) == 0:
            out.append((np.empty(0, dtype=np.int64), np.empty(0), np.empty(0)))
            continue
        d, t = point_segment(xy[i, 0], xy[i, 1], trip.a_xy0[arcs, 0], trip.a_xy0[arcs, 1], trip.a_xy1[arcs, 0], trip.a_xy1[arcs, 1])
        o = np.argsort(d, kind="stable")[:k]
        out.append((arcs[o], d[o], t[o]))
    return out


def gps_kinematics(trip, w=2):
    xy, t = trip.xy, trip.t
    n = len(xy)
    lo = np.clip(np.arange(n) - w, 0, n - 1)
    hi = np.clip(np.arange(n) + w, 0, n - 1)
    d = xy[hi] - xy[lo]
    dist = np.hypot(d[:, 0], d[:, 1])
    dt = np.maximum(t[hi] - t[lo], 1e-6)
    speed = dist / dt
    bearing = np.degrees(np.arctan2(d[:, 0], d[:, 1])) % 360
    valid = (dist > 3.0).astype(float)
    dt_prev = np.concatenate([[1.0], np.diff(t)])
    return speed, bearing, valid, dt_prev


def build_features(trip, cands, radius=50.0, win=3):
    n = len(cands)
    speed, bearing, hvalid, dt_prev = gps_kinematics(trip)
    maps = [dict(zip(c[0].tolist(), zip(c[1].tolist(), (c[2] * trip.arc_len[c[0]]).tolist()))) for c in cands]
    rows, pid, aid = [], [], []
    miss = 2 * radius
    for i, (arcs, d, t) in enumerate(cands):
        m = len(arcs)
        if m == 0:
            continue
        dmin = d.min()
        rank = np.argsort(np.argsort(d, kind="stable"), kind="stable")
        hd = np.abs((trip.arc_bearing[arcs] - bearing[i] + 180) % 360 - 180)
        if hvalid[i] == 0:
            hd = np.full(m, 90.0)
        pos = t * trip.arc_len[arcs]
        for j, a in enumerate(arcs.tolist()):
            pv = maps[i - 1].get(a) if i > 0 else None
            nx = maps[i + 1].get(a) if i + 1 < n else None
            wd, wc = [], 0
            for q in range(max(0, i - win), min(n, i + win + 1)):
                if q == i:
                    continue
                v = maps[q].get(a)
                wc += v is not None
                wd.append(v[0] if v is not None else miss)
            along = (pos[j] - pv[1]) if pv is not None else 0.0
            rows.append([
                d[j], rank[j], d[j] - dmin, d[j] / (dmin + 1.0), hd[j], np.cos(np.radians(hd[j])),
                hvalid[i], speed[i], trip.arc_len[a], t[j], float(t[j] <= 0 or t[j] >= 1), m,
                float(pv is not None), float(nx is not None), pv[0] if pv else miss, nx[0] if nx else miss,
                float(np.mean(wd)) if wd else miss, wc / max(1, len(wd)), float(np.clip(along, -50, 50)),
                trip.out_deg[trip.arcs[a, 1]], trip.in_deg[trip.arcs[a, 0]], dt_prev[i],
            ])
            pid.append(i)
            aid.append(a)
    X = np.asarray(rows, dtype=np.float32).reshape(-1, len(FEATURES))
    return X, np.asarray(pid, dtype=np.int64), np.asarray(aid, dtype=np.int64)
