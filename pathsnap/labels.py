import numpy as np
from .data import point_segment


def align_route(trip, chunk=512):
    r = trip.route
    a0, a1 = trip.a_xy0[r], trip.a_xy1[r]
    n, m = len(trip.xy), len(r)
    back = np.zeros((n, m), dtype=np.int32)
    prev = None
    dists = np.zeros((n, m), dtype=np.float32)
    for s in range(0, n, chunk):
        p = trip.xy[s:s + chunk]
        d, _ = point_segment(p[:, :1], p[:, 1:], a0[None, :, 0], a0[None, :, 1], a1[None, :, 0], a1[None, :, 1])
        dists[s:s + chunk] = d
    idx = np.arange(m)
    for i in range(n):
        if prev is None:
            cost = dists[i].astype(np.float64)
            back[i] = idx
        else:
            pm = np.minimum.accumulate(prev)
            is_new = np.concatenate([[True], prev[1:] < pm[:-1]])
            arg = np.maximum.accumulate(np.where(is_new, idx, 0))
            back[i] = arg
            cost = dists[i] + pm
        prev = cost
    j = int(np.argmin(prev))
    path = np.empty(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        path[i] = j
        j = back[i, j]
    label_arc = r[path]
    label_dist = dists[np.arange(n), path]
    return label_arc, label_dist.astype(float), path
