import os
import numpy as np

EARTH_R = 6371008.8


def list_trips(root):
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))


def _load(path, dtype=float, cols=None):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    a = np.loadtxt(path, dtype=dtype, ndmin=2)
    if cols is not None and a.shape[1] != cols:
        raise ValueError(f"{path}: expected {cols} columns, got {a.shape[1]}")
    return a


class Trip:
    def __init__(self, root, trip_id, load_route=False):
        base = os.path.join(root, trip_id, trip_id)
        self.trip_id = trip_id
        track = _load(base + ".track", float, 3)
        nodes = _load(base + ".nodes", float, 2)
        arcs = _load(base + ".arcs", np.int64, 2)
        track = track[np.isfinite(track).all(axis=1)]
        track = track[np.argsort(track[:, 2], kind="stable")]
        keep = np.concatenate([[True], np.diff(track[:, 2]) > 0])
        self.raw_n = len(track)
        track = track[keep]
        self.track_lonlat = track[:, :2]
        self.t = track[:, 2]
        self.nodes_lonlat = nodes
        self.arcs = arcs
        self.lat0 = float(np.mean(track[:, 1]))
        self.lon0 = float(np.mean(track[:, 0]))
        self.xy = self.project(track[:, 0], track[:, 1])
        self.nxy = self.project(nodes[:, 0], nodes[:, 1])
        self.a_xy0 = self.nxy[arcs[:, 0]]
        self.a_xy1 = self.nxy[arcs[:, 1]]
        d = self.a_xy1 - self.a_xy0
        self.arc_len = np.hypot(d[:, 0], d[:, 1])
        self.arc_bearing = np.degrees(np.arctan2(d[:, 0], d[:, 1])) % 360
        n = len(nodes)
        self.out_deg = np.bincount(arcs[:, 0], minlength=n)
        self.in_deg = np.bincount(arcs[:, 1], minlength=n)
        self.route = None
        if load_route:
            r = _load(base + ".route", np.int64, 1)
            self.route = None if r is None else r[:, 0]

    def project(self, lon, lat):
        k = np.pi / 180.0
        x = EARTH_R * np.cos(self.lat0 * k) * (np.asarray(lon) - self.lon0) * k
        y = EARTH_R * (np.asarray(lat) - self.lat0) * k
        return np.column_stack([x, y])


def point_segment(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    t = np.where(l2 > 0, ((px - ax) * dx + (py - ay) * dy) / np.where(l2 > 0, l2, 1), 0.0)
    tc = np.clip(t, 0, 1)
    qx, qy = ax + tc * dx, ay + tc * dy
    return np.hypot(px - qx, py - qy), tc
