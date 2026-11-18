import numpy as np
from .candidates import gps_kinematics, FEATURES

DIRECTION_FEATURES = ["hd_long", "cos_long", "long_valid", "hd_prev_fast", "dt_prev_fast", "hd_next_fast", "dt_next_fast"]
ALL_FEATURES = FEATURES + DIRECTION_FEATURES
N_BASE = len(FEATURES)


def _bearing(a, b):
    d = b - a
    return np.degrees(np.arctan2(d[0], d[1])) % 360


def long_heading(xy, t, min_disp=20.0, max_dt=60.0):
    n = len(xy)
    h = np.full(n, np.nan)
    for i in range(n):
        j = i
        while j > 0 and t[i] - t[j - 1] <= max_dt:
            j -= 1
            if np.hypot(*(xy[i] - xy[j])) >= min_disp:
                break
        k = i
        while k < n - 1 and t[k + 1] - t[i] <= max_dt:
            k += 1
            if np.hypot(*(xy[k] - xy[i])) >= min_disp:
                break
        if np.hypot(*(xy[k] - xy[j])) >= min_disp:
            h[i] = _bearing(xy[j], xy[k])
    return h


def carried_heading(trip, fast_speed=5.0, cap=600.0):
    speed, bearing, valid, _ = gps_kinematics(trip)
    fast = (speed > fast_speed) & (valid > 0)
    n = len(speed)
    t = trip.t
    prev_b = np.full(n, np.nan); prev_dt = np.full(n, cap)
    last = -1
    for i in range(n):
        if fast[i]:
            last = i
        if last >= 0:
            prev_b[i] = bearing[last]; prev_dt[i] = min(cap, t[i] - t[last])
    next_b = np.full(n, np.nan); next_dt = np.full(n, cap)
    last = -1
    for i in range(n - 1, -1, -1):
        if fast[i]:
            last = i
        if last >= 0:
            next_b[i] = bearing[last]; next_dt[i] = min(cap, t[last] - t[i])
    return prev_b, prev_dt, next_b, next_dt


def _diff(arc_b, h):
    d = np.abs((arc_b - h + 180) % 360 - 180)
    return np.where(np.isnan(h), 90.0, d)


def direction_features(trip, pid, aid):
    hl = long_heading(trip.xy, trip.t)
    pb, pdt, nb, ndt = carried_heading(trip)
    ab = trip.arc_bearing[aid]
    hd_long = _diff(ab, hl[pid])
    return np.column_stack([
        hd_long, np.cos(np.radians(hd_long)), (~np.isnan(hl[pid])).astype(float),
        _diff(ab, pb[pid]), pdt[pid], _diff(ab, nb[pid]), ndt[pid],
    ]).astype(np.float32)
