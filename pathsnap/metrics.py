import os
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, log_loss

_ARCS = {}


def trip_arcs(root, tid):
    key = (root, tid)
    if key not in _ARCS:
        _ARCS[key] = np.loadtxt(os.path.join(root, tid, tid + ".arcs"), dtype=np.int64, ndmin=2)
    return _ARCS[key]


def top1_choice(pid, prob, n_points, aid):
    order = np.lexsort((-prob, pid))
    p_sorted = pid[order]
    first = np.concatenate([[True], p_sorted[1:] != p_sorted[:-1]])
    best = order[first]
    pred = np.full(n_points, -1, dtype=np.int64)
    pred[pid[best]] = aid[best]
    chosen = np.zeros(len(pid), dtype=np.int8)
    chosen[best] = 1
    return pred, chosen


def continuity(pred, arcs):
    p = pred[pred >= 0]
    if len(p) < 2:
        return 1.0
    a, b = p[:-1], p[1:]
    ok = (a == b) | (arcs[a, 1] == arcs[b, 0])
    return float(ok.mean())


def sequence_scores(pred, route):
    p = pred[pred >= 0]
    if len(p) == 0:
        return 0.0, 0.0, 0.0
    seq = p[np.concatenate([[True], p[1:] != p[:-1]])]
    ps, rs = set(seq.tolist()), set(route.tolist())
    inter = len(ps & rs)
    prec = inter / max(1, len(ps))
    rec = inter / max(1, len(rs))
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return prec, rec, f1


def evaluate(trips, probs, root=None):
    ys, ps, chosen_all, top1_hits, valid_n = [], [], [], 0, 0
    cont, seqf = [], []
    per_trip = []
    preds = []
    for d, prob in zip(trips, probs):
        pred, chosen = top1_choice(d["pid"], prob, d["n_points"], d["aid"])
        preds.append(pred)
        v = d["label_valid"]
        hits = int((pred[v] == d["label_arc"][v]).sum())
        top1_hits += hits
        valid_n += int(v.sum())
        rv = d["row_valid"]
        ys.append(d["y"][rv]); ps.append(prob[rv]); chosen_all.append(chosen[rv])
        row = {"trip_id": d["trip_id"], "top1": hits / max(1, v.sum())}
        if root is not None:
            arcs = trip_arcs(root, d["trip_id"])
            row["continuity"] = continuity(pred, arcs)
            route = np.loadtxt(os.path.join(root, d["trip_id"], d["trip_id"] + ".route"), dtype=np.int64, ndmin=1)
            row["seq_precision"], row["seq_recall"], row["seq_f1"] = sequence_scores(pred, route)
            cont.append(row["continuity"]); seqf.append(row["seq_f1"])
        per_trip.append(row)
    y = np.concatenate(ys); p = np.clip(np.concatenate(ps), 1e-7, 1 - 1e-7); c = np.concatenate(chosen_all)
    t05 = (p >= 0.5).astype(int)
    res = {
        "top1": top1_hits / max(1, valid_n),
        "macro_f1": f1_score(y, c, average="macro"),
        "macro_precision": precision_score(y, c, average="macro", zero_division=0),
        "macro_recall": recall_score(y, c, average="macro", zero_division=0),
        "macro_f1_at_0.5": f1_score(y, t05, average="macro"),
        "accuracy_at_0.5": accuracy_score(y, t05),
        "log_loss": log_loss(y, p, labels=[0, 1]),
    }
    if cont:
        res["continuity"] = float(np.mean(cont))
        res["seq_f1"] = float(np.mean(seqf))
    return res, per_trip, preds
