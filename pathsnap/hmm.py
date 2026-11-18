import heapq
import numpy as np

NEG = -1e18


class RoadGraph:
    def __init__(self, arcs, arc_len):
        self.arcs = arcs
        self.arc_len = arc_len
        n = int(arcs.max()) + 1 if len(arcs) else 0
        order = np.argsort(arcs[:, 0], kind="stable")
        self.out_arcs = order
        self.ptr = np.searchsorted(arcs[order, 0], np.arange(n + 1))
        self.cache = {}

    def dijkstra(self, src, limit):
        c = self.cache.get(src)
        if c is not None and c[0] >= limit:
            return c[1], c[2]
        dist = {src: 0.0}
        pred = {}
        heap = [(0.0, src)]
        arcs, alen, oa, ptr = self.arcs, self.arc_len, self.out_arcs, self.ptr
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, np.inf) or d > limit:
                continue
            if u + 1 >= len(ptr):
                continue
            for e in oa[ptr[u]:ptr[u + 1]]:
                v = int(arcs[e, 1])
                nd = d + alen[e]
                if nd < dist.get(v, np.inf) and nd <= limit:
                    dist[v] = nd
                    pred[v] = int(e)
                    heapq.heappush(heap, (nd, v))
        self.cache[src] = (limit, dist, pred)
        return dist, pred

    def path_arcs(self, a, b, limit=5000.0):
        if a == b:
            return [a]
        src, dst = int(self.arcs[a, 1]), int(self.arcs[b, 0])
        dist, pred = self.dijkstra(src, limit)
        if dst not in dist:
            return [a, b]
        mid = []
        v = dst
        while v != src:
            e = pred[v]
            mid.append(e)
            v = int(self.arcs[e, 0])
        return [a] + mid[::-1] + [b]


def decode(graph, xy, t, pid, aid, pos, prob, params):
    beta = params.get("beta", 30.0)
    w_e = params.get("w_emission", 1.0)
    vmax = params.get("vmax", 50.0)
    back_tol = params.get("back_tol", 15.0)
    break_pen = params.get("break_penalty", 30.0)
    n = len(xy)
    order = np.argsort(pid, kind="stable")
    pid_s = pid[order]
    starts = np.searchsorted(pid_s, np.arange(n + 1))
    states = []
    for i in range(n):
        idx = order[starts[i]:starts[i + 1]]
        if len(idx) == 0:
            states.append(None)
            continue
        p = prob[idx] + 1e-9
        p = p / p.sum()
        states.append((aid[idx], pos[idx], w_e * np.log(p)))
    score = None
    backs = [None] * n
    prev_i = None
    breaks = 0
    for i in range(n):
        st = states[i]
        if st is None:
            continue
        a_cur, p_cur, e_cur = st
        if score is None:
            score = e_cur.copy()
            backs[i] = (None, np.full(len(a_cur), -1))
            prev_i = i
            continue
        a_prev, p_prev, _ = states[prev_i]
        gc = float(np.hypot(*(xy[i] - xy[prev_i])))
        dt = max(float(t[i] - t[prev_i]), 1.0)
        limit = max(200.0, 3.0 * gc + 100.0, min(vmax * dt, 5000.0))
        T = np.full((len(a_prev), len(a_cur)), NEG)
        for u in range(len(a_prev)):
            au, pu = int(a_prev[u]), float(p_prev[u])
            rem = graph.arc_len[au] - pu
            dist = None
            for v in range(len(a_cur)):
                av, pv = int(a_cur[v]), float(p_cur[v])
                if au == av and pv >= pu - back_tol:
                    d = max(pv - pu, 0.0)
                else:
                    if dist is None:
                        dist, _ = graph.dijkstra(int(graph.arcs[au, 1]), limit)
                    sp = dist.get(int(graph.arcs[av, 0]))
                    if sp is None:
                        continue
                    d = rem + sp + pv
                if d / dt > vmax:
                    continue
                T[u, v] = -abs(d - gc) / (beta * np.sqrt(dt))
        cand = score[:, None] + T
        best = cand.argmax(axis=0)
        new = cand[best, np.arange(len(a_cur))]
        if new.max() <= NEG / 2:
            breaks += 1
            best = np.full(len(a_cur), int(score.argmax()))
            new = np.full(len(a_cur), score.max() - break_pen)
        score = new + e_cur
        backs[i] = (prev_i, best)
        prev_i = i
    pred = np.full(n, -1, dtype=np.int64)
    if score is None:
        return pred, breaks
    j = int(score.argmax())
    i = prev_i
    while i is not None:
        pred[i] = states[i][0][j]
        pi, b = backs[i]
        j = int(b[j]) if pi is not None else j
        i = pi
    last = -1
    for i in range(n):
        if pred[i] < 0:
            pred[i] = last
        else:
            last = pred[i]
    return pred, breaks


def route_from_points(graph, pred):
    p = pred[pred >= 0]
    if len(p) == 0:
        return []
    seq = p[np.concatenate([[True], p[1:] != p[:-1]])].tolist()
    route = [seq[0]]
    for b in seq[1:]:
        a = route[-1]
        if graph.arcs[a, 1] == graph.arcs[b, 0]:
            route.append(b)
        else:
            route.extend(graph.path_arcs(a, b)[1:])
    return route
