import os
import sys
import time
import logging
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap import config
from pathsnap.data import list_trips
from pathsnap.inference import Predictor, server_side_scores

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pathsnap.api")

cfg = config.load()
EVAL_DIR = os.environ.get("PATHSNAP_EVAL_DIR", cfg["eval_dir"])
predictor = Predictor(cfg)
log.info("model loaded: %s | hmm=%s | eval_dir=%s", type(predictor.model).__name__, predictor.hmm, EVAL_DIR)

app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    trip_id: str


@app.middleware("http")
async def access_log(request: Request, call_next):
    t0 = time.time()
    resp = await call_next(request)
    log.info("%s %s from %s -> %d in %.0f ms", request.method, request.url.path, request.client.host if request.client else "?", resp.status_code, 1000 * (time.time() - t0))
    return resp


@app.get("/health")
def health():
    return {"status": "ok", "model": type(predictor.model).__name__, "eval_dir": EVAL_DIR}


@app.get("/info")
def info():
    m = predictor.model
    params = m.get_params() if hasattr(m, "get_params") else {}
    keep = ("n_estimators", "learning_rate", "num_leaves", "min_child_samples", "subsample", "colsample_bytree", "reg_lambda", "reg_alpha")
    return {
        "model": type(m).__name__,
        "n_features": int(getattr(m, "n_features_in_", 0)),
        "hyperparameters": {k: params[k] for k in keep if k in params},
        "hmm": {k: v for k, v in predictor.hmm.items() if isinstance(v, (int, float))},
        "candidate_radius_m": cfg["radius"],
        "max_candidates": cfg["k"],
        "eval_dir": EVAL_DIR,
    }


@app.get("/trips")
def trips():
    if not os.path.isdir(EVAL_DIR):
        raise HTTPException(500, f"evaluation directory not found: {EVAL_DIR}")
    return {"trips": list_trips(EVAL_DIR)}


@app.post("/predict")
def predict(req: PredictRequest):
    if req.trip_id not in list_trips(EVAL_DIR):
        raise HTTPException(404, f"unknown trip {req.trip_id}")
    t0 = time.time()
    res = predictor.predict(EVAL_DIR, req.trip_id)
    infer_s = time.time() - t0
    trip = res["trip"]
    scores = server_side_scores(EVAL_DIR, req.trip_id, trip, res, cfg["label_max_dist"])
    nl = trip.nodes_lonlat
    route_coords = []
    for e in res["route"]:
        a, b = trip.arcs[e]
        if not route_coords or route_coords[-1] != [float(nl[a, 1]), float(nl[a, 0])]:
            route_coords.append([float(nl[a, 1]), float(nl[a, 0])])
        route_coords.append([float(nl[b, 1]), float(nl[b, 0])])
    snapped = []
    for i, e in enumerate(res["hmm"]):
        if e < 0:
            snapped.append(None)
            continue
        a, b = trip.arcs[e]
        p, q = trip.nxy[a], trip.nxy[b]
        d = q - p
        l2 = float(d @ d)
        t = 0.0 if l2 == 0 else float(np.clip((trip.xy[i] - p) @ d / l2, 0, 1))
        lon, lat = nl[a] + t * (nl[b] - nl[a])
        snapped.append([float(lat), float(lon)])
    out = {
        "trip_id": req.trip_id,
        "n_points": int(len(trip.xy)),
        "gps": [[float(lat), float(lon)] for lon, lat in trip.track_lonlat],
        "predicted_route": route_coords,
        "predicted_route_arcs": [int(e) for e in res["route"]],
        "snapped_points": snapped,
        "viterbi_breaks": int(res["breaks"]),
        "timings_s": {k: round(v, 3) for k, v in res["timings"].items()} | {"total_inference_s": round(infer_s, 3)},
        "metrics": scores,
    }
    log.info("trip=%s points=%d route_arcs=%d inference=%.2fs metrics=%s", req.trip_id, len(trip.xy), len(res["route"]), infer_s,
             {k: (round(v, 4) if isinstance(v, float) else v) for k, v in (scores or {}).items()})
    return out
