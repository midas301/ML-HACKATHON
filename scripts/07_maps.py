import os
import sys
import numpy as np
import folium

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap import config
from pathsnap.data import list_trips
from pathsnap.inference import Predictor, server_side_scores

cfg = config.load()
root = sys.argv[1] if len(sys.argv) > 1 else cfg["student_test_dir"]
label = sys.argv[2] if len(sys.argv) > 2 else "student_test"
os.makedirs(cfg["maps_dir"], exist_ok=True)
pred = Predictor(cfg)


def arcs_line(trip, arcs):
    nl = trip.nodes_lonlat
    pts = []
    for e in arcs:
        a, b = trip.arcs[e]
        if not pts or pts[-1] != (nl[a, 1], nl[a, 0]):
            if pts:
                yield pts
                pts = []
            pts.append((nl[a, 1], nl[a, 0]))
        pts.append((nl[b, 1], nl[b, 0]))
    if pts:
        yield pts


for tid in list_trips(root):
    res = pred.predict(root, tid)
    trip = res["trip"]
    sc = server_side_scores(root, tid, trip, res, cfg["label_max_dist"])
    route = np.loadtxt(os.path.join(root, tid, tid + ".route"), dtype=np.int64, ndmin=1)
    lo, hi = trip.track_lonlat.min(0) - 0.002, trip.track_lonlat.max(0) + 0.002
    m = folium.Map(tiles="OpenStreetMap")
    m.fit_bounds([[lo[1], lo[0]], [hi[1], hi[0]]])
    mid = trip.nodes_lonlat[trip.arcs].mean(1)
    inside = np.where((mid[:, 0] > lo[0]) & (mid[:, 0] < hi[0]) & (mid[:, 1] > lo[1]) & (mid[:, 1] < hi[1]))[0]
    net = folium.FeatureGroup("Road network", show=True)
    for e in inside:
        a, b = trip.arcs[e]
        folium.PolyLine([(trip.nodes_lonlat[a, 1], trip.nodes_lonlat[a, 0]), (trip.nodes_lonlat[b, 1], trip.nodes_lonlat[b, 0])], color="#888888", weight=1, opacity=0.6).add_to(net)
    net.add_to(m)
    ref = folium.FeatureGroup("Reference route (ground truth)")
    for seg in arcs_line(trip, route):
        folium.PolyLine(seg, color="#2b9348", weight=8, opacity=0.6).add_to(ref)
    ref.add_to(m)
    clf = folium.FeatureGroup("Classifier-only arcs", show=False)
    p = res["clf"]
    for e in np.unique(p[p >= 0]):
        a, b = trip.arcs[e]
        folium.PolyLine([(trip.nodes_lonlat[a, 1], trip.nodes_lonlat[a, 0]), (trip.nodes_lonlat[b, 1], trip.nodes_lonlat[b, 0])], color="#f4a261", weight=4).add_to(clf)
    clf.add_to(m)
    hmm = folium.FeatureGroup("Predicted route (classifier + HMM)")
    for seg in arcs_line(trip, res["route"]):
        folium.PolyLine(seg, color="#d00000", weight=3).add_to(hmm)
    hmm.add_to(m)
    gps = folium.FeatureGroup("Raw GPS")
    step = max(1, len(trip.xy) // 1500)
    for lon, lat in trip.track_lonlat[::step]:
        folium.CircleMarker((lat, lon), radius=2, color="#1f6feb", fill=True, fill_opacity=0.8, weight=1).add_to(gps)
    gps.add_to(m)
    folium.Marker((trip.track_lonlat[0, 1], trip.track_lonlat[0, 0]), tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker((trip.track_lonlat[-1, 1], trip.track_lonlat[-1, 0]), tooltip="End", icon=folium.Icon(color="red")).add_to(m)
    legend = f"""<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;padding:10px;border:1px solid #999;font-size:13px">
    <b>Trip {tid} ({label})</b><br>
    <span style="color:#1f6feb">&#9679;</span> Raw GPS<br>
    <span style="color:#2b9348">&#9644;</span> Reference route<br>
    <span style="color:#f4a261">&#9644;</span> Classifier-only arcs<br>
    <span style="color:#d00000">&#9644;</span> Predicted route (HMM)<br>
    Top-1: clf {sc['clf_top1_accuracy']:.3f} / HMM {sc['hmm_top1_accuracy']:.3f}<br>
    Continuity: clf {sc['clf_route_continuity']:.3f} / HMM {sc['hmm_route_continuity']:.3f}</div>"""
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    out = os.path.join(cfg["maps_dir"], f"{label}_{tid}.html")
    m.save(out)
    print(out, {k: round(v, 3) for k, v in sc.items() if isinstance(v, float)}, flush=True)
