const $ = (id) => document.getElementById(id);
const state = { api: "", trips: [], results: {}, current: null, running: false };

const map = L.map("map", { zoomControl: true, preferCanvas: true }).setView([20, 0], 2);
const bases = {
  streets: L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 19, attribution: "&copy; OpenStreetMap contributors" }),
  light: L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", { maxZoom: 19, subdomains: "abcd", attribution: "&copy; OpenStreetMap contributors &copy; CARTO" }),
};
let base = bases.streets.addTo(map);
let tileErrors = 0;
Object.values(bases).forEach((b) => {
  b.on("tileerror", () => { tileErrors += 1; if (tileErrors > 4) $("map-note").hidden = false; });
  b.on("tileload", () => { tileErrors = 0; $("map-note").hidden = true; });
});
document.querySelectorAll('input[name="base"]').forEach((r) => r.addEventListener("change", () => {
  if (base) map.removeLayer(base);
  base = bases[r.value] || null;
  if (base) base.addTo(map);
  $("map-note").hidden = true;
}));

const layers = { gps: L.layerGroup().addTo(map), route: L.layerGroup().addTo(map), snap: L.layerGroup() };
[["lyr-gps", "gps"], ["lyr-route", "route"], ["lyr-snap", "snap"]].forEach(([id, k]) => {
  $(id).addEventListener("change", (e) => (e.target.checked ? layers[k].addTo(map) : map.removeLayer(layers[k])));
});

const pct = (v) => (v === null || v === undefined ? "n/a" : (100 * v).toFixed(1) + "%");
const plateNum = (t) => { const m = String(t).match(/^(0*?)(0|[1-9][0-9]*)$/); return m ? `<span class="lz">${m[1]}</span>${m[2]}` : String(t); };
const fix = (v, d = 3) => (v === null || v === undefined ? "n/a" : Number(v).toFixed(d));

function setConn(kind, text) {
  $("conn").className = "conn " + kind;
  $("conn-text").textContent = text;
}

async function call(path, opts) {
  const r = await fetch(state.api + path, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch (_) {}
    throw new Error(`${r.status}: ${msg}`);
  }
  return r.json();
}

async function connect(ev) {
  if (ev) ev.preventDefault();
  state.api = $("api").value.trim().replace(/\/$/, "");
  setConn("", "Connecting");
  try {
    const [info, trips] = await Promise.all([call("/info"), call("/trips")]);
    state.trips = trips.trips;
    $("model-line").textContent = `${info.model} with ${info.n_features} features, HMM decoder, candidates within ${info.candidate_radius_m} m`;
    setConn("ok", `Connected, ${state.trips.length} trips`);
    renderPlates();
    $("run-all").disabled = state.trips.length === 0;
    $("trips-hint").textContent = state.trips.length ? "Pick a trip, or evaluate all of them in order." : "The API's evaluation folder has no trips. Check PATHSNAP_EVAL_DIR on the API laptop.";
  } catch (e) {
    setConn("err", "Not connected");
    $("trips-hint").textContent = `Can't reach the API at ${state.api}. Check that uvicorn is running on the API laptop with --host 0.0.0.0, that the IP and port are right, and that the firewall allows the port. (${e.message})`;
  }
}

function renderPlates() {
  const list = $("trip-list");
  list.innerHTML = "";
  state.trips.forEach((t) => {
    const li = document.createElement("li");
    const b = document.createElement("button");
    b.className = "plate-btn";
    b.id = "plate-" + t;
    b.innerHTML = `<span class="num">${plateNum(t)}</span><span class="state">Not evaluated</span><span class="fill"><i></i></span>`;
    b.addEventListener("click", () => { if (!state.running) evaluate(t, true); });
    li.appendChild(b);
    list.appendChild(li);
  });
}

function markPlate(t, res, running) {
  const b = $("plate-" + t);
  if (!b) return;
  b.classList.toggle("running", !!running);
  if (running) { b.querySelector(".state").textContent = "Matching"; return; }
  if (!res) return;
  const v = res.metrics ? res.metrics.hmm_top1_accuracy : null;
  b.querySelector(".state").textContent = v === null ? "Done" : `${pct(v)} correct`;
  b.querySelector(".fill i").style.width = v === null ? "100%" : `${100 * v}%`;
  b.classList.remove("done", "mid", "low");
  b.classList.add(v === null || v >= 0.9 ? "done" : v >= 0.8 ? "mid" : "low");
}

function select(t) {
  state.current = t;
  document.querySelectorAll(".plate-btn").forEach((b) => b.setAttribute("aria-current", b.id === "plate-" + t ? "true" : "false"));
}

async function evaluate(t, show) {
  select(t);
  markPlate(t, null, true);
  $("busy").hidden = false;
  $("busy-text").textContent = `Matching trip ${t}`;
  try {
    const res = await call("/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ trip_id: t }) });
    state.results[t] = res;
    markPlate(t, res, false);
    renderSummary();
    if (show) { draw(res); renderResults(res); }
    return res;
  } catch (e) {
    markPlate(t, null, false);
    $("plate-" + t).querySelector(".state").textContent = "Failed";
    $("res-empty").hidden = false;
    $("res").hidden = true;
    $("res-empty").textContent = `Trip ${t} failed on the server: ${e.message}. The API terminal shows the full error.`;
    return null;
  } finally {
    $("busy").hidden = true;
  }
}

async function runAll() {
  state.running = true;
  $("run-all").disabled = true;
  for (const t of state.trips) {
    const res = await evaluate(t, false);
    if (res) { draw(res); renderResults(res); }
  }
  state.running = false;
  $("run-all").disabled = false;
}

function draw(d) {
  Object.values(layers).forEach((l) => l.clearLayers());
  const step = Math.max(1, Math.floor(d.gps.length / 2000));
  const r = L.canvas({ padding: 0.5 });
  d.gps.forEach((p, i) => { if (i % step === 0) L.circleMarker(p, { renderer: r, radius: 2.5, color: "#2f5bff", weight: 0, fillOpacity: 0.75 }).addTo(layers.gps); });
  d.snapped_points.forEach((p, i) => { if (p && i % step === 0) L.circleMarker(p, { renderer: r, radius: 2.5, color: "#0e8a7e", weight: 0, fillOpacity: 0.9 }).addTo(layers.snap); });
  const line = L.polyline(d.predicted_route, { color: "#c8102e", weight: 4, opacity: 0.9 }).addTo(layers.route);
  const pin = (p, label, fill) => L.circleMarker(p, { radius: 7, color: "#14284b", weight: 2, fillColor: fill, fillOpacity: 1 }).bindTooltip(label).addTo(layers.gps);
  pin(d.gps[0], "Start", "#ffffff");
  pin(d.gps[d.gps.length - 1], "End", "#14284b");
  const bounds = L.latLngBounds(d.gps);
  if (d.predicted_route.length) bounds.extend(line.getBounds());
  map.fitBounds(bounds, { padding: [30, 30] });
}

function barRow(cls, label, v) {
  return `<div class="bar-row ${cls}"><span class="lbl">${label}</span><span class="track"><i style="width:${v === null ? 0 : 100 * v}%"></i></span><span class="val">${pct(v)}</span></div>`;
}

function pair(id, clf, hmm) {
  const delta = clf !== null && hmm !== null ? 100 * (hmm - clf) : null;
  $(id).innerHTML = barRow("clf", "Classifier only", clf) + barRow("hmm", "With HMM", hmm) +
    (delta === null ? "" : `<p class="delta ${delta < 0 ? "neg" : ""}">HMM ${delta >= 0 ? "adds" : "loses"} ${Math.abs(delta).toFixed(1)} points</p>`);
}

function renderResults(d) {
  const m = d.metrics || {};
  $("res-empty").hidden = true;
  $("res").hidden = false;
  $("res-plate").innerHTML = plateNum(d.trip_id);
  $("res-name").textContent = `Trip ${d.trip_id}`;
  $("res-meta").textContent = `${d.n_points.toLocaleString()} GPS points, ${(m.scored_points ?? 0).toLocaleString()} scored, ${d.predicted_route_arcs.length} road segments in the predicted route`;
  pair("bars-acc", m.clf_top1_accuracy ?? null, m.hmm_top1_accuracy ?? null);
  pair("bars-cont", m.clf_route_continuity ?? null, m.hmm_route_continuity ?? null);
  const rows = [
    ["Segments on the reference route (precision)", pct(m.hmm_route_precision)],
    ["Reference segments recovered (recall)", pct(m.hmm_route_recall)],
    ["Route segment F1", pct(m.hmm_route_f1)],
    ["Reference length covered", pct(m.route_length_overlap_recall)],
    ["Predicted length that is correct", pct(m.route_length_overlap_precision)],
    ["Points with no correct candidate", pct(m.candidate_miss_rate)],
    ["Decoder restarts at network gaps", d.viterbi_breaks],
  ];
  $("route-stats").innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
  const t = d.timings_s;
  const parts = [["Load", t.load_s, "#9aa8bd"], ["Features", t.features_s, "#2f5bff"], ["Classifier", t.classifier_s, "#14284b"], ["HMM", t.hmm_s, "#c8102e"]];
  const total = parts.reduce((a, p) => a + p[1], 0) || 1;
  $("time-bar").innerHTML = parts.map(([, v, c]) => `<span style="width:${(100 * v) / total}%;background:${c}"></span>`).join("");
  $("time-bar").setAttribute("aria-label", `Total ${t.total_inference_s} seconds`);
  $("time-legend").innerHTML = parts.map(([k, v, c]) => `<li><i style="background:${c}"></i>${k} <b>${fix(v, 2)} s</b></li>`).join("") + `<li>Total <b>${fix(t.total_inference_s, 2)} s</b></li>`;
}

function renderSummary() {
  const done = state.trips.filter((t) => state.results[t]);
  const body = $("sum-table").querySelector("tbody");
  const foot = $("sum-table").querySelector("tfoot");
  if (!done.length) { body.innerHTML = '<tr class="empty"><td colspan="4">No trips evaluated yet</td></tr>'; foot.innerHTML = ""; return; }
  body.innerHTML = done.map((t) => {
    const m = state.results[t].metrics || {};
    return `<tr><td>${t}</td><td>${pct(m.hmm_top1_accuracy)}</td><td>${pct(m.hmm_route_continuity)}</td><td>${pct(m.hmm_route_f1)}</td></tr>`;
  }).join("");
  let hit = 0, n = 0, c = 0, f = 0;
  done.forEach((t) => { const m = state.results[t].metrics || {}; hit += (m.hmm_top1_accuracy || 0) * (m.scored_points || 0); n += m.scored_points || 0; c += m.hmm_route_continuity || 0; f += m.hmm_route_f1 || 0; });
  foot.innerHTML = `<tr><td>All ${done.length}</td><td>${pct(n ? hit / n : null)}</td><td>${pct(c / done.length)}</td><td>${pct(f / done.length)}</td></tr>`;
  $("download").disabled = false;
}

function downloadCsv() {
  const cols = ["trip_id", "n_points", "scored_points", "clf_top1_accuracy", "hmm_top1_accuracy", "clf_route_continuity", "hmm_route_continuity", "hmm_route_precision", "hmm_route_recall", "hmm_route_f1", "route_length_overlap_recall", "candidate_miss_rate", "viterbi_breaks", "total_inference_s"];
  const lines = [cols.join(",")];
  state.trips.filter((t) => state.results[t]).forEach((t) => {
    const d = state.results[t], m = d.metrics || {};
    const v = { trip_id: t, n_points: d.n_points, viterbi_breaks: d.viterbi_breaks, total_inference_s: d.timings_s.total_inference_s, ...m };
    lines.push(cols.map((k) => (v[k] ?? "")).join(","));
  });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/csv" }));
  a.download = "pathsnap_evaluation.csv";
  a.click();
}

$("connect-form").addEventListener("submit", connect);
$("run-all").addEventListener("click", runAll);
$("download").addEventListener("click", downloadCsv);
const params = new URLSearchParams(location.search);
if (params.get("api")) {
  $("api").value = params.get("api");
  connect().then(() => { const t = params.get("trip"); if (t && state.trips.includes(t)) evaluate(t, true); });
}
