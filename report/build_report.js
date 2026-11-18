const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell, WidthType, ShadingType,
  ImageRun, LevelFormat, Footer, Header, PageNumber, TableOfContents, PageBreak, BorderStyle,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const D = JSON.parse(fs.readFileSync(path.join(ROOT, "results", "report_data.json"), "utf8"));
const FIG = (f) => path.join(ROOT, "figures", f);
const W = 9026;
const INK = "14284B";

const f4 = (v) => (v === null || v === undefined ? "N/A" : Number(v).toFixed(4));
const f3 = (v) => (v === null || v === undefined ? "N/A" : Number(v).toFixed(3));
const f1 = (v) => (v === null || v === undefined ? "N/A" : Number(v).toFixed(1));
const pc = (v, d = 1) => (v === null || v === undefined ? "N/A" : (100 * Number(v)).toFixed(d) + "%");
const pts = (a, b) => { const v = 100 * (b - a); return Math.abs(v) < 1 ? v.toFixed(2) : v.toFixed(1); };
const M = Object.fromEntries(D.models.map((m) => [m.key, m]));
const H = Object.fromEntries(D.hmm.map((h) => [h.model, h]));
const tr = D.parts.train, st = D.parts.student_test;
const fin = M.LightGBM_dir_tuned, un29 = M.LightGBM_dir;

function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`|\[\[[^\]]+\]\])/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else if (t.startsWith("`")) out.push(new TextRun({ text: t.slice(1, -1), font: "Consolas", size: 19, ...base }));
    else out.push(new TextRun({ text: t.slice(2, -2), highlight: "yellow", bold: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}
const P = (text, opts = {}) => new Paragraph({ children: runs(text), spacing: { after: 120, line: 276 }, ...opts });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)] });
const B = (text, level = 0) => new Paragraph({ numbering: { reference: "bullets", level }, children: runs(text), spacing: { after: 60, line: 264 } });
let listInstance = 0;
const NL = (items) => { listInstance += 1; return items.map((t) => new Paragraph({ numbering: { reference: "nums", level: 0, instance: listInstance }, children: runs(t), spacing: { after: 60, line: 264 } })); };
const QA = (q, paras) => [new Paragraph({ children: [new TextRun({ text: q, bold: true, color: INK })], spacing: { before: 120, after: 60 }, keepNext: true }), ...paras];
const PH = (text) => new Paragraph({
  children: [new TextRun({ text: "[TO DO] " + text, highlight: "yellow", bold: true })],
  spacing: { before: 120, after: 120 },
  border: { top: { style: BorderStyle.DASHED, size: 6, color: "B26B00" }, bottom: { style: BorderStyle.DASHED, size: 6, color: "B26B00" } },
});

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
let figN = 0;
function figure(file, caption, widthFrac = 1) {
  if (!fs.existsSync(FIG(file))) return [PH(`Missing figure ${file}`)];
  figN += 1;
  const { w, h } = pngSize(FIG(file));
  const width = Math.round(600 * widthFrac);
  const height = Math.round((width * h) / w);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120 }, children: [new ImageRun({ type: "png", data: fs.readFileSync(FIG(file)), transformation: { width, height }, altText: { title: caption, description: caption, name: file } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [new TextRun({ text: `Figure ${figN}. `, bold: true, size: 18, color: "4A5A74" }), ...runs(caption, { size: 18, color: "4A5A74", italics: true })] }),
  ];
}
let tabN = 0;
function table(headers, rows, widths, caption, opts = {}) {
  tabN += 1;
  const total = widths.reduce((a, b) => a + b, 0);
  const ws = widths.map((x) => Math.floor((x / total) * W));
  ws[ws.length - 1] += W - ws.reduce((a, b) => a + b, 0);
  const size = opts.size || 17;
  const cell = (t, i, head, hl) => new TableCell({
    width: { size: ws[i], type: WidthType.DXA },
    shading: { type: ShadingType.CLEAR, color: "auto", fill: head ? INK : hl ? "FFF3CD" : "FFFFFF" },
    margins: { top: 50, bottom: 50, left: 80, right: 80 },
    children: [new Paragraph({ alignment: i > 0 && opts.rightAlign !== false && /^[-+\d.%N/ ×x±]+$/.test(String(t)) ? AlignmentType.RIGHT : AlignmentType.LEFT, children: runs(String(t), { size, color: head ? "FFFFFF" : "1D2433", bold: head || undefined }) })],
  });
  const out = [];
  if (caption) out.push(new Paragraph({ keepNext: true, spacing: { before: 160, after: 80 }, children: [new TextRun({ text: `Table ${tabN}. `, bold: true, size: 18, color: "4A5A74" }), ...runs(caption, { size: 18, color: "4A5A74", italics: true })] }));
  out.push(new Table({
    width: { size: W, type: WidthType.DXA }, columnWidths: ws,
    rows: [new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, i, true)) }),
      ...rows.map((r) => new TableRow({ cantSplit: true, children: r.map((c, i) => cell(c, i, false, opts.highlightRow && opts.highlightRow(r))) }))],
  }));
  out.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
  return out;
}

const C = [];
const push = (...xs) => xs.flat().forEach((x) => C.push(x));

push(
  new Paragraph({ spacing: { before: 2400 }, alignment: AlignmentType.LEFT, children: [new TextRun({ text: "PathSnap", bold: true, size: 64, color: INK })] }),
  new Paragraph({ spacing: { after: 200 }, children: [new TextRun({ text: "Map-matching and route reconstruction from noisy GPS", size: 32, color: "4A5A74" })] }),
  new Paragraph({ spacing: { after: 600 }, children: [new TextRun({ text: "Technical report, UE24CS352A Machine Learning hackathon, PES University", size: 24 })] }),
  P("Team name: [[team name]]"),
  P("Team members (name, SRN): Kanchhit kumar (PES1UG24CS213), [[add teammates]]"),
  P("Section: [[section]]"),
  P("GitHub repository (public): [[https://github.com/<user>/<repo>]]"),
  P("Date: [[submission date]]"),
  new Paragraph({ spacing: { before: 600 }, children: runs(`**Headline result (student-test partition, 10 trips):** observation-level Top-1 accuracy ${f4(fin.test_top1)} from the classifier alone and ${f4(fin.test_hmm_top1)} after HMM/Viterbi decoding; route continuity ${f4(fin.test_hmm_continuity)}; nearest-arc baseline ${f4(D.dataset.student_test.nearest_arc_top1)}.`) }),
  new Paragraph({ children: [new PageBreak()] }),
  H1("Contents"),
  P("Right-click the table below and choose Update Field if it appears empty."),
  new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
  new Paragraph({ children: [new PageBreak()] }),
);

push(
  H1("1. Overview"),
  P(`PathSnap reconstructs the road route a vehicle drove from noisy GPS fixes and a road network. Every GPS fix is compared with nearby road segments (arcs), a classifier scores how likely each candidate arc is the one actually driven, and an HMM decoded with the Viterbi algorithm chooses a sequence of arcs that is consistent with the road network. The system is deployed as a FastAPI backend (Laptop 1) and a browser interface (Laptop 2); the reference route is read only on the server, after prediction, to compute scores.`),
  ...figure("pipeline_overview.png", "End-to-end pipeline. Labels for training are derived from the .route file; prediction never reads it."),
  P(`The work followed the order given in the hackathon guide: data exploration (Section 2), a five-model comparison on a shared feature set (Section 3), Bayesian optimisation of the selected model (Section 4), HMM/Viterbi reconstruction (Section 5), Folium maps (Section 6) and deployment (Section 7). One iteration is reported explicitly: error analysis after the first round showed that most mistakes picked the right road in the wrong direction at low speed, so seven direction features were added to the selected model (Section 2.5 and 3.4). Every number in this report is produced by the scripts in the repository; \`results/report_data.json\` collects them.`),
);

push(
  H1("2. Dataset understanding and exploratory data analysis"),
  P(`All exploration in this section uses the **train** (80 trips) and **student-test** (10 trips) partitions; the partition is stated for each table and figure. The evaluator-test partition was never opened. Scripts: \`scripts/01_eda.py\`, \`scripts/02_build_dataset.py\`, \`scripts/11_dataset_visual.py\`.`),
  H2("2.1 Raw data structure"),
  ...table(["File", "Columns (no header)", "Type", "Meaning"], [
    [".track", "longitude, latitude, time (s)", "float, float, float", "GPS fixes of one trip, nominally 1 Hz"],
    [".nodes", "longitude, latitude", "float, float", "Road-network junctions; row number = node ID"],
    [".arcs", "from_node, to_node", "int, int", "Directed road segments; row number = arc ID"],
    [".route", "arc ID", "int", "Reference route: arcs driven, in order"],
  ], [1.2, 2.6, 1.7, 3.5], "Raw files per trip (verified by reading every file of both partitions)."),
  P(`Each trip ships its own clipped road network, so node and arc IDs are local to a trip: arc 571 in one trip is unrelated to arc 571 in another. GPS rows are linked to the network only through geometry; \`.route\` lists arcs but not which GPS fix belongs to which arc.`),
  ...figure("dataset_explained.png", "Train trip 00000048 (train partition). (a) GPS fixes, reference route and network; (b) 140 m zoom showing directed arcs, nodes, GPS fixes, derived labels and the eight candidate arcs within 50 m of one fix; (c) GPS-to-reference distance over 40 train trips with sampling-interval inset; (d) trip centroids worldwide (train and student-test)."),
  ...table(["Quantity", "Train", "Student test"], [
    ["Trips", tr.trips, st.trips],
    ["GPS fixes (total)", tr.points.toLocaleString(), st.points.toLocaleString()],
    ["GPS fixes per trip (min / mean / max)", `${tr.points_min} / ${Math.round(tr.points_mean)} / ${tr.points_max}`, `${st.points_min} / ${Math.round(st.points_mean)} / ${st.points_max}`],
    ["Mean trajectory length (km) / max", `${f1(tr.len_km_mean)} / ${f1(tr.len_km_max)}`, `${f1(st.len_km_mean)} / ${f1(st.len_km_max)}`],
    ["Road nodes per trip (min to max)", `${tr.nodes_min.toLocaleString()} to ${tr.nodes_max.toLocaleString()}`, `${st.nodes_min.toLocaleString()} to ${st.nodes_max.toLocaleString()}`],
    ["Road arcs per trip (min to max)", `${tr.arcs_min.toLocaleString()} to ${tr.arcs_max.toLocaleString()}`, `${st.arcs_min.toLocaleString()} to ${st.arcs_max.toLocaleString()}`],
    ["File sizes: .track / .nodes / .arcs (MB)", `${f1(tr.track_mb)} / ${f1(tr.nodes_mb)} / ${f1(tr.arcs_mb)}`, `${f1(st.track_mb)} / ${f1(st.nodes_mb)} / ${f1(st.arcs_mb)}`],
    ["Median arc length (m)", f1(tr.arclen_med), f1(st.arclen_med)],
    ["Arcs with a reverse twin (two-way roads)", pc(tr.bidir), pc(st.bidir)],
    ["Weakly connected components per trip (median / max)", `${tr.weak_med} / ${tr.weak_max}`, `${st.weak_med} / ${st.weak_max}`],
    ["Isolated nodes", tr.isolated, st.isolated],
    ["Consecutive route arcs that connect head-to-tail", pc(tr.route_conn, 2), pc(st.route_conn, 2)],
  ], [4, 2.2, 2.2], "Dataset inventory (scripts/01_eda.py; per-trip detail in results/eda_trip_inventory.csv)."),
  P(`Geographical extent: latitudes ${f1(D.extent.lat_min)} to ${f1(D.extent.lat_max)} and longitudes ${f1(D.extent.lon_min)} to ${f1(D.extent.lon_max)}. Trips lie on several continents, so no single projected coordinate system fits all of them (Figure 2d).`),
  H2("2.2 Missing-value and data-quality analysis"),
  ...table(["Check", "Train", "Student test"], [
    ["Missing / non-numeric values in .track", tr.nan, st.nan],
    ["Duplicate or backward timestamps", tr.dup, st.dup],
    ["Steps exactly 1 s apart (mean per trip)", pc(tr.dt1), pc(st.dt1)],
    ["Trips with at least one gap > 5 s", tr.gap_trips, st.gap_trips],
    ["Largest gap (s)", `${Math.round(tr.max_gap).toLocaleString()} (trip ${tr.max_gap_trip})`, `${Math.round(st.max_gap)} (trip ${st.max_gap_trip})`],
    ["Fixes > 50 m from their reference arc", pc(1 - D.dataset.train.label_valid_frac), pc(1 - D.dataset.student_test.label_valid_frac, 2)],
  ], [4, 2.4, 2.4], "Missing and invalid observations (results/eda_missing_invalid.csv)."),
  ...figure("eda_missing_invalid.png", "Missing and invalid observation counts by partition (log scale; zero-height bars are zero counts).", 0.75),
  ...figure("eda_sampling_intervals.png", "Seconds between consecutive GPS fixes, train partition (clipped at 20 s).", 0.7),
  P(`There are no missing values and no duplicate or backward timestamps, so the cleaning filters never fired on this data; they remain in the code for unseen trips. The real quality issues are elsewhere:`),
  B(`**Long gaps:** ${tr.gap_trips} of 80 train trips contain gaps over 5 s; train trip ${tr.max_gap_trip} has a gap of ${Math.round(tr.max_gap / 3600)} hours, which is effectively two journeys.`),
  B(`**Partial reference routes:** train trip ${D.far_trips[0].trip} has ${pc(D.far_trips[0].far, 0)} of its ${D.far_trips[0].points.toLocaleString()} fixes more than 50 m from the reference route and trip ${D.far_trips[1].trip} has ${pc(D.far_trips[1].far, 0)}. The reference route covers only part of these traces. Treating the reference as error-free ground truth would be wrong, as the dataset guide warns.`),
  B(`**Fragmented networks:** a median of ${tr.weak_med} weakly connected components per train trip, because each network is clipped from a larger map. Some consecutive fixes therefore cannot be joined by any path, which the decoder must handle (Section 5).`),
  H2("2.3 GPS trajectory and road-network visualisations"),
  ...figure("eda_trajectories_network.png", "GPS fixes (blue), reference route (green) and sampled road network (grey) for three train trips."),
  ...figure("eda_distributions.png", "GPS fixes per trip, trajectory length, median GPS-to-reference distance, and network size vs fixes (train and student-test)."),
  ...figure("eda_trip_centroids.png", "Trip centroids by partition (train and student-test).", 0.6),
  P(`An interactive Folium map of train trip 00000058 with the network, reference route and GPS fixes is committed as \`maps/eda_train_00000058.html\`. Folium maps of every student-test trip are shown in Section 6.`),
  H2("2.4 Question 1: How did we preprocess the data?"),
  ...NL([
    `**Read and clean** (\`pathsnap/data.py\`): drop non-finite rows, sort fixes by time, drop fixes whose timestamp does not increase. Nothing was removed on this data; the rules guard the evaluator trips.`,
    `**Project to metres:** distances in degrees are wrong and trips span continents, so each trip is projected to a local equirectangular plane centred on its own mean latitude and longitude. Trips have a median extent of 8.4 km (maximum 75 km); the east-west scale error is a median 0.05% and at most 0.66%, which is at most about 0.3 m on a 50 m distance. A per-trip UTM projection would be more exact but adds a dependency for no measurable benefit.`,
    `**Derive per-fix labels** (\`pathsnap/labels.py\`): \`.route\` gives arcs in order, not one label per fix. A monotonic dynamic-programming alignment assigns each fix to a route arc so that the route index never moves backwards and the total fix-to-arc distance is minimal. Unlike snapping each fix to its nearest route arc, this respects the order of travel when a route passes the same place twice.`,
    `**Mark unreliable labels:** fixes more than 50 m from their aligned arc are excluded from training and scoring (${pc(1 - D.dataset.train.label_valid_frac)} of train fixes, almost all from trips ${D.far_trips[0].trip} and ${D.far_trips[1].trip}; ${pc(1 - D.dataset.student_test.label_valid_frac, 2)} of student-test fixes). The threshold equals the candidate radius, so a fix the candidate generator could never get right is not counted as a model error.`,
    `**Generate candidates** (\`pathsnap/candidates.py\`): every arc is densified with a point every 8 m and indexed in a KD-tree; for each fix, up to k = 8 arcs within 50 m are kept (falling back to 200 m if none). This gives ${D.dataset.train.rows.toLocaleString()} candidate rows for train. The reference arc is missing from the candidate set for ${pc(D.dataset.train.candidate_miss_rate, 2)} of train fixes and ${pc(D.dataset.student_test.candidate_miss_rate, 2)} of student-test fixes (candidate miss rate), which caps achievable Top-1 accuracy.`,
    `**Build features** for every (fix, candidate) row (Section 2.5) and cache them per trip in \`artifacts/cache/\`. Scaling, where a model needs it, is fitted inside each training fold only.`,
  ]),
  H2("2.5 Question 2: Patterns observed and features extracted"),
  P(`**Patterns.** Sampling is mostly 1 Hz. Arcs are very short (median ${f1(tr.arclen_med)} m), so one street is split into many consecutive arcs, and ${pc(tr.bidir, 0)} of arcs have a reverse twin lying on the same line. GPS noise is a median of about 5 m with a long tail (Figure 2c). Snapping each fix to its nearest arc reaches only ${f4(D.dataset.train.nearest_arc_top1)} Top-1 on train and ${f4(D.dataset.student_test.nearest_arc_top1)} on student test, so distance alone is far from enough: most confusion is between adjacent short arcs and between the two directions of the same road.`),
  P(`**Features.** Each row describes one candidate arc for one GPS fix. Twenty-two base features were used for the five-model comparison; seven direction features were added after error analysis.`),
  ...table(["Group", "Features", "Why"], [
    ["Geometry", "dist, dist_rank, dist_minus_min, dist_ratio, proj_t, at_endpoint, arc_len", "Closeness relative to competing candidates matters more than absolute distance"],
    ["Heading", "heading_diff, heading_cos, heading_valid, speed", "Separates an arc from its reverse twin; heading is flagged invalid when the vehicle barely moved"],
    ["Temporal context", "prev_has, next_has, prev_dist, next_dist, win_dist, win_presence, along_delta", "A true arc stays close over consecutive fixes and the vehicle progresses along it"],
    ["Topology and sampling", "out_deg_to, in_deg_from, n_cand, dt_prev", "Junction density, ambiguity, and gaps"],
    ["Direction (added)", "hd_long, cos_long, long_valid, hd_prev_fast, dt_prev_fast, hd_next_fast, dt_next_fast", "Heading over the last ~20 m travelled, and direction carried from the nearest fast segment, so slow segments still have a reliable direction"],
  ], [1.4, 3.6, 3.6], "Engineered features (29 in the final model). None use the .route file."),
  P(`**Why the direction features were added.** Error analysis of the first tuned LightGBM (22 features) on student test showed that ${D.ablation.X_old.test_hmm_wrong_direction_errors} of its errors after HMM were the right road in the wrong direction, concentrated below 5 m/s. At 1 to 2 m/s the vehicle moves about 1 to 2 m per fix while GPS noise is about 5 m, so a heading measured over five fixes is noise. The new features measure heading over at least 20 m of displacement and borrow the direction of the nearest segment where the vehicle moved faster than 5 m/s. With identical hyperparameters, folds and HMM settings, cross-validation Top-1 rose from ${f4(D.ablation.X_old.cv_top1_mean)} to ${f4(D.ablation.X_new.cv_top1_mean)}, every fold improved, and wrong-direction errors fell from ${D.ablation.X_old.test_hmm_wrong_direction_errors} to ${D.ablation.X_new.test_hmm_wrong_direction_errors} (Figure 8). The long-window heading became the second most important feature of the final model (Figure 9).`),
  ...figure("feature_ablation_direction.png", "Feature ablation on student test and train CV: same tuned LightGBM, same folds, same HMM; only the seven direction features differ."),
  ...figure("feature_importance.png", "Gain importance of the final model; red bars are the direction features.", 0.7),
);

const base = ["LogReg", "KNN", "RandomForest", "MLP", "LightGBM"];
const allK = ["LogReg", "KNN", "RandomForest", "MLP", "LightGBM", "LightGBM_tuned22", "LightGBM_dir", "LightGBM_dir_tuned"];
const cfgShort = {
  LogReg: "Scaler + L2, C = 1", KNN: "Scaler, k = 25, distance-weighted, 150k-row subsample", RandomForest: "200 trees, min_leaf 5, max_samples 0.3",
  MLP: "Scaler, (64, 32) ReLU, Adam, 25 epochs", LightGBM: "300 trees, lr 0.05, 31 leaves", LightGBM_tuned22: "Optuna best (Section 4), 22 features",
  LightGBM_dir: "300 trees, lr 0.05, 31 leaves, 29 features", LightGBM_dir_tuned: "Optuna best, 29 features",
};
push(
  H1("3. Model experimentation and comparison"),
  P(`**Task framing.** Arc IDs are local to each trip, so the problem cannot be a multi-class classifier over arc IDs. Each (fix, candidate) row is a binary sample: 1 if the candidate is the fix's reference arc. At prediction time the candidate with the highest probability is chosen for each fix, and the probabilities feed the HMM. About ${pc(D.dataset.train.positive_rate, 0)} of rows are positive; classes were not reweighted, because reweighting distorts the probabilities the HMM uses.`),
  P(`**Protocol.** All models were fitted on the 80 train trips with 5-fold GroupKFold grouped by trip (no trip on both sides of a split), refitted on all 80 trips, and evaluated on the 10 student-test trips. The five-model comparison used the identical 22-feature set. Seeds are fixed at 42; configurations, fold scores and logs are in \`results/\`.`),
  P(`**Metrics.** Observation-level Top-1 accuracy: for each scored fix, the highest-probability candidate is compared with the reference arc (a fix whose reference arc is not a candidate counts as wrong). Macro precision, recall and F1: computed on candidate rows, where the predicted positive is the Top-1 candidate of each fix, averaged over the two classes. Log loss: binary cross-entropy on candidate rows. Route continuity: share of consecutive predictions that are the same arc or head-to-tail connected arcs. Route segment F1: overlap between the predicted arc set and the reference arc set. Accuracy at a 0.5 threshold is reported but is inflated, because most candidate rows are easy negatives.`),
  P(`**Primary selection metric:** cross-validation Top-1 accuracy, because the task is one choice per fix and it is measured on train folds only; cross-validation log loss breaks ties, because the HMM consumes the probabilities. Under the Top-1 decision rule macro F1 is close to a function of Top-1 accuracy, so it is reported for completeness rather than used for selection.`),
  H2("3.1 Overall model comparison"),
  ...table(["Model", "Family", "Key configuration", "CV Macro F1", "Top-1 (clf / HMM)", "Route contin. (clf / HMM)", "Test Macro F1", "Log Loss", "Inference ms / 1k fixes"],
    allK.map((k) => { const m = M[k]; return [m.display, m.family, cfgShort[k], `${f4(m.cv_macro_f1_mean)} ± ${f3(m.cv_macro_f1_std)}`, `${f4(m.test_top1)} / ${f4(m.test_hmm_top1)}`, `${f4(m.test_continuity)} / ${f4(m.test_hmm_continuity)}`, f4(m.test_macro_f1), f4(m.test_log_loss), f1(m.inference_ms_per_1k_obs)]; }),
    [1.6, 1.0, 1.9, 1.15, 1.2, 1.2, 0.9, 0.8, 0.9], "Consolidated comparison. Top-1, continuity, Macro F1 and log loss are on student test (10 trips); CV is 5-fold GroupKFold on 80 train trips. Rows 1 to 5: five-model comparison on identical 22 features. Rows 6 to 8: refinements of the selected model. Highlighted: final model.",
    { size: 15, highlightRow: (r) => r[0] === M.LightGBM_dir_tuned.display }),
  ...table(["Model", "CV Top-1 (mean ± sd)", "Accuracy @0.5", "Macro precision", "Macro recall", "Route F1 (HMM)", "Train time (s)"],
    allK.map((k) => { const m = M[k]; return [m.display, `${f4(m.cv_top1_mean)} ± ${f3(m.cv_top1_std)}`, f4(m["test_accuracy_at_0.5"]), f4(m.test_macro_precision), f4(m.test_macro_recall), f4(m.test_hmm_seq_f1), f1(m.train_seconds)]; }),
    [2.1, 1.5, 1.1, 1.1, 1.1, 1.1, 1.0], "Further metrics on student test. Training time is a single fit on all train rows (2 CPU cores).", { size: 15 }),
  P(`**Model selection.** On the shared 22 features, Random Forest and LightGBM tie on cross-validation Top-1 (${f4(M.RandomForest.cv_top1_mean)} vs ${f4(M.LightGBM.cv_top1_mean)}) and macro F1, with Logistic Regression and MLP within about 0.6 points and KNN last. The fold standard deviation (about 4 points) is far larger than any gap between the top four models, so the ranking among them is not statistically meaningful; it is driven by which noisy trips fall in each fold. LightGBM was selected over Random Forest because it trains about ${Math.round(M.RandomForest.train_seconds / M.LightGBM.train_seconds)} times faster, its model file is about 0.4 MB against 48 MB, its probabilities are well calibrated for the HMM (Figure 15), and it offers a richer space for Bayesian tuning. The largest single improvement in the project came from features, not from the choice of algorithm: the seven direction features added ${pts(M.LightGBM_tuned22.cv_top1_mean, fin.cv_top1_mean)} points of cross-validation Top-1, against ${pts(M.LightGBM.cv_top1_mean, M.LightGBM_tuned22.cv_top1_mean)} from tuning. The same features also lifted Logistic Regression in a partial re-run (CV Top-1 ${f4(M.LogReg.cv_top1_mean)} to ${f4(D.partial29.LogReg.cv_top1_mean)}; \`results/archive_partial_29features/\`), which confirms that the gain belongs to the features. The other four models were not re-run on 29 features because of time, so rows 7 and 8 must not be read as LightGBM beating them by 3 points.`),
);

const cardFig = (k) => figure(`model_card_${k}.png`, `${M[k].display}: CV Top-1 per fold, per-trip student-test Top-1 (classifier vs HMM), calibration on student test, and row-normalised confusion matrix.`);
const hist = D.hist;
const hU = hist["LightGBM + direction features (untuned)"], hT = hist["LightGBM + direction features (tuned, final)"], hM = hist["MLP (22 features)"];
const modelSec = [
  ["3.2.1 Logistic Regression", "LogReg", {
    why: `The decision is largely monotone in a few features: a closer candidate, a better heading match and presence across neighbouring fixes all raise the chance that a candidate is correct. A calibrated linear baseline measures how far such additive evidence goes.`,
    sl: [`Strengths: fastest model (${f1(M.LogReg.inference_ms_per_1k_obs)} ms per 1000 fixes), stable across folds, reasonable probabilities (log loss ${f4(M.LogReg.test_log_loss)}).`, `Limitations: cannot represent interactions such as "heading only matters when the vehicle moves", so it trails the tree models; its log loss is ${Math.round(100 * (M.LogReg.test_log_loss / M.LightGBM.test_log_loss - 1))}% worse than LightGBM's.`],
    hp: `StandardScaler, then L2-regularised logistic regression with C = 1.0, lbfgs solver, max_iter = 500. Values were library defaults; this is the linear baseline.`,
    loss: `Binary cross-entropy with an L2 penalty on the weights.`,
    fit: `Mild underfitting. Training-sample Top-1 (${f4(M.LogReg.train_top1)}) is not above CV Top-1 (${f4(M.LogReg.cv_top1_mean)}), and higher-capacity models gain over it on the same features, so model capacity, not overfitting, is the limit. The training sample contains trips with partial reference routes, which is why its score can fall below CV.`,
  }],
  ["3.2.2 K-nearest neighbours (distance-weighted)", "KNN", {
    why: `Candidate situations that look alike (for example "second-closest arc, heading opposite to travel") should share labels; KNN tests this with no parametric assumption.`,
    sl: [`Strengths: no training, non-parametric decision boundary.`, `Limitations: prediction cost grows with the training set. Even on a 150,000-row subsample it needs ${Math.round(M.KNN.inference_ms_per_1k_obs).toLocaleString()} ms per 1000 fixes, about ${Math.round(M.KNN.inference_ms_per_1k_obs / M.LightGBM.inference_ms_per_1k_obs)} times LightGBM, which is impractical for the API. It is sensitive to scaling and to the 22 dimensions, and its probabilities are coarse (at most 25 distinct levels), giving the worst log loss of the five (${f4(M.KNN.test_log_loss)}).`],
    hp: `StandardScaler, k = 25, distance weighting, Euclidean metric, trained on a random 150,000-row subsample (seed 42) because full-data neighbour search was infeasible on two cores.`,
    loss: `None: KNN is a lazy learner and stores the training rows.`,
    fit: `Slight overfitting: training-sample Top-1 ${f4(M.KNN.train_top1)} against CV ${f4(M.KNN.cv_top1_mean)}. Distance weighting gives a training row an exact match to itself, which inflates training scores.`,
  }],
  ["3.2.3 Random Forest", "RandomForest", {
    why: `The problem has nonlinear thresholds (for example relative distance under a few metres) and interactions, and averaging many trees is robust to the label noise introduced by imperfect reference routes.`,
    sl: [`Strengths: joint-best CV Top-1 among the five (${f4(M.RandomForest.cv_top1_mean)}) and the best test log loss among them (${f4(M.RandomForest.test_log_loss)}).`, `Limitations: slowest to train (${f1(M.RandomForest.train_seconds)} s), a 48 MB model file, ${f1(M.RandomForest.inference_ms_per_1k_obs)} ms per 1000 fixes, and probabilities from vote fractions.`],
    hp: `200 trees, min_samples_leaf = 5, max_features = sqrt, max_samples = 0.3 (bootstrap on 30% of rows, chosen for training time; it also regularises).`,
    loss: `Gini impurity at each split.`,
    fit: `Balanced: training-sample Top-1 ${f4(M.RandomForest.train_top1)} against CV ${f4(M.RandomForest.cv_top1_mean)}. The leaf-size limit and row subsampling kept trees from memorising rows.`,
  }],
  ["3.2.4 Multi-layer perceptron", "MLP", {
    why: `A neural network learns smooth nonlinear combinations of the features and tests whether learned representations beat tree splits on tabular candidate features.`,
    sl: [`Strengths: fast inference (${f1(M.MLP.inference_ms_per_1k_obs)} ms per 1000 fixes) and good probabilities (log loss ${f4(M.MLP.test_log_loss)}).`, `Limitations: needs scaling, training is stochastic, and its CV Top-1 is about 0.5 points below the tree ensembles.`],
    hp: `StandardScaler, hidden layers (64, 32) with ReLU, Adam with learning rate 1e-3, L2 alpha 1e-4, batch size 1024, 25 epochs, no early stopping.`,
    loss: `Binary cross-entropy with an L2 penalty.`,
    fit: `Mild overfitting late in training, shown by the epoch-wise curves (Figure 19): validation log loss on 16 held-out train trips reaches its minimum of ${f4(hM.valid_min)} at epoch ${hM.valid_argmin} and ends at ${f4(hM.valid_end)}, while training loss keeps falling from ${f4(hM.train_start)} to ${f4(hM.train_end)}. Early stopping near epoch ${hM.valid_argmin} would be appropriate.`,
  }],
  ["3.2.5 LightGBM (selected model)", "LightGBM", {
    why: `Gradient boosting corrects the hard boundary cases sequentially, histogram binning keeps it fast on about a million rows, and leaf-wise growth with subsampling and L1/L2 regularisation gives a rich space for Bayesian tuning.`,
    sl: [`Strengths: joint-best CV Top-1 (${f4(M.LightGBM.cv_top1_mean)}), small model, ${f1(M.LightGBM.train_seconds)} s training, well-calibrated probabilities (Figure 15).`, `Limitations: leaf-wise trees can overfit small groups; untuned it is not the fastest at inference.`],
    hp: `Untuned: 300 trees, learning rate 0.05, 31 leaves, min_child_samples 20, no subsampling, no regularisation. Tuned values are in Section 4.`,
    loss: `Binary log loss (gradient boosting on its gradients and Hessians).`,
    fit: `Little overfitting. For the final 29-feature version (Figure 19), the untuned model's validation log loss reaches ${f4(hU.valid_min)} at round ${hU.valid_argmin} and is ${f4(hU.valid_end)} at round 300 while training loss falls to ${f4(hU.train_end)}, so the last rounds overfit slightly. The tuned model, with a learning rate of 0.0106, is still improving at its last round (${f4(hT.valid_min)}). Training-sample Top-1 of the final model is ${f4(fin.train_top1)} against CV ${f4(fin.cv_top1_mean)}.`,
  }],
];
push(H2("3.2 Compulsory questions for every model"),
  P(`Answers refer to student-test results unless stated. Each model's per-model figure shows fold stability, per-trip results, calibration and the confusion matrix.`));
modelSec.forEach(([title, k, a]) => {
  const m = M[k];
  push(H3(title),
    ...QA("A1. Why did we consider this algorithm suitable for PathSnap?", [P(a.why)]),
    ...QA("A2. Major strengths and limitations for this task", a.sl.map((s) => B(s))),
    ...QA("B5. Important hyperparameters and initial values", [P(a.hp)]),
    ...QA("B6. Loss function or optimisation objective", [P(a.loss)]),
    ...QA("C7. Performance on the student-testing dataset", [P(`Classifier-only Top-1 ${f4(m.test_top1)}, ${f4(m.test_hmm_top1)} after HMM decoding; route continuity ${f4(m.test_continuity)} rising to ${f4(m.test_hmm_continuity)} with the HMM; route segment F1 ${f4(m.test_hmm_seq_f1)} after HMM.`)]),
    ...QA("C8. Observation-level Top-1 accuracy, Macro F1 and Log Loss (student test)", [P(`Top-1 accuracy ${f4(m.test_top1)}; Macro F1 ${f4(m.test_macro_f1)} (macro precision ${f4(m.test_macro_precision)}, macro recall ${f4(m.test_macro_recall)}); Log Loss ${f4(m.test_log_loss)}.`)]),
    ...QA("C9. Overfitting or underfitting, and the evidence", [P(a.fit)]),
    ...cardFig(k));
  if (k === "LightGBM") push(...cardFig("LightGBM_dir_tuned"));
});

push(
  H2("3.3 Model visualisations"),
  ...figure("cmp_cv_macro_f1.png", "Cross-validation Macro F1 across all models (5-fold GroupKFold, mean ± sd)."),
  ...figure("cmp_test_top1_f1.png", "Student-test Top-1 accuracy (classifier and with HMM) and Macro F1 across all models."),
  ...figure("cmp_log_loss.png", "Student-test log loss across all models (log scale)."),
  ...figure("training_histories.png", "Training and validation log loss for models with iterative training histories: LightGBM per boosting round (final features, untuned and tuned) and MLP per epoch. Validation = 16 held-out train trips. No curves are drawn for Logistic Regression, KNN or Random Forest, which do not produce epoch-wise histories."),
  ...figure("cmp_confusion_matrices.png", "Candidate-level confusion matrices with the Top-1 decision rule, student test."),
  ...figure("cmp_train_vs_cv.png", "Training-sample vs cross-validation Top-1 (over/underfitting check)."),
  ...figure("cmp_speed_accuracy.png", "Accuracy after HMM against classifier inference cost; supports choosing LightGBM over Random Forest and KNN.", 0.75),
);

const bo = D.bayes, bp = bo.best_params;
const imp = Object.entries(bo.importance).sort((a, b) => b[1] - a[1]);
push(
  H1("4. Bayesian hyperparameter optimisation (LightGBM)"),
  P(`Script: \`scripts/04_bayes_opt.py\`. Trials: \`results/bayes_opt_trials.csv\`; summary: \`results/bayes_opt_summary.json\`; best values: \`artifacts/best_lgbm_params.json\`.`),
  ...QA("1. Which hyperparameters were included in the search?", [P(`n_estimators, learning_rate, num_leaves, min_child_samples, subsample (with subsample_freq fixed to 1 so that it takes effect), colsample_bytree, reg_lambda and reg_alpha.`)]),
  ...QA("2. What ranges or distributions were considered?", table(["Hyperparameter", "Range", "Scale"], [
    ["n_estimators", "100 to 800", "log-uniform integer"], ["learning_rate", "0.01 to 0.2", "log-uniform"], ["num_leaves", "15 to 255", "log-uniform integer"],
    ["min_child_samples", "5 to 200", "log-uniform integer"], ["subsample", "0.5 to 1.0", "uniform"], ["colsample_bytree", "0.5 to 1.0", "uniform"],
    ["reg_lambda", "0.001 to 10", "log-uniform"], ["reg_alpha", "0.001 to 10", "log-uniform"],
  ], [2, 2, 2], "Search space.")),
  ...QA("3. What objective function was optimised?", [P(`Mean observation-level Top-1 accuracy over 3-fold GroupKFold (grouped by trip) on the 80 train trips, maximised with Optuna's Tree-structured Parzen Estimator (TPE) sampler, seed 42. TPE models which values led to good and bad trials and samples new values from the promising region, unlike grid or random search. Three folds were used for cost; the final comparison below uses five.`)]),
  ...QA("4. How many optimisation trials were performed?", [P(`${bo.n_trials} trials, taking ${Math.round(bo.search.search_seconds / 60)} minutes on two CPU cores. Trial 0 was set to the default configuration so the baseline appears in the same record.`)]),
  ...QA("5. What were the best hyperparameter values?", table(["Hyperparameter", "Untuned", "Tuned"], [
    ["n_estimators", 300, bp.n_estimators], ["learning_rate", "0.05", f4(bp.learning_rate)], ["num_leaves", 31, bp.num_leaves], ["min_child_samples", 20, bp.min_child_samples],
    ["subsample (subsample_freq)", "1.0 (0)", `${f3(bp.subsample)} (1)`], ["colsample_bytree", "1.0", f3(bp.colsample_bytree)], ["reg_lambda", "0", f3(bp.reg_lambda)], ["reg_alpha", "0", f3(bp.reg_alpha)],
  ], [2.5, 1.5, 1.5], `Best configuration (3-fold CV Top-1 ${f4(bo.search.best_value)}).`)),
  P(`fANOVA importance: ${imp.slice(0, 4).map(([k, v]) => `${k} ${v}`).join(", ")}. The best region is "many small steps": a low learning rate with more trees and leaves.`),
  ...figure("bayes_opt_history.png", "Optimisation history: each trial's 3-fold CV Top-1 and the best so far.", 0.8),
  ...figure("bayes_opt_param_importance.png", "Hyperparameter importance (fANOVA).", 0.6),
  ...QA("6. Did the tuned model outperform the untuned model?", [
    P(`Only marginally. All ${bo.n_trials} trials scored between ${f4(bo.trial_min)} and ${f4(bo.trial_max)} (standard deviation ${f4(bo.trial_std)}), so the objective surface is flat. The comparison below uses the same 5 folds, the same student-test trips, the same metrics and sequential timing for both models.`),
    ...table(["Metric", "Untuned (22 f)", "Tuned (22 f)", "Untuned (29 f)", "Tuned (29 f, final)"], [
      ["CV Macro F1 (5-fold)", f4(M.LightGBM.cv_macro_f1_mean), f4(M.LightGBM_tuned22.cv_macro_f1_mean), f4(un29.cv_macro_f1_mean), f4(fin.cv_macro_f1_mean)],
      ["CV Top-1 (5-fold)", f4(M.LightGBM.cv_top1_mean), f4(M.LightGBM_tuned22.cv_top1_mean), f4(un29.cv_top1_mean), f4(fin.cv_top1_mean)],
      ["Student-test Top-1", f4(M.LightGBM.test_top1), f4(M.LightGBM_tuned22.test_top1), f4(un29.test_top1), f4(fin.test_top1)],
      ["Student-test Top-1 with HMM", f4(M.LightGBM.test_hmm_top1), f4(M.LightGBM_tuned22.test_hmm_top1), f4(un29.test_hmm_top1), f4(fin.test_hmm_top1)],
      ["Student-test Macro F1", f4(M.LightGBM.test_macro_f1), f4(M.LightGBM_tuned22.test_macro_f1), f4(un29.test_macro_f1), f4(fin.test_macro_f1)],
      ["Student-test Log Loss", f4(M.LightGBM.test_log_loss), f4(M.LightGBM_tuned22.test_log_loss), f4(un29.test_log_loss), f4(fin.test_log_loss)],
      ["Training time (s)", f1(M.LightGBM.train_seconds), f1(M.LightGBM_tuned22.train_seconds), f1(un29.train_seconds), f1(fin.train_seconds)],
      ["Inference (ms / 1k fixes)", f1(M.LightGBM.inference_ms_per_1k_obs), f1(M.LightGBM_tuned22.inference_ms_per_1k_obs), f1(un29.inference_ms_per_1k_obs), f1(fin.inference_ms_per_1k_obs)],
    ], [2.4, 1.3, 1.3, 1.3, 1.5], "Untuned vs tuned LightGBM on both feature sets."),
    P(`On the 22-feature set, tuning gained ${pts(M.LightGBM.cv_top1_mean, M.LightGBM_tuned22.cv_top1_mean)} CV Top-1 points, left student-test Top-1 unchanged and improved log loss by ${(100 * (1 - M.LightGBM_tuned22.test_log_loss / M.LightGBM.test_log_loss)).toFixed(1)}%, at about ${Math.round(M.LightGBM_tuned22.inference_ms_per_1k_obs / M.LightGBM.inference_ms_per_1k_obs)} times the inference time. On the final 29 features the same values gained ${pts(un29.cv_top1_mean, fin.cv_top1_mean)} CV points and ${pts(un29.test_top1, fin.test_top1)} student-test points. The search was run on the 22-feature set and its best values were reused for the 29-feature model rather than searched again; this is a limitation. The tuned model was kept because the decision rule (cross-validation, fixed in advance) favoured it; the untuned 29-feature model is an equally defensible choice where latency matters.`),
  ]),
  ...figure("tuned_vs_untuned.png", "Final model: hyperparameters and results, untuned vs tuned (29 features)."),
);

const hc = D.hmm_cfg, hg = D.hmm_grid, eh = D.errors["classifier+HMM"], ec = D.errors.classifier;
const ptrip = D.per_trip;
const better = ptrip.filter((r) => r.hmm_top1 > r.clf_top1).length;
const worst = [...ptrip].sort((a, b) => a.hmm_top1 - b.hmm_top1)[0];
const bestGain = [...ptrip].sort((a, b) => (b.hmm_top1 - b.clf_top1) - (a.hmm_top1 - a.clf_top1)).slice(0, 3);
push(
  H1("5. HMM and Viterbi route reconstruction"),
  P(`Implementation: \`pathsnap/hmm.py\`; experiments: \`scripts/06_hmm_eval.py\`; configuration: \`artifacts/hmm_config.json\`.`),
  ...QA("1. What are the hidden states?", [P(`For each GPS fix, its candidate arcs (at most 8), each paired with the projected position of the fix along that arc in metres. The decoded state is directly an arc ID, so the candidate-to-road-segment correspondence is preserved and every fix with candidates receives a predicted arc.`)]),
  ...QA("2. How are emission probabilities calculated?", [P(`log P(fix | state) = w · log p̂, where p̂ is the final LightGBM probability renormalised over that fix's candidates and w is the emission weight. Distance and heading already enter the classifier, so no separate distance term is added; adding one would count the same evidence twice.`)]),
  ...QA("3. How are transition probabilities calculated?", [
    P(`For a move from (arc a, position u) to (arc b, position v), the network distance d is v − u on the same arc moving forward (with 15 m tolerance for backward GPS jitter), otherwise the rest of arc a plus the shortest directed path from the end of a to the start of b plus v. Shortest paths come from a bounded Dijkstra search over the directed graph, cached per source node, with bound max(200 m, 3 × straight-line distance + 100 m, min(50 m/s × Δt, 5 km)).`),
    P(`log P(b | a) = −|d − d_gc| / (β · √Δt), where d_gc is the straight-line distance between the two fixes: on a real move the two distances are close, and a large difference means an implausible detour. Moves faster than 50 m/s, or with no path inside the bound, are forbidden. The Viterbi recurrence is δ_t(j) = max_i [δ_{t−1}(i) + log P(j | i)] + log P(fix_t | j), with back-pointers for the best path.`),
    P(`**Parameter choice.** β and w were chosen by a ${hg.n}-point grid on out-of-fold probabilities of the final model for 40 train trips (student-test and evaluator-test data were not used). Only the product w·β affects the Viterbi argmax, because scaling all scores by a constant does not change the best path, so the grid is effectively one-dimensional. Results ranged from ${f4(hg.min)} to ${f4(hg.max)} (classifier alone ${f4(hg.clf)}); the selected values are β = ${hc.beta} m and w = ${hc.w_emission}.`),
    ...figure("hmm_grid.png", "HMM Top-1 for each (β, w) pair on train out-of-fold probabilities.", 0.55),
  ]),
  ...QA("4. How are missing candidates or large GPS gaps handled?", [P(`A fix with no candidate within 200 m is skipped by the recursion and inherits the previous decoded arc. For gaps, the transition scale grows with √Δt and the Dijkstra bound grows with Δt up to 5 km, so longer gaps allow longer detours. Fixes after a gap are kept, because each fix is still a genuine position; only the path between fixes is inferred.`)]),
  ...QA("5. What happens when consecutive observations suggest disconnected road segments?", [P(`If no transition into any current candidate is feasible (clipped network, GPS jump), Viterbi does not fail: it restarts from the best previous score minus a penalty of 30 and records a break. There were ${ptrip.reduce((a, r) => a + r.breaks, 0)} breaks across the 10 student-test trips. For the returned route geometry, consecutive decoded arcs that do not touch are joined by the shortest network path when one exists.`)]),
  ...QA("6. Did HMM decoding improve the final route?", [
    P(`Yes, for every model and on ${better} of 10 student-test trips for the final model. For the final model, Top-1 rose from ${f4(H.LightGBM_dir_tuned.clf_top1)} to ${f4(H.LightGBM_dir_tuned.hmm_top1)}, route continuity from ${f4(H.LightGBM_dir_tuned.clf_continuity)} to ${f4(H.LightGBM_dir_tuned.hmm_continuity)}, route segment precision from ${f4(H.LightGBM_dir_tuned.clf_seq_precision)} to ${f4(H.LightGBM_dir_tuned.hmm_seq_precision)} and route segment F1 from ${f4(H.LightGBM_dir_tuned.clf_seq_f1)} to ${f4(H.LightGBM_dir_tuned.hmm_seq_f1)}. Errors fell from ${ec.wrong} to ${eh.wrong} of ${eh.total} scored fixes.`),
    ...table(["Model", "Top-1 clf", "Top-1 HMM", "Continuity clf", "Continuity HMM", "Route F1 clf", "Route F1 HMM"],
      allK.map((k) => { const h = H[k]; return [M[k].display, f4(h.clf_top1), f4(h.hmm_top1), f4(h.clf_continuity), f4(h.hmm_continuity), f4(h.clf_seq_f1), f4(h.hmm_seq_f1)]; }),
      [2.4, 1, 1, 1.1, 1.1, 1, 1], "Classifier-only vs classifier + HMM/Viterbi on student test (pointwise and sequence-level).", { size: 15, highlightRow: (r) => r[0] === fin.display }),
    ...figure("hmm_vs_classifier.png", "Classifier-only vs classifier + HMM for every model: Top-1, route continuity and route segment F1 (student test)."),
    ...table(["Trip", "Fixes", "Top-1 clf", "Top-1 HMM", "Continuity HMM", "Route precision", "Route recall", "Breaks"],
      ptrip.map((r) => [r.trip_id, r.points.toLocaleString(), f4(r.clf_top1), f4(r.hmm_top1), f4(r.hmm_continuity), f4(r.hmm_seq_precision), f4(r.hmm_seq_recall), r.breaks]),
      [1.3, 0.9, 1, 1, 1.1, 1.1, 1, 0.7], "Per-trip results of the final model on student test.", { size: 15 }),
    ...figure("hmm_per_trip.png", "Per-trip Top-1 of the final model, classifier vs HMM (student test).", 0.85),
  ]),
  H2("5.1 Where the HMM helped and where it did not"),
  B(`**Largest gains:** ${bestGain.map((r) => `trip ${r.trip_id} (${f3(r.clf_top1)} to ${f3(r.hmm_top1)})`).join(", ")}. The HMM removes isolated wrong picks that would break the route, which is why route precision rises much more than route recall (${f4(H.LightGBM_dir_tuned.hmm_seq_precision)} vs ${f4(H.LightGBM_dir_tuned.hmm_seq_recall)}): it rarely adds wrong roads, but it cannot recover short reference arcs that no fix lands on cleanly.`),
  B(`**Hardest trip:** ${worst.trip_id} reaches only ${f4(worst.hmm_top1)} Top-1 even though its predicted route covers about all of the reference length (Figure 31). Its residual errors are fixes moving slowly along very short arcs, where the derived labels themselves are uncertain.`),
  B(`**Low speed remains the weakness.** After the HMM, Top-1 is ${D.speed.map((s) => `${f3(s.top1)} at ${s.band} m/s (${s.points.toLocaleString()} fixes)`).join(", ")}. At walking pace the movement between fixes is smaller than GPS noise, so neither the classifier nor the transition model can tell the two directions of a road apart reliably. Of the ${eh.wrong} remaining errors, ${eh.kinds.find((x) => x.kind.startsWith("same road")).n} are the right road in the wrong direction.`),
  B(`**A decoder change that did not help:** reducing the backward tolerance from 15 m to 5, 2 and 0.5 m (to stop slow traces drifting onto the reverse arc) lowered train out-of-fold HMM Top-1 from 0.897 to 0.775, 0.707 and 0.670, because GPS jitter while stopped then looks like repeated U-turns. The 15 m value was kept.`),
  ...table(["Error type (final model + HMM)", "Fixes", "Share of errors"], eh.kinds.map((k) => [k.kind, k.n, k.share.toFixed(1) + "%"]), [4, 1, 1.2], `Remaining errors after HMM on student test (${eh.wrong} of ${eh.total} fixes; scripts/14_error_analysis.py).`),
);

push(
  H1("6. Geographical visualisation using Folium"),
  P(`\`scripts/07_maps.py\` generates one interactive Folium map per student-test trip (\`maps/student_test_<trip>.html\`) from the final pipeline. Each map has: raw GPS fixes (blue circles), the road-network geometry around the trip (grey lines), the reference route (wide green line), classifier-only arcs (orange, off by default), the HMM-predicted route (red line), start and end markers, a legend with the trip's scores, a layer control to toggle every layer, and an extent fitted to the trajectory. The legend shows classifier and HMM Top-1 and continuity so the observed trajectory and the predicted route can be compared directly.`),
  ...figure("folium_screenshot_00000077.png", "Exported view of the Folium map for student-test trip 00000077 (layers: road network, reference route, predicted route, raw GPS; layer control top right, legend bottom left). Rendered headless without internet, so the OpenStreetMap background tiles are absent; the interactive HTML shows them."),
  PH("Optional: replace the figure above with a screenshot of maps/student_test_00000077.html opened in a browser with internet, so the OpenStreetMap background is visible."),
  ...figure("folium_screenshot_00000056.png", "Exported view of the Folium map for student-test trip 00000056."),
  ...figure("route_static_00000077.png", "Static detail view of trip 00000077 drawn from the same pipeline: reference route (green), classifier-only arcs (orange), HMM route (red), GPS (blue).", 0.75),
);

push(
  H1("7. API, UI and deployment"),
  P(`**Architecture.** Laptop 1 runs \`uvicorn api.app:app --host 0.0.0.0 --port 8000\` with \`PATHSNAP_EVAL_DIR\` pointing at the evaluation trips. It loads \`artifacts/final_model.joblib\` and \`artifacts/hmm_config.json\` once at start-up. Laptop 2 serves \`ui/\` with \`python -m http.server\` and the evaluator opens it in a browser with the API address. Leaflet is bundled with the UI, so the map draws without internet; only background tiles need it.`),
  ...table(["Method", "Path", "Purpose"], [
    ["GET", "/health", "Status, loaded model, evaluation folder"],
    ["GET", "/info", "Model type, number of features, hyperparameters, HMM settings"],
    ["GET", "/trips", "Evaluation trip IDs (folder names only)"],
    ["POST", "/predict {trip_id}", "Runs the full pipeline on the trip, then computes scores server-side; returns GPS, predicted route polyline and arc IDs, snapped fixes, timings and metrics"],
  ], [1, 2, 6], "API endpoints."),
  P(`**Ground truth stays on the server.** \`Predictor.predict\` reads only \`.track\`, \`.nodes\` and \`.arcs\`; \`server_side_scores\` reads \`.route\` after prediction to compute Top-1 accuracy (classifier and HMM), route continuity, route segment precision, recall and F1, route-length overlap and candidate miss rate. The UI receives only these scores, never reference geometry. Nothing is precomputed or hardcoded: each request runs the whole pipeline. Every request is logged in Laptop 1's terminal with client IP, status, latency and per-trip metrics.`),
  P(`**UI.** The interface lists the evaluation trips as plates that fill with each trip's score once evaluated, runs one trip or all of them, draws raw GPS, the predicted route and snapped fixes on a map with layer and background controls, and shows classifier-only vs HMM accuracy and continuity side by side, route precision, recall and F1, candidate miss rate, decoder restarts and a processing-time breakdown. A run summary table pools results across evaluated trips and downloads as CSV.`),
  ...figure("../docs/ui_screenshot.png", "Development screenshot of the UI connected to the API on one machine (student-test trip 00000040; no internet, so no background tiles)."),
  PH("Insert a screenshot of the UI on Laptop 2 during the two-laptop run, showing an evaluation trip and its results."),
  PH("Insert a screenshot of the API terminal (CMD) on Laptop 1 showing the logged /trips and /predict requests and their metrics."),
  P(`Example log lines from the local end-to-end test (\`docs/api_log_sample.txt\`):`),
  ...D.api_log.slice(0, 5).map((l) => new Paragraph({ spacing: { after: 0 }, children: [new TextRun({ text: l.length > 150 ? l.slice(0, 150) + " ..." : l, font: "Consolas", size: 14 })] })),
  new Paragraph({ spacing: { after: 160 }, children: [] }),
  P(`**Deployment files:** \`requirements.txt\` (pinned scikit-learn version for loading the saved models), \`config.json\` (paths, radius, k, label threshold, seed), \`run_all.sh\` (full reproduction order) and \`README.md\` (installation, execution, deployment and file map).`),
);

push(
  H1("8. Limitations and future work"),
  B(`**Labels are derived.** Per-fix labels come from our alignment of \`.route\` to the GPS trace; near junctions and on 5 to 25 m arcs they are uncertain, so pointwise Top-1 understates route quality (trip ${worst.trip_id}).`),
  B(`**Reference routes are imperfect.** Two train trips have references that cover only part of the trace; their unmatched fixes were excluded rather than repaired.`),
  B(`**Unequal feature sets in the comparison.** Only LightGBM was re-run with the 29 features. A complete study would re-run all five models and repeat the Bayesian search on the final features.`),
  B(`**Model differences are small relative to fold variance.** About 4 points of fold-to-fold spread against under 1 point between the top four models.`),
  B(`**Low-speed direction ambiguity** remains the main error source; a direction-change penalty inside the HMM is the obvious next step.`),
  B(`**Candidate set not tuned.** The 50 m radius and k = 8 cap accuracy through the ${pc(D.dataset.student_test.candidate_miss_rate, 2)} miss rate on student test.`),
  B(`**Offline features.** The carried-direction features use the next fast segment, which is valid for whole-trip processing as in the API but not for live, point-by-point matching.`),
  H1("9. Conclusion"),
  P(`On student test, the final system (tuned LightGBM on 29 features with HMM/Viterbi decoding) reaches ${f4(fin.test_hmm_top1)} observation-level Top-1 accuracy and ${f4(fin.test_hmm_continuity)} route continuity, against ${f4(D.dataset.student_test.nearest_arc_top1)} for nearest-arc snapping. Above 10 m/s it is ${f3(D.speed[D.speed.length - 1].top1)} accurate. The experiments show that feature engineering (+${pts(M.LightGBM_tuned22.cv_top1_mean, fin.cv_top1_mean)} CV points) and sequential decoding (+${pts(H.LightGBM_dir_tuned.clf_top1, H.LightGBM_dir_tuned.hmm_top1)} test points) mattered far more than the choice among strong classifiers or Bayesian tuning (+${pts(M.LightGBM.cv_top1_mean, M.LightGBM_tuned22.cv_top1_mean)} CV points).`),
  H1("10. AI assistance disclosure"),
  PH("Required by the hackathon policy. Replace this paragraph with an accurate statement of which AI tools your team used, for which parts (for example code, analysis, figures, report text), and how the team verified and understood the results. Do not leave this section unedited."),
  H1("Appendix A. Reproducibility"),
  P(`Seeds are 42 throughout. \`./run_all.sh\` reruns the whole study in order. Without retraining, \`python scripts/13_show_metrics.py --live-all\` reloads every saved model and recomputes its student-test scores, and \`python scripts/12_show_preprocessed.py train 00000048 3\` prints preprocessed candidate rows. Key result files: \`results/model_comparison.csv\`, \`results/hmm_vs_classifier_student_test.csv\`, \`results/bayes_opt_trials.csv\`, \`results/feature_ablation_direction.json\`, \`results/error_analysis.txt\`, \`results/report_data.json\`. Earlier 22-feature results are archived in \`results/archive_v1_22features/\`.`),
);

const doc = new Document({
  creator: "PathSnap team", title: "PathSnap technical report",
  features: { updateFields: true },
  styles: {
    default: {
      document: { run: { font: "Arial", size: 21 } },
      title: { run: { font: "Arial", size: 56, bold: true, color: INK }, paragraph: { spacing: { before: 0, after: 200 }, keepNext: true } },
      heading1: { run: { font: "Arial", size: 32, bold: true, color: INK }, paragraph: { spacing: { before: 360, after: 160 }, keepNext: true } },
      heading2: { run: { font: "Arial", size: 26, bold: true, color: INK }, paragraph: { spacing: { before: 280, after: 120 }, keepNext: true } },
      heading3: { run: { font: "Arial", size: 23, bold: true, color: "C8102E" }, paragraph: { spacing: { before: 240, after: 100 }, keepNext: true } },
    },
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 360, hanging: 260 } } } }] },
      { reference: "nums", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 360, hanging: 300 } } } }] },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1300, left: 1440, right: 1440 } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: "PathSnap technical report", size: 16, color: "7D8AA0" })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: ["Page ", PageNumber.CURRENT], size: 16, color: "7D8AA0" })] })] }) },
    children: C,
  }],
});
Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(path.join(ROOT, "REPORT.docx"), buf);
  console.log("REPORT.docx written,", figN, "figures,", tabN, "tables");
});
