import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap import config

cfg = config.load()
steps = [
    ("Raw trip files", ".track .nodes .arcs", "#e8edf3"),
    ("Clean + project", "sort, dedupe, metres", "#e8edf3"),
    ("Candidates", "arcs within 50 m, k = 8", "#dfe9ff"),
    ("29 features", "distance, heading,\ndirection, context", "#dfe9ff"),
    ("Classifier", "tuned LightGBM\nP(candidate is correct)", "#14284b"),
    ("HMM + Viterbi", "network-consistent\nroute decoding", "#c8102e"),
    ("Route + scores", "API -> UI", "#e8edf3"),
]
fig, ax = plt.subplots(figsize=(15, 3.2))
ax.set_xlim(0, len(steps) * 2.1); ax.set_ylim(0, 3); ax.axis("off")
for i, (t, s, c) in enumerate(steps):
    x = 0.15 + i * 2.1
    dark = c in ("#14284b", "#c8102e")
    ax.add_patch(FancyBboxPatch((x, 0.75), 1.75, 1.5, boxstyle="round,pad=0.02,rounding_size=0.12", fc=c, ec="#14284b", lw=1.2))
    ax.text(x + 0.875, 1.85, t, ha="center", va="center", fontsize=12, weight="bold", color="white" if dark else "#14284b")
    ax.text(x + 0.875, 1.2, s, ha="center", va="center", fontsize=9.5, color="white" if dark else "#4a5a74")
    if i < len(steps) - 1:
        ax.add_patch(FancyArrowPatch((x + 1.78, 1.5), (x + 2.12, 1.5), arrowstyle="-|>", mutation_scale=14, color="#14284b", lw=1.4))
ax.text(0.15, 0.3, "Labels for training come from aligning each GPS point to the .route file (train and student-test only). The .route file is never read during prediction.", fontsize=9.5, color="#4a5a74")
plt.tight_layout()
plt.savefig(os.path.join(cfg["figures_dir"], "pipeline_overview.png"), dpi=150)
