"""Draw the report and slide figures from the pipeline outputs (artifacts/trust.json, reports/metrics.json).

    python scripts/make_figures.py
"""
import json

import numpy as np
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
T = json.loads((ROOT / "artifacts" / "trust.json").read_text(encoding="utf8"))
M = T["metrics"]
CI = T["ci"]

NAVY, BLUE, AMBER, GREEN, RED, GREY, LIGHT = "#232B30", "#B8432F", "#86650F", "#365F48", "#A8392A", "#5D6768", "#F6F5F1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#CDCBC2", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.labelcolor": NAVY, "xtick.color": GREY, "ytick.color": GREY})


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor="none", transparent=True)
    plt.close(fig)


# 1. model comparison (ROC-AUC with 95% intervals)
order = [("rule_recency", "Days since last order (rule)", AMBER), ("tree_depth4", "Decision tree (depth 4)", GREY),
         ("logreg_cal", "Logistic regression", GREEN), ("mlp_challenger", "Neural network (MLP)", GREY),
         ("svm_rbf", "SVM (RBF kernel)", GREY), ("random_forest", "Random forest", GREEN), ("bgnbd", "BG/NBD (classic model)", GREY), ("lgbm_cal", "LightGBM (served)", BLUE)]
fig, ax = plt.subplots(figsize=(7.6, 3.6))
for i, (k, label, col) in enumerate(order):
    y = len(order) - 1 - i
    lo, hi = CI[f"{k}|roc_auc"]
    ax.plot([lo, hi], [y, y], color=col, lw=3, solid_capstyle="round")
    ax.scatter([M[k]["roc_auc"]], [y], color=col, s=70, zorder=3)
    ax.text(0.855, y, f"{M[k]['roc_auc']:.3f}", va="center", ha="left", color=NAVY, fontweight="bold" if k == "lgbm_cal" else "normal")
ax.set_yticks(range(len(order)))
ax.set_yticklabels([o[1] for o in order][::-1])
ax.axvline(0.5, color=GREY, ls=":", lw=1)
ax.text(0.5, len(order) - 0.35, "coin flip", ha="center", color=GREY, fontsize=9)
ax.set_xlim(0.48, 0.9)
ax.set_xlabel("ROC-AUC on the out-of-time test period (95% interval)")
ax.grid(axis="x", color="#DDDBD3", lw=0.8)
ax.set_axisbelow(True)
save(fig, "fig_models.png")

# 2. catch-rate curve
g = T["gains"]
fig, ax = plt.subplots(figsize=(5.4, 4.2))
ax.plot(g["x"], g["x"], color=GREY, ls="--", lw=1.6, label="Random calling")
ax.plot(g["x"], g["rule_recency"], color=AMBER, lw=2.4, label="Rule: longest since last order first")
if "bgnbd" in g:
    ax.plot(g["x"], g["bgnbd"], color=GREEN, lw=2, label="BG/NBD")
ax.plot(g["x"], g["lgbm_cal"], color=BLUE, lw=2.8, label="This model")
ax.set_xlabel("Share of customers contacted (highest risk first)")
ax.set_ylabel("Share of lapsing customers reached")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.01)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.grid(color="#DDDBD3", lw=0.8)
ax.legend(loc="lower right", frameon=False, fontsize=8.5)
save(fig, "fig_gains.png")

# 3. reliability + prevalence shift
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.8), gridspec_kw={"width_ratios": [1.15, 1]})
a1.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1.4, label="Perfect calibration")
for key, col, lab in (("lgbm_raw", RED, "Raw model"), ("lgbm_cal", BLUE, "After calibration (served)")):
    pts = T["reliability"][key]
    a1.plot([p["predicted"] for p in pts], [p["observed"] for p in pts], marker="o", color=col, lw=2, label=lab)
a1.set_xlabel("Predicted probability of no purchase")
a1.set_ylabel("Observed rate of no purchase")
a1.set_xlim(0, 1)
a1.set_ylim(0, 1)
a1.grid(color="#DDDBD3", lw=0.8)
a1.legend(frameon=False, fontsize=8.5, loc="upper left")
blocks = [("train", "Training"), ("val", "Validation"), ("cal", "Calibration"), ("test", "Test")]
vals = [T["sizes"][b]["prevalence"] for b, _ in blocks]
cols = [BLUE, GREEN, AMBER, RED]
a2.barh([n for _, n in blocks][::-1], vals[::-1], color=cols[::-1])
for i, v in enumerate(vals[::-1]):
    a2.text(v + 0.01, i, f"{v:.0%}", va="center", color=NAVY)
a2.set_xlim(0, 0.75)
a2.set_xlabel("Share of customers who went quiet")
a2.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
a2.set_title("Why calibration drifted", color=NAVY, fontsize=10)
save(fig, "fig_calibration.png")

# 4. split timeline
import datetime as dt
sp = T["splits"]
pts = [(dt.date.fromisoformat(d), b) for b, ds in sp["blocks"].items() for d in ds] + \
      [(dt.date.fromisoformat(d), "dropped") for d in sp["dropped_by_embargo"]]
pts.sort()
colmap = {"train": BLUE, "val": GREEN, "cal": AMBER, "test": RED}
fig, ax = plt.subplots(figsize=(9.2, 1.9))
ax.axhline(0, color="#CDCBC2", lw=1.4)
for d, b in pts:
    if b == "dropped":
        ax.scatter([d], [0], s=190, facecolors="white", edgecolors=GREY, linestyle="--", linewidths=1.6, zorder=3)
    else:
        ax.scatter([d], [0], s=190, color=colmap[b], zorder=3)
for lab, col in (("Train", BLUE), ("Validation", GREEN), ("Calibration", AMBER), ("Test", RED)):
    ax.scatter([], [], s=90, color=col, label=lab)
ax.scatter([], [], s=90, facecolors="white", edgecolors=GREY, label="Dropped: outcome not yet known")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=5, frameon=False, fontsize=8.5)
ax.set_ylim(-1, 1)
ax.set_yticks([])
for s in ("left",):
    ax.spines[s].set_visible(False)
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %y"))
save(fig, "fig_timeline.png")

# 5. segments
seg = [s for s in T["segments"]]
labels = [f"{s['group']}" for s in seg]
fig, ax = plt.subplots(figsize=(7.2, 3.8))
colors = [BLUE if s["roc_auc"] >= 0.7 else AMBER for s in seg]
y = range(len(seg))
ax.barh(list(y)[::-1], [s["roc_auc"] for s in seg], color=colors)
ax.set_yticks(list(y)[::-1])
ax.set_yticklabels([f"{s['dimension'].split(' (')[0]}: {s['group']}" for s in seg], fontsize=8.5)
for i, s in zip(list(y)[::-1], seg):
    ax.text(s["roc_auc"] + 0.005, i, f"{s['roc_auc']:.2f}  (n={s['n']:,})", va="center", fontsize=8.5, color=NAVY)
ax.axvline(0.5, color=GREY, ls=":", lw=1)
ax.set_xlim(0.45, 0.92)
ax.set_xlabel("ROC-AUC within the group (test period)")
ax.grid(axis="x", color="#DDDBD3", lw=0.8)
ax.set_axisbelow(True)
save(fig, "fig_segments.png")

# 6. snapshot concept
fig, ax = plt.subplots(figsize=(9.2, 2.3))
ax.set_xlim(0, 100)
ax.set_ylim(0, 24)
ax.axis("off")
ax.add_patch(Rectangle((4, 9), 48, 7, fc="#E3E0D6", ec=BLUE, lw=1.5))
ax.text(28, 12.5, "Features: everything up to the scoring date", ha="center", va="center", color=NAVY, fontsize=9.5)
ax.add_patch(Rectangle((59, 9), 39, 7, fc="#F1DDD6", ec=RED, lw=1.5))
ax.text(78.5, 12.5, "Label: any order in the next 90 days?", ha="center", va="center", color=NAVY, fontsize=8.6)
ax.plot([56, 56], [6, 19], color=NAVY, lw=2)
ax.text(56, 21.2, "Scoring date (month-end)", ha="center", color=NAVY, fontweight="bold", fontsize=10)
ax.text(28, 6, "Eligible if they ordered in the last 180 days", ha="center", color=GREY, fontsize=9)
ax.text(78, 6, "Used only after the 90 days have passed", ha="center", color=GREY, fontsize=9)
ax.annotate("", xy=(97, 3), xytext=(4, 3), arrowprops=dict(arrowstyle="->", color=GREY))
ax.text(50, 0.5, "time", ha="center", color=GREY, fontsize=9)
save(fig, "fig_snapshot.png")

# 7. architecture
fig, ax = plt.subplots(figsize=(9.6, 3.1))
ax.set_xlim(0, 100)
ax.set_ylim(0, 30)
ax.axis("off")
boxes = [("Online Retail II\n(download + cache)", 2), ("Clean\ndedupe, returns,\nnon-product codes", 21), ("Snapshots\nfeatures + labels\n(no future data)", 40),
         ("Models\nrule, LR, forest,\nMLP, BG/NBD,\nLightGBM", 59), ("Calibrate +\nevaluate\nbootstrap CIs", 78)]
for txt, x in boxes:
    ax.add_patch(FancyBboxPatch((x, 15), 17, 12, boxstyle="round,pad=0.4,rounding_size=1.2", fc=LIGHT, ec=BLUE, lw=1.6))
    ax.text(x + 8.5, 21, txt, ha="center", va="center", color=NAVY, fontsize=8.5)
for x in (19.4, 38.4, 57.4, 76.4):
    ax.add_patch(FancyArrowPatch((x, 21), (x + 1.9, 21), arrowstyle="-|>", mutation_scale=12, color=AMBER, lw=2))
outs = [("model_bundle.joblib\nmodel + calibrator", 14), ("FastAPI service\n/score, /worklist, /trust", 42), ("Web UI + Docker", 70)]
for txt, x in outs:
    ax.add_patch(FancyBboxPatch((x, 1.5), 24, 8, boxstyle="round,pad=0.4,rounding_size=1.2", fc="#DDE8DE", ec=GREEN, lw=1.6))
    ax.text(x + 12, 5.5, txt, ha="center", va="center", color=NAVY, fontsize=8.5)
ax.add_patch(FancyArrowPatch((86, 14.6), (30, 9.9), arrowstyle="-|>", mutation_scale=12, color=GREY, lw=1.6, connectionstyle="arc3,rad=0.15"))
ax.add_patch(FancyArrowPatch((38.4, 5.5), (41.6, 5.5), arrowstyle="-|>", mutation_scale=12, color=AMBER, lw=2))
ax.add_patch(FancyArrowPatch((66.4, 5.5), (69.6, 5.5), arrowstyle="-|>", mutation_scale=12, color=AMBER, lw=2))
save(fig, "fig_architecture.png")
print("figures written to", OUT)


# 8. customer segments (K-Means on behaviour, profiled on the test period)
seg_path = ROOT / "reports" / "segments.json"
if seg_path.exists():
    sg = json.loads(seg_path.read_text(encoding="utf8"))
    names = {x["id"]: x["name"] for x in sg["segments"]}
    palette = {x["id"]: c for x, c in zip(sg["segments"], ("#C0392B", "#D9A400", "#2E6FA8", "#2E8B57", "#7F8C8D", "#6C3483", "#16A085"))}   # distinct hues, independent of the theme palette
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.9), gridspec_kw={"width_ratios": [1.1, 1]})
    pc = sg["pca_sample"]
    for sid, nm in names.items():
        pts = [(x, y) for x, y, g in zip(pc["x"], pc["y"], pc["segment"]) if g == sid]
        a1.scatter([p[0] for p in pts], [p[1] for p in pts], s=7, alpha=0.55, color=palette[sid], label=nm)
    xs, ys = np.array(pc["x"]), np.array(pc["y"])
    a1.set_xlim(np.percentile(xs, 0.5) - 0.5, np.percentile(xs, 99.5) + 0.5)
    a1.set_ylim(np.percentile(ys, 0.5) - 0.5, np.percentile(ys, 99.5) + 0.5)
    a1.set_xlabel(f"PCA component 1 ({sg['pca_explained_variance'][0]:.0%} of variance)")
    a1.set_ylabel(f"PCA component 2 ({sg['pca_explained_variance'][1]:.0%})")
    a1.legend(frameon=False, fontsize=8.5, markerscale=2)
    a1.set_title(f"K-Means, k = {sg['k']} (chosen by silhouette)", color=NAVY, fontsize=10)
    rows = sorted(sg["segments"], key=lambda r: r["lapse_rate"])
    ypos = range(len(rows))
    a2.barh([i + 0.2 for i in ypos], [r["lapse_rate"] for r in rows], height=0.38, color=NAVY, label="Actually went quiet")
    a2.barh([i - 0.2 for i in ypos], [r["mean_predicted"] for r in rows], height=0.38, color=AMBER, label="Model predicted")
    a2.set_yticks(list(ypos))
    a2.set_yticklabels([f"{r['name']} ({r['share']:.0%})" for r in rows], fontsize=8.5)
    for i, r in zip(ypos, rows):
        a2.text(max(r["lapse_rate"], r["mean_predicted"]) + 0.02, i, "ROC-AUC " + ("n/a" if r["model_roc_auc"] is None else f"{r['model_roc_auc']:.2f}"), va="center", fontsize=8, color=GREY)
    a2.set_xlim(0, 1.15)
    a2.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    a2.legend(frameon=False, fontsize=8.5, loc="lower right")
    a2.set_title("Lapse rate and model quality by segment (test period)", color=NAVY, fontsize=10)
    save(fig, "fig_clusters.png")
    print("clusters figure written")
