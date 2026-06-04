"""
calibration.py — is the W/D/L logistic model calibrated? (i.e. when it says 60%,
do the home teams actually win ~60% of the time?)

Trains the production model on <2024, scores the held-out 2024+ games, and renders a
reliability diagram + a "where the predictions land" histogram to docs/calibration.png.
Also prints the Expected Calibration Error (ECE) per class.

Logistic regression minimises log-loss (a proper scoring rule), so its outputs come out
calibrated by construction — this script is the evidence, not a fix. Re-run after any
methodology change to refresh the committed plot.
"""

# --- run from anywhere: make project root importable + cwd ---
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)
# -------------------------------------------------------------

import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import elo

warnings.filterwarnings("ignore", category=ConvergenceWarning)

C = ["home_win", "draw", "away_win"]
LAB = {"home_win": "Home win", "draw": "Draw", "away_win": "Away win"}
COL = {"home_win": "#1f6feb", "draw": "#d98a00", "away_win": "#d1495b"}
F = elo.V1_FEATURES

# ---- train on <2024, evaluate on held-out 2024+ ----
feat, _ = elo.build_features(pd.read_csv("results.csv", parse_dates=["date"]))
d = feat[feat.result.notna() & (feat.date >= "2010-01-01")].copy().reset_index(drop=True)
yr = d.date.dt.year.values
tr, te = yr < 2024, yr >= 2024
m = Pipeline([("sc", StandardScaler()),
              ("clf", LogisticRegression(C=0.01, max_iter=2000))]).fit(d.loc[tr, F], d.result.values[tr])
order = [list(m.classes_).index(c) for c in C]
P = m.predict_proba(d.loc[te, F])[:, order]
y = d.result.values[te]
ll = -np.mean(np.log(np.clip(P[np.arange(len(y)), [C.index(t) for t in y]], 1e-15, 1)))

bins = np.linspace(0, 1, 11)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1.35, 1]})

# ---- reliability diagram ----
ax1.plot([0, 1], [0, 1], "--", color="#888", lw=1.4, label="perfect calibration", zorder=1)
ax1.fill_between([0, 1], [0, 1], [0, 0], color="#1f6feb", alpha=0.025)
eces = {}
for cls in C:
    pred, actual = P[:, C.index(cls)], (y == cls).astype(float)
    xs, ys, ns, ece = [], [], [], 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (pred >= lo) & (pred < hi)
        if mask.sum() < 8:
            continue
        pp, aa = pred[mask].mean(), actual[mask].mean()
        xs.append(pp); ys.append(aa); ns.append(mask.sum())
        ece += (mask.sum() / len(pred)) * abs(aa - pp)
    eces[cls] = ece
    ns = np.array(ns)
    ax1.plot(xs, ys, "-", color=COL[cls], lw=1.6, alpha=0.85, zorder=2)
    ax1.scatter(xs, ys, s=20 + ns / ns.max() * 230, color=COL[cls], alpha=0.85,
                edgecolor="white", lw=1.1, zorder=3, label=f"{LAB[cls]}  (ECE {ece:.3f})")
ax1.set_xlim(0, 0.85); ax1.set_ylim(0, 0.85)
ax1.set_xlabel("Predicted probability", fontsize=11)
ax1.set_ylabel("Observed frequency (held-out 2024+)", fontsize=11)
ax1.set_title("Reliability diagram — W/D/L logistic model", fontsize=12.5, weight="bold")
ax1.legend(loc="upper left", fontsize=9.5, framealpha=0.9)
ax1.grid(alpha=0.18)
ax1.text(0.97, 0.04, "points on the line = calibrated\nabove = model under-confident\nbelow = over-confident\n(marker size ∝ # games)",
         transform=ax1.transAxes, ha="right", va="bottom", fontsize=8.2, color="#555",
         bbox=dict(boxstyle="round", fc="#f6f8fa", ec="#ddd"))

# ---- where the predictions land ----
for cls in C:
    ax2.hist(P[:, C.index(cls)], bins=np.linspace(0, 0.85, 26), color=COL[cls], alpha=0.5, label=LAB[cls])
ax2.set_xlabel("Predicted probability", fontsize=11)
ax2.set_ylabel("# of test games", fontsize=11)
ax2.set_title("Where the predictions land", fontsize=12.5, weight="bold")
ax2.legend(fontsize=9.5); ax2.grid(alpha=0.18)
ax2.text(0.97, 0.97, f"draws never exceed ~{P[:,1].max():.0%}\n— why the model rarely\n*picks* a draw",
         transform=ax2.transAxes, ha="right", va="top", fontsize=8.4, color="#555",
         bbox=dict(boxstyle="round", fc="#fff8e1", ec="#e0c060"))

avg_ece = float(np.mean(list(eces.values())))
fig.suptitle(f"World Cup 2026 model — calibration on {len(y)} held-out games (2024+)   ·   "
             f"log-loss {ll:.3f}   ·   avg ECE {avg_ece:.3f}", fontsize=11.5, y=1.005, color="#333")
fig.tight_layout()
os.makedirs("docs", exist_ok=True)
out = os.path.join("docs", "calibration.png")
fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")

print(f"saved: {out}")
print(f"\nlog-loss (test): {ll:.4f}")
for cls in C:
    print(f"  {LAB[cls]:<10} ECE = {eces[cls]:.3f}")
print(f"  {'AVERAGE':<10} ECE = {avg_ece:.3f}")
print("\nscale:  <0.02 well calibrated | 0.02-0.05 slightly off | >0.05 needs calibration")
