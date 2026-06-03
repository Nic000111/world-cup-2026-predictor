"""
tune_confed.py — tune the confederation-Elo learning rate k_confed by HELD-OUT log-loss.

Two-level Elo (elo.py confed_elo): a per-confederation offset, moved only by cross-confed
games, shared by every team in the confederation, added straight into each team's rating.
k_confed sets how hard those games push it. k_confed = 0 is the current pure-team Elo.

The offset can only ever help the ~14% of games that cross confederations, so we score
OVERALL and CROSS-CONFED log-loss separately, on the production feature set.

Protocol (no test peeking):
    TRAIN  < 2022       fit the V1 logistic (once per k_confed)
    VAL    2022-2023    sweep k_confed here, pick the best
    TEST   2024+        ONE confirmation of the chosen value vs k_confed = 0
Leak-free: every recorded rating is pre-match; offsets accumulate causally.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import confed
import elo

warnings.filterwarnings("ignore", category=ConvergenceWarning)
CLASSES = ["home_win", "draw", "away_win"]
F = elo.V1_FEATURES
RAW = pd.read_csv("results.csv", parse_dates=["date"])
CONFS = ["UEFA", "CONMEBOL", "CAF", "AFC", "CONCACAF", "OFC"]


def ll(p, y):
    idx = np.array([CLASSES.index(t) for t in y])
    return -np.mean(np.log(np.clip(p[np.arange(len(y)), idx], 1e-15, 1)))


def build(kc):
    p = dict(elo.DEFAULT_PARAMS)
    p["confed_elo"], p["k_confed"] = (kc > 0), kc
    feat, final = elo.build_features(RAW, p)
    d = feat[feat.result.notna() & (feat.date >= "2010-01-01")].copy().reset_index(drop=True)
    d["hc"] = d.home_team.map(confed.confed_of)
    d["ac"] = d.away_team.map(confed.confed_of)
    return d, feat.attrs.get("confed_offsets", {})


def evaluate(d):
    yr = d.date.dt.year.values
    cross = ((d.hc != d.ac) & (d.hc != "OTHER") & (d.ac != "OTHER")).values
    tr, va, te = yr < 2022, (yr >= 2022) & (yr < 2024), yr >= 2024
    y = d.result.values
    m = Pipeline([("sc", StandardScaler()),
                  ("clf", LogisticRegression(C=0.01, max_iter=2000))]).fit(d.loc[tr, F], y[tr])
    order = [list(m.classes_).index(c) for c in CLASSES]

    def sc(mask):
        p = m.predict_proba(d.loc[mask, F])[:, order]
        yy = y[mask]
        xm = cross[mask]
        return ll(p, yy), ll(p[xm], yy[xm]), int(xm.sum())
    return sc(va), sc(te)


grid = [0, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0]
print("                  VALIDATION 22-23        TEST 24+ (confirm)")
print(f"{'k_conf':>6} | {'overall':>8} {'x-confed':>8} | {'overall':>8} {'x-confed':>8} | offsets (Elo)")
print("-" * 92)
rows = {}
for kc in grid:
    d, off = build(kc)
    (vo, vx, vn), (to, tx, tn) = evaluate(d)
    rows[kc] = (vo, vx, to, tx, off)
    os = " ".join(f"{c[:4]}{off.get(c,0):+.0f}" for c in CONFS) if off else "(pure team Elo)"
    tag = "  <- baseline" if kc == 0 else ""
    print(f"{kc:>6.2f} | {vo:>8.4f} {vx:>8.4f} | {to:>8.4f} {tx:>8.4f} | {os}{tag}")

b0 = rows[0]
print(f"\nbaseline x-confed:  val {b0[1]:.4f}  test {b0[3]:.4f}")
print("Pick the smallest k that captures most of the cross-confed gain with SANE offsets —")
print("not the raw minimum (which overfits tiny confeds like OFC).")
