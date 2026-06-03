"""
Validate the cross-confederation K-boost with forward-chaining (rolling-origin) CV.
For each boost we recompute the Elo, then measure across 5 time-folds:
  - val_LL      : overall log-loss (does the boost HURT the bulk of games?)
  - xconf_LL    : cross-confed log-loss (does it HELP the games it touches?)
  - |resid|     : mean |cross-confed Elo residual| (does the compression shrink toward 0?)
  - EUR-SAM     : Europe-minus-South-America Elo gap among contenders (does Europe rise toward the market?)
"""

# --- run from anywhere: make project root importable + cwd ---
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)
# -------------------------------------------------------------

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
EUR = ["Spain", "France", "England", "Germany", "Portugal", "Netherlands"]
SAM = ["Argentina", "Brazil", "Colombia", "Ecuador", "Uruguay"]


def build(boost):
    params = dict(elo.DEFAULT_PARAMS); params["cross_confed_boost"] = boost
    feat, ratings = elo.build_features(RAW, params)
    d = feat[feat.result.notna() & (feat.date >= "2010-01-01")].copy().reset_index(drop=True)
    d["hc"] = d.home_team.map(confed.confed_of); d["ac"] = d.away_team.map(confed.confed_of)
    return d, ratings


def ll(p, y):
    pos = [CLASSES.index(t) for t in y]
    return -np.mean(np.log(np.clip(p[np.arange(len(y)), pos], 1e-15, 1)))


def cv(d):
    yr = d.date.dt.year.values
    cross = ((d.hc != d.ac) & (d.hc != "OTHER") & (d.ac != "OTHER")).values
    ov, xc = [], []
    for Y in [2019, 2020, 2021, 2022, 2023]:
        tr = yr < Y; va = yr == Y
        m = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(C=0.01, max_iter=2000))]).fit(d.loc[tr, F], d.loc[tr, "result"])
        p = m.predict_proba(d.loc[va, F])[:, [list(m.classes_).index(c) for c in CLASSES]]
        y = d.loc[va, "result"].values
        ov.append(ll(p, y))
        xm = cross[va]
        xc.append(ll(p[xm], y[xm]))
    return np.array(ov), np.array(xc)


def resid_absmean(d):
    ha = np.where(d.neutral, 0., 70.)
    exp = 1 / (1 + 10 ** (-((d.home_elo + ha) - d.away_elo) / 400))
    act = np.where(d.home_score > d.away_score, 1., np.where(d.home_score < d.away_score, 0., .5))
    c = d[(d.hc != d.ac) & (d.hc != "OTHER") & (d.ac != "OTHER")].copy()
    c["r"] = (act - exp)[(d.hc != d.ac) & (d.hc != "OTHER") & (d.ac != "OTHER")]
    h = pd.DataFrame({"c": c.hc, "r": c.r}); a = pd.DataFrame({"c": c.ac, "r": -c.r})
    return pd.concat([h, a]).groupby("c").r.mean().abs().mean()


print(f"{'boost':>6} | {'val_LL (±)':>16} | {'xconf_LL (±)':>16} | {'|resid|':>8} | {'EUR-SAM':>8}")
print("-" * 66)
for boost in [1.0, 1.5, 2.0, 2.5, 3.0]:
    d, ratings = build(boost)
    ov, xc = cv(d)
    mr = resid_absmean(d)
    gap = np.mean([ratings[t] for t in EUR]) - np.mean([ratings[t] for t in SAM])
    print(f"{boost:>6.1f} | {ov.mean():.4f} ±{ov.std():.4f} | {xc.mean():.4f} ±{xc.std():.4f} | {mr:>8.3f} | {gap:>+8.0f}")
