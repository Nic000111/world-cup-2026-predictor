"""
goals_model.py - V2 goals model: Elo-informed Poisson + Dixon-Coles correction.

Instead of ~250 noisy per-team attack/defense dummies, the scoring rate uses our ONLINE Elo
ratings as covariates -- so the model is always current (Elo updates every match) and robust:

    log E[goals_for] = b0 + b1*scorer_elo + b2*opponent_elo + b3*is_home + b4*is_friendly

with is_home neutral-aware (0 at neutral venues, matching elo.py). A Dixon-Coles rho then
corrects the low-score draw cells. Derives 1X2 + full scoreline matrix.

Development + selection on VALIDATION (2022-23). TEST (2024+) is NOT touched here.
"""

# --- run from anywhere: make project root importable + cwd ---
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)
# -------------------------------------------------------------

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import elo

CLASSES = ["home_win", "draw", "away_win"]
MAXG = 10
FEATS = ["scorer_elo", "opp_elo", "is_home", "is_friendly"]

feat, _ = elo.build_features(pd.read_csv("results.csv", parse_dates=["date"]))
d = feat[feat.result.notna() & (feat.date >= "2010-01-01")].copy().reset_index(drop=True)
d["is_friendly"] = d.tournament.str.contains("friendly", case=False).astype(float)
yr = d.date.dt.year
tr = (yr <= 2021).values
va = ((yr >= 2022) & (yr <= 2023)).values   # TEST (2024+) intentionally left frozen


def build_long(rows):
    """Two goal-observations per match; online Elo covariates; neutral-aware home term."""
    home = pd.DataFrame(dict(goals=rows.home_score.values, scorer_elo=rows.home_elo.values,
                             opp_elo=rows.away_elo.values,
                             is_home=np.where(rows.neutral, 0.0, 1.0),   # #1: no home boost at neutral
                             is_friendly=rows.is_friendly.values))
    away = pd.DataFrame(dict(goals=rows.away_score.values, scorer_elo=rows.away_elo.values,
                             opp_elo=rows.home_elo.values, is_home=0.0,
                             is_friendly=rows.is_friendly.values))
    return pd.concat([home, away], ignore_index=True)


gmodel = Pipeline([("sc", StandardScaler()), ("po", PoissonRegressor(alpha=1e-4, max_iter=5000))])
gmodel.fit(build_long(d[tr])[FEATS], build_long(d[tr])["goals"])
co = gmodel.named_steps["po"].coef_
print("Poisson GLM coefficients (on standardized features):")
for f, c in zip(FEATS, co):
    print(f"   {f:12} {c:+.3f}")


def lambdas(rows):
    hh = pd.DataFrame(dict(scorer_elo=rows.home_elo.values, opp_elo=rows.away_elo.values,
                           is_home=np.where(rows.neutral, 0.0, 1.0), is_friendly=rows.is_friendly.values))
    aa = pd.DataFrame(dict(scorer_elo=rows.away_elo.values, opp_elo=rows.home_elo.values,
                           is_home=0.0, is_friendly=rows.is_friendly.values))
    return gmodel.predict(hh[FEATS]), gmodel.predict(aa[FEATS])


# ---- estimate Dixon-Coles rho on TRAIN (two-stage: lambdas fixed, optimise the tau likelihood) ----
def tau(x, y, lh, la, rho):
    t = np.ones(len(x))
    m = (x == 0) & (y == 0); t[m] = 1 - lh[m] * la[m] * rho
    m = (x == 0) & (y == 1); t[m] = 1 + lh[m] * rho
    m = (x == 1) & (y == 0); t[m] = 1 + la[m] * rho
    m = (x == 1) & (y == 1); t[m] = 1 - rho
    return t


lh_tr, la_tr = lambdas(d[tr])
xt, yt = d[tr].home_score.values.astype(int), d[tr].away_score.values.astype(int)
rho_hat = minimize_scalar(lambda r: -np.sum(np.log(np.clip(tau(xt, yt, lh_tr, la_tr, r), 1e-9, None))),
                          bounds=(-0.2, 0.2), method="bounded").x
print(f"\nDixon-Coles rho (train MLE) = {rho_hat:+.4f}")


def match_probs(lh, la, rho):
    g = np.arange(MAXG + 1)
    M = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
    M[0, 0] *= 1 - lh * la * rho
    M[0, 1] *= 1 + lh * rho
    M[1, 0] *= 1 + la * rho
    M[1, 1] *= 1 - rho
    M = np.clip(M, 0, None)
    h, dr, a = np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()
    return np.array([h, dr, a]) / (h + dr + a)


def metrics(P, y):
    pos = np.array([CLASSES.index(t) for t in y])
    pt = np.clip(P[np.arange(len(y)), pos], 1e-15, 1)
    oh = np.zeros_like(P); oh[np.arange(len(y)), pos] = 1
    pred = np.array(CLASSES)[P.argmax(1)]
    return dict(logloss=float(-np.mean(np.log(pt))),
                rps=float(np.mean(((np.cumsum(P, 1) - np.cumsum(oh, 1)) ** 2).sum(1) / 2)),
                acc=float((pred == y).mean()),
                draw_recall=float((pred[y == "draw"] == "draw").mean()),
                mean_draw_p=float(P[:, 1].mean()))


# ---- evaluate on VALIDATION ----
rows = d[va]
yv = np.where(rows.home_score.values > rows.away_score.values, "home_win",
              np.where(rows.home_score.values < rows.away_score.values, "away_win", "draw"))
lh, la = lambdas(rows)
P_indep = np.vstack([match_probs(lh[i], la[i], 0.0) for i in range(len(rows))])
P_dc = np.vstack([match_probs(lh[i], la[i], rho_hat) for i in range(len(rows))])

# Elo-logistic reference on the same val slice
F = elo.V1_FEATURES
lr = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(C=0.01, max_iter=2000))]).fit(d.loc[tr, F], d.loc[tr, "result"])
pv = lr.predict_proba(d.loc[va, F])[:, [list(lr.classes_).index(c) for c in CLASSES]]
m_lr = metrics(pv, d.loc[va, "result"].values)

print("\n" + "=" * 78)
print("VALIDATION 2022-23  (test frozen):")
print(f"{'model':28} {'logloss':>8} {'rps':>7} {'acc':>7} {'draw_rec':>9} {'mean_draw_p':>12}")
for name, m in [("Elo-logistic", m_lr), ("Goals: independent Poisson", metrics(P_indep, yv)),
                ("Goals: + Dixon-Coles", metrics(P_dc, yv))]:
    print(f"{name:28} {m['logloss']:8.4f} {m['rps']:7.4f} {m['acc']:7.4f} {m['draw_recall']:9.3f} {m['mean_draw_p']:12.3f}")
print(f"\n(base draw rate on val = {(yv=='draw').mean():.3f})")

print("\n--- example val scorelines (Dixon-Coles) ---")
ex = rows.head(6).reset_index(drop=True)
for i in range(len(ex)):
    g = np.arange(MAXG + 1)
    M = np.outer(poisson.pmf(g, lh[i]), poisson.pmf(g, la[i]))
    si, sj = np.unravel_index(M.argmax(), M.shape)
    pr = match_probs(lh[i], la[i], rho_hat)
    print(f"  {ex.home_team[i][:15]:15} {lh[i]:.2f}-{la[i]:.2f} {ex.away_team[i][:15]:15} | "
          f"H/D/A {pr[0]:.2f}/{pr[1]:.2f}/{pr[2]:.2f} | likely {si}-{sj} | actual {int(ex.home_score[i])}-{int(ex.away_score[i])}")
