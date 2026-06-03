"""
group_predictions.py - 2026 World Cup group-stage forecasts (the 72 known-matchup games).

Both models, side by side:
  - Elo-logistic  -> 1X2 probabilities (home / draw / away)
  - Goals model (Elo-Poisson + Dixon-Coles) -> 1X2 + expected goals + most-likely scoreline
Writes a readable .txt (table + legend) AND a .csv to ~/Downloads.
"""
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import elo

CLASSES = ["home_win", "draw", "away_win"]; MAXG = 10
FEATS = ["scorer_elo", "opp_elo", "is_home", "is_friendly"]

feat, _ = elo.build_features(pd.read_csv("results.csv", parse_dates=["date"]))
pl = feat[feat.result.notna() & (feat.date >= "2010-01-01")].copy()
pl["is_friendly"] = pl.tournament.str.contains("friendly", case=False).astype(float)

# ---- model 1: Elo-logistic (1X2) ----
F = elo.V1_FEATURES
LR = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(C=0.01, max_iter=2000))]).fit(pl[F], pl["result"])
def lr_probs(rows):
    return LR.predict_proba(rows[F])[:, [list(LR.classes_).index(c) for c in CLASSES]]

# ---- model 2: goals model (1X2 + scorelines) ----
def long(r):
    h = pd.DataFrame(dict(goals=r.home_score.values, scorer_elo=r.home_elo.values, opp_elo=r.away_elo.values, is_home=np.where(r.neutral, 0., 1.), is_friendly=r.is_friendly.values))
    a = pd.DataFrame(dict(goals=r.away_score.values, scorer_elo=r.away_elo.values, opp_elo=r.home_elo.values, is_home=0., is_friendly=r.is_friendly.values))
    return pd.concat([h, a], ignore_index=True)
GM = Pipeline([("sc", StandardScaler()), ("po", PoissonRegressor(alpha=1e-4, max_iter=5000))]).fit(long(pl)[FEATS], long(pl)["goals"])
lh = GM.predict(pd.DataFrame(dict(scorer_elo=pl.home_elo, opp_elo=pl.away_elo, is_home=np.where(pl.neutral, 0., 1.), is_friendly=pl.is_friendly))[FEATS])
la = GM.predict(pd.DataFrame(dict(scorer_elo=pl.away_elo, opp_elo=pl.home_elo, is_home=0., is_friendly=pl.is_friendly))[FEATS])
xt, yt = pl.home_score.values.astype(int), pl.away_score.values.astype(int)
def tau(x, y, lh, la, r):
    t = np.ones(len(x)); m = (x == 0) & (y == 0); t[m] = 1 - lh[m] * la[m] * r; m = (x == 0) & (y == 1); t[m] = 1 + lh[m] * r
    m = (x == 1) & (y == 0); t[m] = 1 + la[m] * r; m = (x == 1) & (y == 1); t[m] = 1 - r; return t
RHO = float(minimize_scalar(lambda r: -np.sum(np.log(np.clip(tau(xt, yt, lh, la, r), 1e-9, None))), bounds=(-0.2, 0.2), method="bounded").x)

def goal_pred(li, lj):
    """(1X2 probs, single most-likely scoreline) from the Dixon-Coles-corrected matrix."""
    g = np.arange(MAXG + 1); M = np.outer(poisson.pmf(g, li), poisson.pmf(g, lj))
    M[0, 0] *= 1 - li * lj * RHO; M[0, 1] *= 1 + li * RHO; M[1, 0] *= 1 + lj * RHO; M[1, 1] *= 1 - RHO
    M = np.clip(M, 0, None); M /= M.sum()
    h, d, a = np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()
    si, sj = np.unravel_index(M.argmax(), M.shape)
    return np.array([h, d, a]), f"{si}-{sj}"

# ---- 72 group fixtures ----
wc = feat[(feat.tournament == "FIFA World Cup") & (feat.date.dt.year == 2026) & feat.result.isna()].copy().reset_index(drop=True)
LH = GM.predict(pd.DataFrame(dict(scorer_elo=wc.home_elo, opp_elo=wc.away_elo, is_home=np.where(wc.neutral, 0., 1.), is_friendly=0.))[FEATS])
LA = GM.predict(pd.DataFrame(dict(scorer_elo=wc.away_elo, opp_elo=wc.home_elo, is_home=0., is_friendly=0.))[FEATS])
LRP = lr_probs(wc)

# ---- group labels (anchor teams) ----
adj = defaultdict(set)
for _, r in wc.iterrows():
    adj[r.home_team].add(r.away_team); adj[r.away_team].add(r.home_team)
ANCH = {"Mexico": "A", "Canada": "B", "Brazil": "C", "United States": "D", "Germany": "E", "Netherlands": "F",
        "Belgium": "G", "Spain": "H", "France": "I", "Argentina": "J", "Portugal": "K", "England": "L"}
letter = {}
for t in set(wc.home_team) | set(wc.away_team):
    grp = {t} | adj[t]; L = next((ANCH[x] for x in grp if x in ANCH), None)
    if L:
        for x in grp: letter[x] = L

rows = []
for i, r in wc.iterrows():
    gp, score = goal_pred(LH[i], LA[i])
    rows.append(dict(group=letter[r.home_team], date=r.date.strftime("%m-%d"), home=r.home_team, away=r.away_team,
                     log_H=LRP[i, 0], log_D=LRP[i, 1], log_A=LRP[i, 2], gl_H=gp[0], gl_D=gp[1], gl_A=gp[2],
                     xg_home=LH[i], xg_away=LA[i], score=score))
out = pd.DataFrame(rows).sort_values(["group", "date"]).reset_index(drop=True)

# ---- build readable table text ----
L_ = []
L_.append("2026 WORLD CUP - GROUP-STAGE FORECASTS  (both models)")
for L in sorted(out.group.unique()):
    L_.append(f"\n=== GROUP {L} {'=' * 56}")
    L_.append(f"  {'date':5} {'home':17}{'away':17}| logistic  |  goals    |   xG    | score")
    for _, r in out[out.group == L].iterrows():
        lg = f"{r.log_H * 100:2.0f}/{r.log_D * 100:2.0f}/{r.log_A * 100:2.0f}"
        gl = f"{r.gl_H * 100:2.0f}/{r.gl_D * 100:2.0f}/{r.gl_A * 100:2.0f}"
        L_.append(f"  {r.date} {r.home[:16]:17}{r.away[:16]:17}| {lg:9} | {gl:9} | {r.xg_home:.1f}-{r.xg_away:.1f} | {r.score}")
L_ += ["", "=" * 72, "LEGEND", "=" * 72,
       "  Two models side by side - both give  Home / Draw / Away  win probabilities:",
       "    logistic : Elo-logistic (logistic regression on the Elo features) - our best 1X2 model",
       "    goals    : Elo-Poisson + Dixon-Coles - models goals, so it also produces scorelines",
       "  reading '80/15/ 6'  =  80% home win  /  15% draw  /  6% away win",
       "  xG     : expected goals for each side (home-away) - the Poisson means behind the score",
       "  score  : single most-likely exact scoreline (low/clean by nature - the full goal",
       "           spread is wider; e.g. a 2-0 favourite also has real 3-0 / 1-0 chances)",
       "  home / away = the fixture's listed sides. At neutral venues neither gets a home boost;",
       "                only the 3 hosts (Mexico/Canada/USA) count as 'home', in their own games."]
text = "\n".join(L_)
print(text)

# ---- save both files to ~/Downloads ----
txt = os.path.expanduser("~/Downloads/wc2026_group_predictions.txt")
csv = os.path.expanduser("~/Downloads/wc2026_group_predictions.csv")
with open(txt, "w") as f:
    f.write(text + "\n")
out.round(3).to_csv(csv, index=False)
print(f"\nSaved readable table + legend -> {txt}")
print(f"Saved CSV (data only)         -> {csv}")
