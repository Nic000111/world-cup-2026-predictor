"""
glicko.py — full Glicko rating engine (rating + uncertainty) with the confederation offset,
tuned by held-out log-loss, with forward feature selection, head-to-head vs the shipped Elo.

Engine = Glicko (adaptive step via rating-deviation) + a per-confederation offset that updates
on cross-continental games (orthogonal cross-continental level fix, carried over from Elo).
Then we forward-select among {uncertainty, momentum, rest} — keep a feature only if it lowers
the 2022-23 validation log-loss. 2024+ is the final test. Everything leak-free (pre-match).
"""

# --- run from anywhere ---
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)
# -------------------------

import math, warnings
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import confed, elo

warnings.filterwarnings("ignore", category=ConvergenceWarning)
CLASSES = ["home_win", "draw", "away_win"]
Q = math.log(10) / 400.0
PI2 = math.pi * math.pi
MW = 10            # momentum window (appearances), fixed like Elo
RESTCAP = 14.0


def ll(p, y):
    idx = np.array([CLASSES.index(t) for t in y])
    return -np.mean(np.log(np.clip(p[np.arange(len(y)), idx], 1e-15, 1)))


def gfac(RD):
    return 1.0 / math.sqrt(1.0 + 3.0 * Q * Q * RD * RD / PI2)


# ---------- precompute ----------
RAW = pd.read_csv("results.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
RAW["neutral"] = RAW.neutral.astype(str).str.upper().eq("TRUE")
teams = sorted(set(RAW.home_team) | set(RAW.away_team)); TI = {t: i for i, t in enumerate(teams)}; N = len(teams)
H = RAW.home_team.map(TI).to_numpy(); A = RAW.away_team.map(TI).to_numpy()
DAY = (RAW.date - RAW.date.min()).dt.days.to_numpy().astype(float)
NEU = RAW.neutral.to_numpy()
hs = RAW.home_score.to_numpy(); as_ = RAW.away_score.to_numpy()
PLAYED = ~(np.isnan(hs) | np.isnan(as_))
SH = np.where(hs > as_, 1.0, np.where(hs < as_, 0.0, 0.5))
IMP = np.array([elo.importance_weight(t) for t in RAW.tournament])
MOV = np.array([elo.mov_multiplier(int(a - b)) if pl else 1.0 for a, b, pl in zip(hs, as_, PLAYED)])
YEAR = RAW.date.dt.year.to_numpy()
RES = np.where(hs > as_, "home_win", np.where(hs < as_, "away_win", "draw"))
HC = np.array([confed.confed_of(t) for t in RAW.home_team]); AC = np.array([confed.confed_of(t) for t in RAW.away_team])
TR = PLAYED & (YEAR >= 2010) & (YEAR < 2022)
VA = PLAYED & (YEAR >= 2022) & (YEAR < 2024)
TE = PLAYED & (YEAR >= 2024)


def run_glicko(P):
    HA, RD0, rdmin, c, usemov, kc = P["HA"], P["RD0"], P["rdmin"], P["c"], P["usemov"], P["kc"]
    r = np.full(N, 1500.0); RD = np.full(N, float(RD0)); last = np.full(N, -1.0)
    conf = defaultdict(float); hist = [[] for _ in range(N)]
    g = {k: np.empty(len(RAW)) for k in ["gap", "rdh", "rda", "momh", "moma", "resth", "resta"]}
    for k in range(len(RAW)):
        h, a, d = H[k], A[k], DAY[k]
        if last[h] >= 0: RD[h] = min(math.sqrt(RD[h] * RD[h] + c * c * (d - last[h])), RD0)
        if last[a] >= 0: RD[a] = min(math.sqrt(RD[a] * RD[a] + c * c * (d - last[a])), RD0)
        ch, ca = HC[k], AC[k]
        oh = conf[ch] if ch != "OTHER" else 0.0
        oa = conf[ca] if ca != "OTHER" else 0.0
        eff_h, eff_a = r[h] + oh, r[a] + oa
        ha = 0.0 if NEU[k] else HA
        g["gap"][k] = eff_h - eff_a; g["rdh"][k] = RD[h]; g["rda"][k] = RD[a]
        g["resth"][k] = min(d - last[h], RESTCAP) if last[h] >= 0 else RESTCAP
        g["resta"][k] = min(d - last[a], RESTCAP) if last[a] >= 0 else RESTCAP
        hh, aa = hist[h], hist[a]
        g["momh"][k] = eff_h - hh[-MW] if len(hh) >= MW else 0.0
        g["moma"][k] = eff_a - aa[-MW] if len(aa) >= MW else 0.0
        hh.append(eff_h); aa.append(eff_a)
        if not PLAYED[k]:
            continue
        w = IMP[k] * (MOV[k] if usemov else 1.0); s = SH[k]
        rh, RDh, ra, RDa = r[h], RD[h], r[a], RD[a]
        go = gfac(RDa); Es = 1.0 / (1 + 10 ** (-go * ((eff_h + ha) - eff_a) / 400.0))
        info = w * Q * Q * go * go * Es * (1 - Es); RD2 = 1.0 / (1.0 / (RDh * RDh) + info)
        r[h] = rh + Q * RD2 * w * go * (s - Es); RD[h] = max(math.sqrt(RD2), rdmin)
        go = gfac(RDh); Ea = 1.0 / (1 + 10 ** (-go * (eff_a - (eff_h + ha)) / 400.0))
        info = w * Q * Q * go * go * Ea * (1 - Ea); RD2 = 1.0 / (1.0 / (RDa * RDa) + info)
        r[a] = ra + Q * RD2 * w * go * ((1 - s) - Ea); RD[a] = max(math.sqrt(RD2), rdmin)
        if kc and ch != ca and ch != "OTHER" and ca != "OTHER":
            dc = kc * w * (s - Es); conf[ch] += dc; conf[ca] -= dc
        last[h] = d; last[a] = d
    return g


def featmat(g, names):
    col = {"gap": g["gap"], "abs": np.abs(g["gap"]), "flag": (~NEU).astype(float),
           "rd": g["rdh"] + g["rda"], "mom": g["momh"] - g["moma"], "rest": g["resth"] - g["resta"]}
    return np.column_stack([col[n] for n in names])


def score(g, names, C=0.05, test=False):
    X = featmat(g, names)
    m = Pipeline([("s", StandardScaler()), ("c", LogisticRegression(C=C, max_iter=2000))]).fit(X[TR], RES[TR])
    o = [list(m.classes_).index(c) for c in CLASSES]
    va = ll(m.predict_proba(X[VA])[:, o], RES[VA])
    return (va, ll(m.predict_proba(X[TE])[:, o], RES[TE])) if test else va


# ---------- 1) tune the engine on base features ----------
GRID = {"HA": [40, 70, 100], "RD0": [180, 250, 320, 400], "rdmin": [20, 35, 50],
        "c": [1.0, 2.0, 4.0, 8.0], "usemov": [True, False], "kc": [0.0, 0.5, 1.0, 1.5, 2.0]}
P = {"HA": 70, "RD0": 250, "rdmin": 35, "c": 2.0, "usemov": True, "kc": 1.0}
best = score(run_glicko(P), ["gap", "abs", "flag"])
for _ in range(2):
    for key, vals in GRID.items():
        cur = P[key]; bv, bval = cur, best
        for v in vals:
            if v == cur: continue
            P[key] = v; s = score(run_glicko(P), ["gap", "abs", "flag"])
            if s < bval: bv, bval = v, s
        P[key] = bv; best = bval
print(f"tuned engine: {P}   base val_ll={best:.4f}")

# ---------- 2) forward feature selection ----------
g = run_glicko(P)
chosen = ["gap", "abs", "flag"]; cur = score(g, chosen)
print(f"\nforward selection (start val_ll={cur:.4f}):")
for _ in range(3):
    cand = [f for f in ["rd", "mom", "rest"] if f not in chosen]
    if not cand: break
    scores = {f: score(g, chosen + [f]) for f in cand}
    bf = min(scores, key=scores.get)
    if scores[bf] < cur - 1e-5:
        chosen.append(bf); cur = scores[bf]
        print(f"   + {bf:5} -> val_ll={cur:.4f}   KEEP")
    else:
        for f in cand: print(f"   + {f:5} -> val_ll={scores[f]:.4f}   (no gain, drop)")
        break
print(f"final features: {chosen}")

# ---------- 3) head-to-head vs shipped Elo (same harness) ----------
def best_C(scorer):
    return min(((C,) + scorer(C) for C in [0.01, 0.03, 0.05, 0.1]), key=lambda t: t[1])
gl = best_C(lambda C: score(g, chosen, C=C, test=True))
# shipped Elo: confed-on, V1 features (rating_gap, abs, flag, mom, rest)
ef = elo.compute_elo(RAW, dict(elo.DEFAULT_PARAMS))[0]
ge = {"gap": (ef.home_elo - ef.away_elo).to_numpy(), "rdh": np.zeros(len(RAW)), "rda": np.zeros(len(RAW)),
      "momh": ef.home_mom.to_numpy(), "moma": ef.away_mom.to_numpy(),
      "resth": ef.home_rest.to_numpy(), "resta": ef.away_rest.to_numpy()}
el = best_C(lambda C: score(ge, ["gap", "abs", "flag", "mom", "rest"], C=C, test=True))

print("\n================  FINAL: full Glicko vs shipped Elo  ================")
print(f"  {'model':<34}{'best C':>7}{'VAL':>9}{'TEST':>9}")
print(f"  {'Elo + confed + V1 feats (shipped)':<34}{el[0]:>7}{el[1]:>9.4f}{el[2]:>9.4f}")
print(f"  {'Glicko + confed + ' + '+'.join(chosen[3:]):<34}{gl[0]:>7}{gl[1]:>9.4f}{gl[2]:>9.4f}")
print(f"  {'improvement':<34}{'':>7}{gl[1]-el[1]:>+9.4f}{gl[2]-el[2]:>+9.4f}")
