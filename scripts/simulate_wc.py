"""
simulate_wc.py - 2026 World Cup Monte-Carlo forecast.

Engine = our Elo-informed Poisson + Dixon-Coles goals model (fit on all played matches),
with each team's pre-tournament Elo. Group games use the real 72 fixtures + neutral/host flags;
knockouts treated as neutral. Group standings use pts -> GD -> goals (head-to-head simplified);
the 8 best third-placed teams are matched to their Round-of-32 slots respecting FIFA's allowed
sets. Knockout draws resolved by penalties split proportional to win probability.
"""

# --- run from anywhere: make project root importable + cwd ---
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)
# -------------------------------------------------------------

import itertools
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import elo

N_SIM = 20000
MAXG = 10
FEATS = ["scorer_elo", "opp_elo", "is_home", "is_friendly"]
rng = np.random.default_rng(0)

# ---------- 1. ratings + goals model (fit on ALL played >=2010) ----------
feat, _ = elo.build_features(pd.read_csv("results.csv", parse_dates=["date"]))
pl = feat[feat.result.notna() & (feat.date >= "2010-01-01")].copy()
pl["is_friendly"] = pl.tournament.str.contains("friendly", case=False).astype(float)

def long(rows):
    h = pd.DataFrame(dict(goals=rows.home_score.values, scorer_elo=rows.home_elo.values, opp_elo=rows.away_elo.values, is_home=np.where(rows.neutral, 0., 1.), is_friendly=rows.is_friendly.values))
    a = pd.DataFrame(dict(goals=rows.away_score.values, scorer_elo=rows.away_elo.values, opp_elo=rows.home_elo.values, is_home=0., is_friendly=rows.is_friendly.values))
    return pd.concat([h, a], ignore_index=True)

gm = Pipeline([("sc", StandardScaler()), ("po", PoissonRegressor(alpha=1e-4, max_iter=5000))]).fit(long(pl)[FEATS], long(pl)["goals"])
lh = gm.predict(pd.DataFrame(dict(scorer_elo=pl.home_elo, opp_elo=pl.away_elo, is_home=np.where(pl.neutral, 0., 1.), is_friendly=pl.is_friendly))[FEATS])
la = gm.predict(pd.DataFrame(dict(scorer_elo=pl.away_elo, opp_elo=pl.home_elo, is_home=0., is_friendly=pl.is_friendly))[FEATS])
xt, yt = pl.home_score.values.astype(int), pl.away_score.values.astype(int)
def tau(x, y, lh, la, r):
    t = np.ones(len(x)); m = (x == 0) & (y == 0); t[m] = 1 - lh[m] * la[m] * r
    m = (x == 0) & (y == 1); t[m] = 1 + lh[m] * r; m = (x == 1) & (y == 0); t[m] = 1 + la[m] * r
    m = (x == 1) & (y == 1); t[m] = 1 - r; return t
RHO = float(minimize_scalar(lambda r: -np.sum(np.log(np.clip(tau(xt, yt, lh, la, r), 1e-9, None))), bounds=(-0.2, 0.2), method="bounded").x)

# ---------- 2. WC teams, groups (inferred from fixtures), pre-tournament Elo ----------
wc = feat[(feat.tournament == "FIFA World Cup") & (feat.date.dt.year == 2026) & feat.result.isna()].copy()
teams = sorted(set(wc.home_team) | set(wc.away_team)); idx = {t: i for i, t in enumerate(teams)}
assert len(teams) == 48
elo_of = {}
adj = defaultdict(set)
for _, r in wc.iterrows():
    elo_of[r.home_team] = r.home_elo; elo_of[r.away_team] = r.away_elo
    adj[r.home_team].add(r.away_team); adj[r.away_team].add(r.home_team)
elos = np.array([elo_of[t] for t in teams])

ANCH = {"Mexico": "A", "Canada": "B", "Brazil": "C", "United States": "D", "Germany": "E", "Netherlands": "F",
        "Belgium": "G", "Spain": "H", "France": "I", "Argentina": "J", "Portugal": "K", "England": "L"}
grp_teams, letter_seen = {}, set()
for t in teams:
    if t in letter_seen: continue
    g = {t} | adj[t]; assert len(g) == 4, (t, g)
    L = next(ANCH[x] for x in g if x in ANCH)
    grp_teams[L] = [idx[x] for x in g]; letter_seen |= g
LETTERS = sorted(grp_teams); assert len(LETTERS) == 12
print("Inferred groups:")
for L in LETTERS:
    print(f"  {L}: " + ", ".join(sorted(teams[i] for i in grp_teams[L])))

# ---------- 3. group-fixture lambdas + pre-drawn scorelines ----------
wc_lh = gm.predict(pd.DataFrame(dict(scorer_elo=wc.home_elo, opp_elo=wc.away_elo, is_home=np.where(wc.neutral, 0., 1.), is_friendly=0.))[FEATS])
wc_la = gm.predict(pd.DataFrame(dict(scorer_elo=wc.away_elo, opp_elo=wc.home_elo, is_home=0., is_friendly=0.))[FEATS])
fixtures = [(idx[r.home_team], idx[r.away_team]) for _, r in wc.iterrows()]
HG = [rng.poisson(wc_lh[fi], N_SIM).tolist() for fi in range(len(fixtures))]
AG = [rng.poisson(wc_la[fi], N_SIM).tolist() for fi in range(len(fixtures))]

def probs1x2(li, lj):
    g = np.arange(MAXG + 1); M = np.outer(poisson.pmf(g, li), poisson.pmf(g, lj))
    M[0, 0] *= 1 - li * lj * RHO; M[0, 1] *= 1 + li * RHO; M[1, 0] *= 1 + lj * RHO; M[1, 1] *= 1 - RHO
    M = np.clip(M, 0, None); h, d, a = np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum(); s = h + d + a
    return h / s, d / s, a / s, M

print("\nExample group-stage forecasts (1X2, most likely score):")
for fi in range(6):
    h, d, a, M = probs1x2(wc_lh[fi], wc_la[fi]); si, sj = np.unravel_index(M.argmax(), M.shape)
    print(f"  {teams[fixtures[fi][0]][:15]:15} vs {teams[fixtures[fi][1]][:15]:15}  {h:.0%}/{d:.0%}/{a:.0%}  likely {si}-{sj}")

# ---------- 4. neutral knockout advance-probability matrix (48x48) ----------
ii, jj = np.meshgrid(np.arange(48), np.arange(48), indexing="ij")
LAM = gm.predict(pd.DataFrame(dict(scorer_elo=elos[ii].ravel(), opp_elo=elos[jj].ravel(), is_home=0., is_friendly=0.))[FEATS]).reshape(48, 48)
ADV = np.full((48, 48), 0.5)
for i in range(48):
    for j in range(48):
        if i == j: continue
        h, d, a, _ = probs1x2(LAM[i, j], LAM[j, i])
        ADV[i, j] = h + d * h / (h + a) if (h + a) > 0 else 0.5

# ---------- 5. third-place slot matching for all 495 combos ----------
SLOTS = [("M74", set("ABCDF")), ("M77", set("CDFGH")), ("M79", set("CEFHI")), ("M80", set("EHIJK")),
         ("M81", set("BEFIJ")), ("M82", set("AEHIJ")), ("M85", set("EFGIJ")), ("M87", set("DEIJL"))]
def match_thirds(qual):
    assign, used = {}, set()
    def bt(i):
        if i == 8: return True
        sid, allowed = SLOTS[i]
        for g in allowed:
            if g in qual and g not in used:
                used.add(g); assign[sid] = g
                if bt(i + 1): return True
                used.discard(g); del assign[sid]
        return False
    return {g: sid for sid, g in assign.items()} if bt(0) else None
THIRD_MAP = {frozenset(c): match_thirds(frozenset(c)) for c in itertools.combinations("ABCDEFGHIJKL", 8)}
assert all(v is not None for v in THIRD_MAP.values())

R32 = {73: ("2A", "2B"), 74: ("1E", "M74"), 75: ("1F", "2C"), 76: ("1C", "2F"), 77: ("1I", "M77"), 78: ("2E", "2I"),
       79: ("1A", "M79"), 80: ("1L", "M80"), 81: ("1D", "M81"), 82: ("1G", "M82"), 83: ("2K", "2L"), 84: ("1H", "2J"),
       85: ("1B", "M85"), 86: ("1J", "2H"), 87: ("1K", "M87"), 88: ("2D", "2G")}
R16 = {89: (74, 77), 90: (73, 75), 91: (76, 78), 92: (79, 80), 93: (83, 84), 94: (81, 82), 95: (86, 88), 96: (85, 87)}
QF = {97: (89, 90), 98: (93, 94), 99: (91, 92), 100: (95, 96)}
SF = {101: (97, 98), 102: (99, 100)}

# ---------- 6. Monte Carlo ----------
reach = {k: np.zeros(48, int) for k in ["WinGroup", "Advance", "R16", "QF", "SF", "Final", "Champion"]}
KO = rng.random((N_SIM, 31))
for s in range(N_SIM):
    pts = [0.0] * 48; gd = [0.0] * 48; gf = [0.0] * 48
    for fi, (h, a) in enumerate(fixtures):
        hg, ag = HG[fi][s], AG[fi][s]
        gf[h] += hg; gf[a] += ag; gd[h] += hg - ag; gd[a] += ag - hg
        if hg > ag: pts[h] += 3
        elif hg < ag: pts[a] += 3
        else: pts[h] += 1; pts[a] += 1
    pos = {}
    for L in LETTERS:
        st = sorted(grp_teams[L], key=lambda t: (pts[t], gd[t], gf[t], elos[t]), reverse=True)
        pos[L] = st; reach["WinGroup"][st[0]] += 1; reach["Advance"][st[0]] += 1; reach["Advance"][st[1]] += 1
    thirds = sorted(((L, pos[L][2]) for L in LETTERS), key=lambda x: (pts[x[1]], gd[x[1]], gf[x[1]], elos[x[1]]), reverse=True)[:8]
    for _, t in thirds: reach["Advance"][t] += 1
    gmap = THIRD_MAP[frozenset(L for L, _ in thirds)]; third_team = {gmap[L]: t for L, t in thirds}
    def team_of(code):
        return pos[code[1]][0] if code[0] == "1" else pos[code[1]][1] if code[0] == "2" else third_team[code]
    win = {}; k = 0
    for m, (c1, c2) in R32.items():
        t1, t2 = team_of(c1), team_of(c2); w = t1 if KO[s, k] < ADV[t1, t2] else t2; win[m] = w; reach["R16"][w] += 1; k += 1
    for stage, nxt in [(R16, "QF"), (QF, "SF"), (SF, "Final")]:
        for m, (a, b) in stage.items():
            t1, t2 = win[a], win[b]; w = t1 if KO[s, k] < ADV[t1, t2] else t2; win[m] = w; reach[nxt][w] += 1; k += 1
    t1, t2 = win[101], win[102]; champ = t1 if KO[s, k] < ADV[t1, t2] else t2; reach["Champion"][champ] += 1

# ---------- 7. results ----------
df = pd.DataFrame({"team": teams, "elo": elos.round(0).astype(int),
                   "win_grp": reach["WinGroup"] / N_SIM, "advance": reach["Advance"] / N_SIM,
                   "reach_QF": reach["QF"] / N_SIM, "reach_SF": reach["SF"] / N_SIM,
                   "final": reach["Final"] / N_SIM, "CHAMPION": reach["Champion"] / N_SIM}).sort_values("CHAMPION", ascending=False)
pd.set_option("display.width", 200)
print(f"\n{'='*78}\n2026 WORLD CUP FORECAST  ({N_SIM:,} simulations,  Dixon-Coles rho={RHO:+.3f})\n{'='*78}")
fmt = df.head(24).copy()
for c in ["win_grp", "advance", "reach_QF", "reach_SF", "final", "CHAMPION"]:
    fmt[c] = (fmt[c] * 100).round(1)
print(fmt.to_string(index=False))
print(f"\n(sanity: champion probs sum to {df.CHAMPION.sum():.3f}; 48 teams)")
