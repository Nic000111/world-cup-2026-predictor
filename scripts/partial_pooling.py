"""
partial_pooling.py — does a partial-pooling (empirical-Bayes shrinkage) cross-continental
adjustment beat the uniform per-confederation offset we ship?

The pooling spectrum, all estimated on TRAIN (<2024) cross-confed games, all fed into the
SAME W/D/L logistic, all scored on the SAME held-out 2024+ games:

  base         no cross-continental adjustment at all                      (reference floor)
  no-pool      each team's OWN raw cross-confed offset (noisy, small-n)     (over-fits)
  complete     every team gets its confederation mean  (= our idea)         (uniform offset)
  partial      shrink each team's own offset toward its confed mean by an
               amount set by how many cross-confed games it has             (the "proper" version)
  online k=1.5 our SHIPPED engine (online confed-Elo offset)                (what we run now)

Empirical Bayes (Gaussian-Gaussian):  theta_t ~ N(mu_confed, tau^2),  obs ~ N(theta_t, sigma^2/n_t)
  shrink weight on own data:  w_t = n_t*tau^2 / (n_t*tau^2 + sigma^2)
  shrunk offset:              theta_t = mu_confed + w_t * (ybar_t - mu_confed)
n_t large  -> w_t -> 1 (trust the team);   n_t small -> w_t -> 0 (shrink to the confed mean).

Adjustment is applied to CROSS-confed games only (additive gap term), exactly like the shipped
offset — so within-confed predictions are identical across every variant and the comparison
isolates the pooling rule. Leak-free: offsets + logistic both fit on train, scored on test.
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
SCALE = 400.0
K = np.log(10) / (4 * SCALE)        # score-units per Elo point near parity (~0.00144)
RAW = pd.read_csv("results.csv", parse_dates=["date"])


def ll(p, y):
    idx = np.array([CLASSES.index(t) for t in y])
    return -np.mean(np.log(np.clip(p[np.arange(len(y)), idx], 1e-15, 1)))


# ---- base (no confed) leak-free pre-match Elo, and the shipped online (confed on) build ----
p_off = dict(elo.DEFAULT_PARAMS); p_off["confed_elo"] = False
feat, _ = elo.build_features(RAW, p_off)
feat_on, _ = elo.build_features(RAW, dict(elo.DEFAULT_PARAMS))         # confed_elo=True, k=1.5 (shipped)

mask = feat.result.notna() & (feat.date >= "2010-01-01")
d = feat[mask].reset_index(drop=True)
gap_online = (feat_on.home_elo - feat_on.away_elo)[mask].reset_index(drop=True).values

d["hc"] = d.home_team.map(confed.confed_of)
d["ac"] = d.away_team.map(confed.confed_of)
cross = ((d.hc != d.ac) & (d.hc != "OTHER") & (d.ac != "OTHER")).values
yr = d.date.dt.year.values
tr, te = yr < 2024, yr >= 2024
base_gap = (d.home_elo - d.away_elo).values
flag, mom, rest = d.home_adv_flag.values, d.mom_diff.values, d.rest_diff.values
y = d.result.values

# ---- empirical-Bayes offsets, fit on TRAIN cross-confed games only ----
sub = d[tr & cross]
ha = np.where(sub.neutral, 0.0, 70.0)
E = 1 / (1 + 10 ** (-((sub.home_elo + ha) - sub.away_elo) / SCALE))
S = np.where(sub.home_score > sub.away_score, 1.0, np.where(sub.home_score < sub.away_score, 0.0, 0.5))
r = (S - E)                                   # home-perspective residual, score units
rec = pd.concat([pd.DataFrame({"team": sub.home_team.values, "r": r}),
                 pd.DataFrame({"team": sub.away_team.values, "r": -r})], ignore_index=True)
g = rec.groupby("team").r.agg(ybar="mean", n="count")
g["confed"] = g.index.map(confed.confed_of)
mu_c = rec.assign(confed=rec.team.map(confed.confed_of)).groupby("confed").r.mean()   # complete-pool means
rec["tmean"] = rec.team.map(g.ybar)
sigma2 = ((rec.r - rec.tmean) ** 2).sum() / (len(rec) - g.shape[0])                   # within-team game noise
dev = g.ybar - g.confed.map(mu_c)
tau2 = max(1e-9, np.average(dev ** 2 - sigma2 / g.n, weights=g.n))                    # between-team spread
g["w"] = g.n * tau2 / (g.n * tau2 + sigma2)
g["theta"] = g.confed.map(mu_c) + g.w * dev                                           # shrunk (score units)

# Elo-unit offset lookups for each pooling rule
def off(rule):
    if rule == "complete":  d_ = {t: mu_c[g.loc[t, "confed"]] / K for t in g.index}
    elif rule == "no":      d_ = {t: g.loc[t, "ybar"] / K for t in g.index}
    elif rule == "partial": d_ = {t: g.loc[t, "theta"] / K for t in g.index}
    # unseen mapped team -> confed mean (w=0); OTHER / no-confed -> 0
    default_c = {c: mu_c.get(c, 0.0) / K for c in ["UEFA", "CONMEBOL", "CAF", "AFC", "CONCACAF", "OFC"]}
    return d_, default_c

def gap_for(rule):
    seen, dc = off(rule)
    def th(team):
        c = confed.confed_of(team)
        if c == "OTHER": return 0.0
        return seen.get(team, dc.get(c, 0.0)) if rule != "no" else seen.get(team, 0.0)
    th_h = d.home_team.map(th).values
    th_a = d.away_team.map(th).values
    return base_gap + np.where(cross, th_h - th_a, 0.0)

# ---- fit logistic on train, score test (overall + cross-confed) ----
def fit_eval(gap):
    X = np.column_stack([gap, np.abs(gap), flag, mom, rest])
    m = Pipeline([("sc", StandardScaler()),
                  ("clf", LogisticRegression(C=0.01, max_iter=2000))]).fit(X[tr], y[tr])
    o = [list(m.classes_).index(c) for c in CLASSES]
    P = m.predict_proba(X[te])[:, o]
    xm = cross[te]
    return ll(P, y[te]), ll(P[xm], y[te][xm]), int(xm.sum())

# ---------------- output ----------------
print(f"Empirical-Bayes hyperparameters (fit on {int((tr&cross).sum())} train cross-confed games):")
print(f"  within-team game noise   sigma = {np.sqrt(sigma2)/K:6.0f} Elo")
print(f"  between-team spread       tau  = {np.sqrt(tau2)/K:6.0f} Elo")
print(f"  games for 50% own-weight       = {sigma2/tau2:6.1f}   (n where w_t = 0.5)\n")

sel = ["Mexico", "United States", "Japan", "South Korea", "Saudi Arabia", "Qatar",
       "Spain", "France", "Brazil", "Argentina", "Morocco"]
print("Shrinkage in action  (raw = own data only, mean = confed mean, partial = shrunk):")
print(f"  {'team':<14}{'confed':<9}{'n':>4}{'weight':>8}{'raw Elo':>9}{'confed mean':>13}{'-> partial':>12}")
for t in sel:
    if t not in g.index: continue
    c = g.loc[t, "confed"]
    print(f"  {t:<14}{c:<9}{int(g.loc[t,'n']):>4}{g.loc[t,'w']:>8.2f}"
          f"{g.loc[t,'ybar']/K:>9.0f}{mu_c[c]/K:>13.0f}{g.loc[t,'theta']/K:>12.0f}")

print("\n" + "=" * 64)
print("HELD-OUT 2024+ log-loss  (lower = better)")
print(f"  {'variant':<22}{'overall':>10}{'cross-confed':>15}")
print("  " + "-" * 47)
rows = [("base (no adjustment)", base_gap),
        ("no-pool (own data)", gap_for("no")),
        ("complete (confed mean)", gap_for("complete")),
        ("partial (shrunk)", gap_for("partial")),
        ("online k=1.5 (shipped)", gap_online)]
res = {}
for name, gp in rows:
    o, xc, n = fit_eval(gp)
    res[name] = (o, xc)
    print(f"  {name:<22}{o:>10.4f}{xc:>13.4f} (n={n})")

b = res["base (no adjustment)"]
print(f"\n  vs base, cross-confed:")
for name, (o, xc) in res.items():
    if name.startswith("base"): continue
    print(f"    {name:<24} {xc-b[1]:+.4f}")
