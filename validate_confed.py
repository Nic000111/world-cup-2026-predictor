"""
Does adding a confederation feature to the goals model fix CONMEBOL inflation?
Train <=2021, evaluate on 2022-23 validation (incl. the 2022 World Cup cross-confed games).
Test slice (2024+) untouched.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import confed
import elo

CLASSES = ["home_win", "draw", "away_win"]; MAXG = 10
NUM = ["scorer_elo", "opp_elo"]; PASS = ["is_home", "is_friendly"]; CAT = ["scorer_confed", "opp_confed"]

feat, RAT = elo.build_features(pd.read_csv("results.csv", parse_dates=["date"]))
d = feat[feat.result.notna() & (feat.date >= "2010-01-01")].copy().reset_index(drop=True)
d["is_friendly"] = d.tournament.str.contains("friendly", case=False).astype(float)
d["home_confed"] = d.home_team.map(confed.confed_of)
d["away_confed"] = d.away_team.map(confed.confed_of)
yr = d.date.dt.year; tr = (yr <= 2021).values; va = ((yr >= 2022) & (yr <= 2023)).values

tf = pd.concat([d.home_team, d.away_team]).value_counts()
un = [t for t in tf.index if confed.confed_of(t) == "OTHER"]
print(f"Confed coverage (both teams mapped): {((d.home_confed != 'OTHER') & (d.away_confed != 'OTHER')).mean():.1%}")
print("Top unmapped teams:", list(tf[tf.index.isin(un)].head(10).index))


def stack(r):
    h = pd.DataFrame(dict(goals=r.home_score.values, scorer_elo=r.home_elo.values, opp_elo=r.away_elo.values, is_home=np.where(r.neutral, 0., 1.), is_friendly=r.is_friendly.values, scorer_confed=r.home_confed.values, opp_confed=r.away_confed.values))
    a = pd.DataFrame(dict(goals=r.away_score.values, scorer_elo=r.away_elo.values, opp_elo=r.home_elo.values, is_home=0., is_friendly=r.is_friendly.values, scorer_confed=r.away_confed.values, opp_confed=r.home_confed.values))
    return pd.concat([h, a], ignore_index=True)


def pf(r, side):
    if side == "home":
        return pd.DataFrame(dict(scorer_elo=r.home_elo.values, opp_elo=r.away_elo.values, is_home=np.where(r.neutral, 0., 1.), is_friendly=r.is_friendly.values, scorer_confed=r.home_confed.values, opp_confed=r.away_confed.values))
    return pd.DataFrame(dict(scorer_elo=r.away_elo.values, opp_elo=r.home_elo.values, is_home=0., is_friendly=r.is_friendly.values, scorer_confed=r.away_confed.values, opp_confed=r.home_confed.values))


def make(with_confed):
    trs = [("num", StandardScaler(), NUM), ("pass", "passthrough", PASS)]
    if with_confed:
        trs.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT))
    return Pipeline([("pre", ColumnTransformer(trs, remainder="drop")), ("po", PoissonRegressor(alpha=1e-3, max_iter=5000))])


def tau(x, y, lh, la, r):
    t = np.ones(len(x)); m = (x == 0) & (y == 0); t[m] = 1 - lh[m] * la[m] * r; m = (x == 0) & (y == 1); t[m] = 1 + lh[m] * r
    m = (x == 1) & (y == 0); t[m] = 1 + la[m] * r; m = (x == 1) & (y == 1); t[m] = 1 - r; return t


def probs(lh, la, rho):
    g = np.arange(MAXG + 1); M = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
    M[0, 0] *= 1 - lh * la * rho; M[0, 1] *= 1 + lh * rho; M[1, 0] *= 1 + la * rho; M[1, 1] *= 1 - rho; M = np.clip(M, 0, None)
    h, dd, a = np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum(); return np.array([h, dd, a]) / (h + dd + a)


def fit(with_confed):
    m = make(with_confed); st = stack(d[tr]); m.fit(st.drop(columns="goals"), st.goals)
    lh, la = m.predict(pf(d[tr], "home")), m.predict(pf(d[tr], "away"))
    xt, yt = d[tr].home_score.values.astype(int), d[tr].away_score.values.astype(int)
    rho = float(minimize_scalar(lambda r: -np.sum(np.log(np.clip(tau(xt, yt, lh, la, r), 1e-9, None))), bounds=(-0.2, 0.2), method="bounded").x)
    return m, rho


def ll(m, rho, mask):
    r = d[mask]; lh, la = m.predict(pf(r, "home")), m.predict(pf(r, "away"))
    P = np.vstack([probs(lh[i], la[i], rho) for i in range(int(mask.sum()))])
    y = np.where(r.home_score.values > r.away_score.values, "home_win", np.where(r.home_score.values < r.away_score.values, "away_win", "draw"))
    pos = np.array([CLASSES.index(t) for t in y]); return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), pos], 1e-15, 1))))


base, rb = fit(False); conf, rc = fit(True)
cross = va & (d.home_confed != d.away_confed).values & (d.home_confed != "OTHER").values & (d.away_confed != "OTHER").values
wc22 = va & (d.tournament == "FIFA World Cup").values
print("\nValidation log-loss (lower = better):")
print(f"{'slice':22}{'baseline':>10}{'+confed':>10}   n")
for lab, mask in [("all val", va), ("cross-confed val", cross), ("2022 WC (val)", wc22)]:
    print(f"{lab:22}{ll(base, rb, mask):>10.4f}{ll(conf, rc, mask):>10.4f}   {int(mask.sum())}")

names = conf.named_steps["pre"].get_feature_names_out(); coef = conf.named_steps["po"].coef_
print("\nLearned scorer adjustment per confederation (+ scores MORE than Elo predicts, - = LESS):")
for n, c in sorted(zip(names, coef), key=lambda x: x[1]):
    if "scorer_confed" in n:
        print(f"   {n.split('__')[-1].replace('scorer_confed_', ''):12} {c:+.3f}")

print("\nEffect on some cross-confed matchups (neutral) — home/draw/away:")
def demo(m, rho, h, a):
    hh = pd.DataFrame([dict(scorer_elo=RAT[h], opp_elo=RAT[a], is_home=0., is_friendly=0., scorer_confed=confed.confed_of(h), opp_confed=confed.confed_of(a))])
    aa = pd.DataFrame([dict(scorer_elo=RAT[a], opp_elo=RAT[h], is_home=0., is_friendly=0., scorer_confed=confed.confed_of(a), opp_confed=confed.confed_of(h))])
    return probs(m.predict(hh)[0], m.predict(aa)[0], rho)
for h, a in [("Colombia", "Portugal"), ("Colombia", "Netherlands"), ("Ecuador", "Germany"), ("Brazil", "England")]:
    pb, pc = demo(base, rb, h, a), demo(conf, rc, h, a)
    print(f"   {h:9} vs {a:12}: baseline {pb[0]:.0%}/{pb[1]:.0%}/{pb[2]:.0%}  ->  +confed {pc[0]:.0%}/{pc[1]:.0%}/{pc[2]:.0%}")
