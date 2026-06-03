"""
final_test.py - the SINGLE honest read on the frozen test set (2024+).

Models are fit on all pre-test data (<=2023) with the hyperparameters already chosen on
validation, then evaluated ONCE on 2024+. No tuning here. Val is shown alongside test purely
as a no-overfit consistency check.
"""
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
TRAIN = (yr <= 2021).values
VAL = ((yr >= 2022) & (yr <= 2023)).values
TRAINVAL = (yr <= 2023).values
TEST = (yr >= 2024).values


def y_of(mask):
    r = d.loc[mask]
    return np.where(r.home_score.values > r.away_score.values, "home_win",
                    np.where(r.home_score.values < r.away_score.values, "away_win", "draw"))


def metrics(P, y):
    pos = np.array([CLASSES.index(t) for t in y])
    pt = np.clip(P[np.arange(len(y)), pos], 1e-15, 1)
    oh = np.zeros_like(P); oh[np.arange(len(y)), pos] = 1
    pred = np.array(CLASSES)[P.argmax(1)]
    return dict(logloss=float(-np.mean(np.log(pt))),
                rps=float(np.mean(((np.cumsum(P, 1) - np.cumsum(oh, 1)) ** 2).sum(1) / 2)),
                acc=float((pred == y).mean()),
                draw_rec=float((pred[y == "draw"] == "draw").mean()),
                mdp=float(P[:, 1].mean()))


# ---- Elo-logistic ----
F = elo.V1_FEATURES
def fit_lr(mask):
    return Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(C=0.01, max_iter=2000))]).fit(d.loc[mask, F], d.loc[mask, "result"])
def lr_P(m, mask):
    return m.predict_proba(d.loc[mask, F])[:, [list(m.classes_).index(c) for c in CLASSES]]


# ---- goals model (Elo-Poisson + Dixon-Coles) ----
def build_long(rows):
    h = pd.DataFrame(dict(goals=rows.home_score.values, scorer_elo=rows.home_elo.values, opp_elo=rows.away_elo.values,
                          is_home=np.where(rows.neutral, 0., 1.), is_friendly=rows.is_friendly.values))
    a = pd.DataFrame(dict(goals=rows.away_score.values, scorer_elo=rows.away_elo.values, opp_elo=rows.home_elo.values,
                          is_home=0., is_friendly=rows.is_friendly.values))
    return pd.concat([h, a], ignore_index=True)
def lambdas(gm, rows):
    hh = pd.DataFrame(dict(scorer_elo=rows.home_elo.values, opp_elo=rows.away_elo.values, is_home=np.where(rows.neutral, 0., 1.), is_friendly=rows.is_friendly.values))
    aa = pd.DataFrame(dict(scorer_elo=rows.away_elo.values, opp_elo=rows.home_elo.values, is_home=0., is_friendly=rows.is_friendly.values))
    return gm.predict(hh[FEATS]), gm.predict(aa[FEATS])
def tau(x, y, lh, la, rho):
    t = np.ones(len(x))
    m = (x == 0) & (y == 0); t[m] = 1 - lh[m] * la[m] * rho
    m = (x == 0) & (y == 1); t[m] = 1 + lh[m] * rho
    m = (x == 1) & (y == 0); t[m] = 1 + la[m] * rho
    m = (x == 1) & (y == 1); t[m] = 1 - rho
    return t
def match_probs(lh, la, rho):
    g = np.arange(MAXG + 1); M = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
    M[0, 0] *= 1 - lh * la * rho; M[0, 1] *= 1 + lh * rho; M[1, 0] *= 1 + la * rho; M[1, 1] *= 1 - rho
    M = np.clip(M, 0, None); h, dr, a = np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()
    return np.array([h, dr, a]) / (h + dr + a)
def fit_goals(mask):
    lo = build_long(d[mask])
    gm = Pipeline([("sc", StandardScaler()), ("po", PoissonRegressor(alpha=1e-4, max_iter=5000))]).fit(lo[FEATS], lo["goals"])
    lh, la = lambdas(gm, d[mask]); xt, yt = d[mask].home_score.values.astype(int), d[mask].away_score.values.astype(int)
    rho = minimize_scalar(lambda r: -np.sum(np.log(np.clip(tau(xt, yt, lh, la, r), 1e-9, None))), bounds=(-0.2, 0.2), method="bounded").x
    return gm, rho
def goals_P(gm, rho, mask):
    lh, la = lambdas(gm, d[mask]); return np.vstack([match_probs(lh[i], la[i], rho) for i in range(mask.sum())])


lr_tr, lr_tv = fit_lr(TRAIN), fit_lr(TRAINVAL)
gm_tr, rho_tr = fit_goals(TRAIN); gm_tv, rho_tv = fit_goals(TRAINVAL)
yval, ytest = y_of(VAL), y_of(TEST)

res = {
    "Elo-logistic": (metrics(lr_P(lr_tr, VAL), yval), metrics(lr_P(lr_tv, TEST), ytest)),
    "Goals (Poisson+DC)": (metrics(goals_P(gm_tr, rho_tr, VAL), yval), metrics(goals_P(gm_tv, rho_tv, TEST), ytest)),
}
P_ens = 0.5 * lr_P(lr_tv, TEST) + 0.5 * goals_P(gm_tv, rho_tv, TEST)
res["Ensemble 50/50"] = (None, metrics(P_ens, ytest))

fav = np.where(d.loc[TEST, "elo_p_home"].values >= 0.5, "home_win", "away_win")
base = d.loc[TRAINVAL, "result"].value_counts(normalize=True)
base_ll = -np.mean(np.log([base[t] for t in ytest]))

print(f"TEST: {TEST.sum()} matches (2024+) | split {pd.Series(ytest).value_counts(normalize=True).round(3).to_dict()}")
print(f"floors (test): uniform LL {np.log(3):.3f} | base-rate LL {base_ll:.3f} | pick-higher-Elo acc {(fav==ytest).mean():.3f}\n")
print(f"{'model':22}{'val_LL':>8}{'test_LL':>9}{'test_RPS':>10}{'test_acc':>10}{'draw_rec':>10}{'mean_dp':>9}")
for k, (v, t) in res.items():
    vll = f"{v['logloss']:.4f}" if v else "   --"
    print(f"{k:22}{vll:>8}{t['logloss']:>9.4f}{t['rps']:>10.4f}{t['acc']:>10.4f}{t['draw_rec']:>10.3f}{t['mdp']:>9.3f}")

P = goals_P(gm_tv, rho_tv, TEST)
cal = pd.DataFrame({"p": P[:, 0], "home": (ytest == "home_win").astype(int)})
cal["bin"] = pd.cut(cal.p, np.linspace(0, 1, 11))
print("\nCalibration - Goals model, home-win prob (test):")
print(cal.groupby("bin", observed=True).agg(n=("home", "size"), predicted=("p", "mean"), actual=("home", "mean")).round(3).to_string())
