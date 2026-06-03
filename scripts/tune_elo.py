"""
tune_elo.py - tune Elo hyperparameters on the 2022-23 validation slice by minimising
Elo's own expected-score log-loss (elo_p_home vs actual home points), then confirm the
tuned ratings also help the downstream 3-class logistic. Leak-free: Elo is causal and we
never touch the 2024+ test slice for *selection*.
"""

# --- run from anywhere: make project root importable + cwd ---
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)
# -------------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import elo

df = pd.read_csv("results.csv", parse_dates=["date"])
base = df.sort_values("date").reset_index(drop=True).copy()
base["neutral"] = base["neutral"].astype(str).str.upper().eq("TRUE")
year = base.date.dt.year.values
played = base.home_score.notna().values & base.away_score.notna().values
res = np.where(base.home_score > base.away_score, "home_win",
               np.where(base.home_score < base.away_score, "away_win", "draw"))
home_pts = np.where(res == "home_win", 1.0, np.where(res == "draw", 0.5, 0.0))
VAL = played & (year >= 2022) & (year <= 2023)
TEST = played & (year >= 2024)


def expscore_logloss(p, mask):
    pp = np.clip(p[mask], 1e-12, 1 - 1e-12)
    y = home_pts[mask]
    return float(-np.mean(y * np.log(pp) + (1 - y) * np.log(1 - pp)))


def elo_pred(params):
    d2, _, _ = elo.compute_elo(base, params)
    ha = np.where(base.neutral.values, 0.0, params["home_advantage"])
    return 1.0 / (1.0 + 10 ** (-((d2.home_elo.values + ha) - d2.away_elo.values) / params["scale"]))


def mk(k, ha, mov):
    return dict(k_base=float(k), home_advantage=float(ha), start_rating=1500.0, scale=400.0, use_mov=mov)


print("Grid-searching Elo params on val (2022-23) expected-score log-loss ...")
rows = []
for k in [10, 15, 20, 25, 30, 40, 50]:
    for ha in [0, 40, 70, 100, 130, 160]:
        for mov in [True, False]:
            p = elo_pred(mk(k, ha, mov))
            rows.append(dict(k_base=k, home_adv=ha, mov=mov,
                             val_ll=expscore_logloss(p, VAL), test_ll=expscore_logloss(p, TEST)))
R = pd.DataFrame(rows).sort_values("val_ll").reset_index(drop=True)
print("\nTop 10 by val expected-score log-loss:")
print(R.head(10).round(4).to_string(index=False))

best = R.iloc[0]
default = R[(R.k_base == 20) & (R.home_adv == 100) & (R.mov)].iloc[0]
print(f"\nDEFAULT  k=20 ha=100 mov=True :  val {default.val_ll:.4f}  test {default.test_ll:.4f}")
print(f"BEST     k={int(best.k_base)} ha={int(best.home_adv)} mov={best.mov} :  val {best.val_ll:.4f}  test {best.test_ll:.4f}")


# ---- downstream check: does the tuned Elo also help the 3-class logistic? ----
CLASSES = ["home_win", "draw", "away_win"]


def three_class_ll(params):
    feat, _ = elo.build_features(df, params)
    d = feat[feat.result.notna() & (feat.date >= "2010-01-01")].copy().reset_index(drop=True)
    yr = d.date.dt.year
    tr, va, te = (yr <= 2021).values, ((yr >= 2022) & (yr <= 2023)).values, (yr >= 2024).values
    F = elo.V1_FEATURES
    m = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(C=0.01, max_iter=2000))])
    m.fit(d.loc[tr, F], d.loc[tr, "result"])
    out = {}
    for nm, msk in [("val", va), ("test", te)]:
        pp = m.predict_proba(d.loc[msk, F])
        idx = [list(m.classes_).index(c) for c in CLASSES]
        pp = pp[:, idx]
        y = d.loc[msk, "result"].values
        pos = [CLASSES.index(t) for t in y]
        out[nm] = float(-np.mean(np.log(np.clip(pp[np.arange(len(y)), pos], 1e-15, 1))))
    return out


best_params = mk(int(best.k_base), int(best.home_adv), bool(best.mov))
d_def = three_class_ll(elo.DEFAULT_PARAMS)
d_best = three_class_ll(best_params)
print("\n3-class logistic (full) log-loss  [the metric we actually care about]:")
print(f"  DEFAULT Elo params:  val {d_def['val']:.4f}  test {d_def['test']:.4f}")
print(f"  TUNED   Elo params:  val {d_best['val']:.4f}  test {d_best['test']:.4f}")
print("\nbest_params =", best_params)
