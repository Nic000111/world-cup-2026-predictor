"""Diagnostic: why is test log-loss (~1.8) so much worse than val (~0.89)?"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import elo

CLASSES = ["home_win", "draw", "away_win"]
feat, _ = elo.build_features(pd.read_csv("results.csv", parse_dates=["date"]))
d = feat[feat.result.notna() & (feat.date >= "2010-01-01")].copy().reset_index(drop=True)
yr = d.date.dt.year
tr = (yr <= 2021).values
va = ((yr >= 2022) & (yr <= 2023)).values
te = (yr >= 2024).values
F = elo.V1_FEATURES

m = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(C=0.1, max_iter=3000))])
m.fit(d.loc[tr, F], d.loc[tr, "result"])


def probs(msk):
    X = d.loc[msk, F]
    y = d.loc[msk, "result"].values
    p = m.predict_proba(X)
    idx = [list(m.classes_).index(c) for c in CLASSES]
    p = p[:, idx]
    pt = p[np.arange(len(y)), [CLASSES.index(t) for t in y]]
    return p, y, pt


print("=== val vs test (logistic full, trained on TRAIN only) ===")
for nm, msk in [("val", va), ("test", te)]:
    p, y, pt = probs(msk)
    print(f"{nm}: n={len(y)} logloss={log_loss(y, p, labels=CLASSES):.4f} "
          f"mean_p_true={pt.mean():.4f} min={pt.min():.6f} "
          f"frac<0.02={np.mean(pt < 0.02):.4f} frac<0.05={np.mean(pt < 0.05):.4f}")
    for c in CLASSES:
        cm = y == c
        print(f"    {c:9} n={cm.sum():4d} mean_logloss={-np.log(pt[cm]).mean():.3f} mean_p_true={pt[cm].mean():.3f}")

p, y, pt = probs(te)
tt = d.loc[te].copy()
tt["pt"] = pt
tt["ll"] = -np.log(pt)
tt["yr"] = tt.date.dt.year

print("\n=== test log-loss by year ===")
print(tt.groupby("yr").agg(n=("ll", "size"), logloss=("ll", "mean"), mean_p_true=("pt", "mean")).round(3).to_string())

print("\n=== test log-loss: friendly vs competitive ===")
tt["friendly"] = tt.tournament.str.contains("friendly", case=False)
print(tt.groupby("friendly").agg(n=("ll", "size"), logloss=("ll", "mean"), mean_p_true=("pt", "mean")).round(3).to_string())

print("\n=== worst 12 test rows by log-loss ===")
cols = ["date", "home_team", "away_team", "tournament", "rating_gap", "home_adv_flag", "mom_diff", "rest_diff", "result", "pt"]
print(tt.sort_values("ll", ascending=False)[cols].head(12).to_string(index=False))

print("\n=== feature ranges: train vs test ===")
for c in F:
    print(f"  {c:14} train[{d.loc[tr, c].min():8.1f},{d.loc[tr, c].max():8.1f}]   test[{d.loc[te, c].min():8.1f},{d.loc[te, c].max():8.1f}]")
