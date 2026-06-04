"""
diagnostics.py — quick model sanity checks, consolidated.

Three checks, run all with `python scripts/diagnostics.py`:
  1. draws        — how the model treats draws (group forecast + held-out test)
  2. consistency  — the headline score always agrees with the W/D/L pick, and the higher-xG
                    side is always the favourite (so xG never contradicts the pick)
  3. score_vs_outcome — illustrates why the single most-likely *score* can be a draw while the
                    most-likely *outcome* is a win (the joint-mode vs marginal-mode distinction)
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

import wc
import glicko_engine as eng

CLASSES = ["home_win", "draw", "away_win"]


def _fav(g):
    return ["H", "D", "A"][int(np.argmax([g["home"], g["draw"], g["away"]]))]


def _winner_of(score):
    h, a = map(int, score.split("-"))
    return "H" if h > a else ("A" if a > h else "D")


def draws(m):
    print("=== DRAWS ===")
    gf = m.group_fixtures()
    pick = np.array(["H", "D", "A"])[gf[["home_win", "draw", "away_win"]].values.argmax(1)]
    print(f"group forecast: draw is the W/D/L headline pick in {(pick == 'D').sum()}/{len(gf)}")
    print(f"               draw prob mean {gf.draw.mean()*100:.0f}%, max {gf.draw.max()*100:.0f}%")

    # held-out test (train <2024, score 2024+) — actual vs predicted draw behaviour
    feat, _ = eng.build_features(pd.read_csv("results.csv", parse_dates=["date"]))
    d = feat[feat.result.notna() & (feat.date >= "2010-01-01")].copy()
    yr = d.date.dt.year.values
    F = eng.V1_FEATURES
    mdl = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(C=0.01, max_iter=2000))]) \
        .fit(d.loc[yr < 2024, F], d.result.values[yr < 2024])
    order = [list(mdl.classes_).index(c) for c in CLASSES]
    Pt = mdl.predict_proba(d.loc[yr >= 2024, F])[:, order]
    yt = d.result.values[yr >= 2024]
    tpick = np.array(CLASSES)[Pt.argmax(1)]
    print(f"held-out test: actual draw rate {(yt == 'draw').mean()*100:.0f}%, "
          f"predicted-draw mean {Pt[:, 1].mean()*100:.0f}%, draw is modal pick {(tpick == 'draw').mean()*100:.0f}%")


def consistency(m):
    print("\n=== CONSISTENCY (headline score vs W/D/L, and xG vs favourite) ===")
    teams = m.ratings_table().head(60).team.tolist()
    n = bad_score = bad_xg = 0
    for i in range(len(teams)):
        for j in range(len(teams)):
            if i == j:
                continue
            p = m.predict_match(teams[i], teams[j], neutral=True)
            g, fav = p["goals"], _fav(p["goals"])
            n += 1
            if _winner_of(p["likely_score"]) != fav:
                bad_score += 1
            xg_fav = "H" if p["xg"][0] > p["xg"][1] else ("A" if p["xg"][1] > p["xg"][0] else "D")
            if xg_fav != fav:
                bad_xg += 1
    print(f"{n} matchups — headline score disagrees with pick: {bad_score}; higher-xG disagrees with fav: {bad_xg}")


def score_vs_outcome(m):
    print("\n=== SCORE vs OUTCOME (joint mode can be a draw while the outcome is a win) ===")
    r = m.predict_match("Mexico", "South Korea", neutral=True)
    g = r["goals"]
    print(f"Mexico vs South Korea: W/D/L {g['home']*100:.0f}/{g['draw']*100:.0f}/{g['away']*100:.0f}  "
          f"(headline score {r['likely_score']})")
    for sc, p in r["top_scores"]:
        h, a = sc.split("-")
        bucket = "draw" if h == a else ("HOME" if int(h) > int(a) else "away")
        print(f"   {sc}  {p*100:4.1f}%  [{bucket}]")


if __name__ == "__main__":
    m = wc.WorldCupModel()
    draws(m)
    consistency(m)
    score_vs_outcome(m)
