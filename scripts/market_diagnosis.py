"""Why do our title odds differ from the market on the favourites? Decompose into
rating (recent form), draw structure (group + path difficulty), and where in the bracket each team drops off."""
# --- run from anywhere ---
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)
# -------------------------

import numpy as np
import pandas as pd
import wc

# de-vigged market champion % (early June 2026 consensus, from market_comparison.py)
MKT = {"France": 15.0, "Spain": 15.0, "England": 11.0, "Argentina": 9.2, "Brazil": 9.2, "Portugal": 8.2,
       "Germany": 5.5, "Netherlands": 3.9, "Belgium": 2.4, "Colombia": 2.3, "United States": 2.0, "Morocco": 2.0}

m = wc.WorldCupModel()
rt = m.ratings_table().set_index("team")
grps = m.groups(); team2grp = {t: L for L, ts in grps.items() for t in ts}
sim = m.simulate_tournament(n_sim=30000, rd_scale=1.5).set_index("team")

res = pd.read_csv("results.csv", parse_dates=["date"])
recent = res[res.date >= "2024-06-01"]   # since Euro 2024 / Copa America 2024


def record(t):
    h, a = recent[recent.home_team == t], recent[recent.away_team == t]
    w = int((h.home_score > h.away_score).sum() + (a.away_score > a.home_score).sum())
    d = int((h.home_score == h.away_score).sum() + (a.away_score == a.home_score).sum())
    n = len(h) + len(a)
    gd = (h.home_score - h.away_score).sum() + (a.away_score - a.home_score).sum()
    return f"{w}-{d}-{n - w - d}", n, (gd / n if n else 0)


print(f"{'team':13}{'mkt%':>5}{'our%':>6}{'rating':>7}{'RD':>4}  grp  {'grpOppAvg':>9}{'recent(W-D-L)':>14}{'GD/g':>6}"
      f"{'adv%':>6}{'QF%':>5}{'SF%':>5}")
for t in ["Spain", "Argentina", "France", "Brazil", "England", "Portugal", "Germany", "Netherlands"]:
    L = team2grp.get(t, "?")
    opps = [x for x in grps.get(L, []) if x != t]
    oppavg = np.mean([rt.loc[x, "rating"] for x in opps]) if opps else 0
    rec, n, gdg = record(t)
    print(f"{t:13}{MKT.get(t, 0):5.0f}{sim.loc[t, 'champion'] * 100:6.1f}{rt.loc[t, 'rating']:7.0f}"
          f"{rt.loc[t, 'uncertainty']:4.0f}  {L:>2}  {oppavg:9.0f}{rec + f' ({n})':>14}{gdg:+6.1f}"
          f"{sim.loc[t, 'advance'] * 100:6.0f}{sim.loc[t, 'reach_QF'] * 100:5.0f}{sim.loc[t, 'reach_SF'] * 100:5.0f}")

print("\nGroup make-up (rating) for Spain vs Brazil:")
for t in ["Spain", "Brazil"]:
    L = team2grp[t]
    print(f"  Group {L}: " + ", ".join(f"{x} {rt.loc[x, 'rating']:.0f}" for x in sorted(grps[L], key=lambda x: -rt.loc[x, 'rating'])))

print("\nTop 10 by our rating:")
print(rt.head(10)[["rating", "uncertainty", "confederation"]].to_string())
