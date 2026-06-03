"""Why does our Elo over-rate CONMEBOL/AFC and under-rate UEFA? Test whether the correction
signal exists in our own data: in cross-confederation games, does each confederation
over/under-perform its Elo expectation?"""

# --- run from anywhere: make project root importable + cwd ---
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)
# -------------------------------------------------------------

import numpy as np
import pandas as pd

import confed
import elo

feat, _ = elo.build_features(pd.read_csv("results.csv", parse_dates=["date"]))
d = feat[feat.result.notna() & (feat.date >= "2010-01-01")].copy()
d["hc"] = d.home_team.map(confed.confed_of)
d["ac"] = d.away_team.map(confed.confed_of)
d = d[(d.hc != "OTHER") & (d.ac != "OTHER")].copy()
ha = np.where(d.neutral, 0.0, 70.0)
d["exp"] = 1 / (1 + 10 ** (-((d.home_elo + ha) - d.away_elo) / 400))
d["act"] = np.where(d.home_score > d.away_score, 1.0, np.where(d.home_score < d.away_score, 0.0, 0.5))
d["resid"] = d.act - d.exp                       # + = home did better than Elo expected

print(f"Played matches (2010+, both mapped): {len(d)}")
print(f"Cross-confederation share: {(d.hc != d.ac).mean():.1%}   (the only games that calibrate pools against each other)")

cross = d[d.hc != d.ac]


def by_confed(sub):
    h = pd.DataFrame({"c": sub.hc, "r": sub.resid})
    a = pd.DataFrame({"c": sub.ac, "r": -sub.resid})
    return pd.concat([h, a]).groupby("c").r.agg(n="count", mean_resid="mean", se="sem")


print("\n=== CROSS-CONFED residual by confederation ===")
print("(actual - Elo-expected;  NEGATIVE = under-performs its Elo => our model OVER-rates it)")
print(by_confed(cross).round(3).sort_values("mean_resid").to_string())

for era, mask in [("2010-2017", cross.date < "2018-01-01"), ("2018-2026", cross.date >= "2018-01-01")]:
    print(f"\n--- {era}  (n={int(mask.sum())} cross-confed games) ---")
    print(by_confed(cross[mask]).mean_resid.round(3).sort_values().to_string())

# how many cross-confed games do the over-rated teams actually play?
print("\n=== cross-confed games played since 2018 (the over-rated suspects) ===")
recent = cross[cross.date >= "2018-01-01"]
for t in ["Argentina", "Colombia", "Ecuador", "Spain", "France", "England", "Brazil"]:
    n = ((recent.home_team == t) | (recent.away_team == t)).sum()
    print(f"   {t:10} {int(n)} cross-confed games")
