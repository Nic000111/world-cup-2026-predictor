"""Benchmark our title-odds simulation against the bookmaker market (early June 2026)."""

# --- run from anywhere: make project root importable + cwd ---
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)
# -------------------------------------------------------------

import numpy as np
import pandas as pd

import wc

# market outright odds as "X-to-1" (early June 2026, Yahoo/DraftKings/BetMGM consensus)
MKT = {"France": 4.5, "Spain": 4.5, "England": 6.5, "Argentina": 8, "Brazil": 8, "Portugal": 9, "Germany": 14,
       "Netherlands": 20, "Belgium": 33, "Colombia": 35, "Morocco": 40, "United States": 40, "Japan": 50, "Uruguay": 50,
       "Croatia": 66, "Ecuador": 66, "Mexico": 66, "Switzerland": 66, "Austria": 100, "Canada": 150, "Paraguay": 150,
       "Czech Republic": 200, "Ivory Coast": 200, "Algeria": 250, "Bosnia and Herzegovina": 250, "Egypt": 250,
       "Ghana": 250, "South Korea": 250, "Scotland": 250, "Australia": 500, "Iran": 500, "Tunisia": 500, "DR Congo": 750,
       "Cape Verde": 1000, "Iraq": 1000, "Jordan": 1000, "New Zealand": 1000, "Panama": 1000, "Qatar": 1000,
       "Saudi Arabia": 1000, "South Africa": 1000, "Uzbekistan": 1000, "Curaçao": 2500, "Haiti": 2500}
imp = {t: 1 / (o + 1) for t, o in MKT.items()}
overround = sum(imp.values())
mkt_p = {t: v / overround for t, v in imp.items()}      # de-vigged "true" market probability

m = wc.WorldCupModel()
sim = m.simulate_tournament(n_sim=30000, rating_sd=125)
ours = dict(zip(sim.team, sim.champion))

df = pd.DataFrame({"team": list(mkt_p)})
df["market"] = df.team.map(lambda t: mkt_p[t] * 100)
df["ours"] = df.team.map(lambda t: ours.get(t, 0.0) * 100)
df["diff"] = df["ours"] - df["market"]
df = df.sort_values("market", ascending=False).reset_index(drop=True)

print(f"Bookmaker overround {overround * 100:.0f}%  ->  de-vigged to fair probabilities\n")
print("TITLE ODDS: market vs our model (top 18 by market)")
print(df.head(18).round(1).to_string(index=False))

a, b = df["ours"].values, df["market"].values
print(f"\ncorrelation(our, market) = {np.corrcoef(a, b)[0, 1]:.3f}   |   mean abs divergence = {np.abs(a - b).mean():.1f} pts")

print("\nMost OVER-rated by us vs the market:")
print(df.sort_values("diff", ascending=False).head(6).round(1).to_string(index=False))
print("\nMost UNDER-rated by us vs the market:")
print(df.sort_values("diff").head(6).round(1).to_string(index=False))

# continental tilt
import collections
conf_diff = collections.defaultdict(float)
import confed
for _, r in df.iterrows():
    conf_diff[confed.confed_of(r.team)] += r["diff"]
print("\nNet over/under-rating by confederation (sum of diffs, + = we're higher than market):")
for c, v in sorted(conf_diff.items(), key=lambda x: -x[1]):
    print(f"   {c:9} {v:+.1f} pts")
