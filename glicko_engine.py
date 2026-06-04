"""
glicko_engine.py - uncertainty-aware (Glicko) rating engine + feature builder.

The shipped strength engine. Replaces Elo: each team carries a rating AND a rating-deviation
(RD = how sure we are), so the update step ADAPTS to confidence:
  * uncertain / new / long-idle teams update fast and predict closer to 50/50,
  * beating a team we know little about (high opponent RD) moves us less,
  * RD grows during inactivity (a team off the radar is fuzzier).
Elo's single fixed step-size throws all of that away — and on a head-to-head, fairly-tuned
on the same data, Glicko beat Elo by ~0.011 test log-loss (gap-only) / ~0.0055 in the full
model. See scripts/glicko.py for the tuning + comparison.

Kept from Elo (orthogonal): the per-confederation offset (updates on cross-continental games),
importance weighting, MOV, home advantage (in the update; prediction home-effect is the flag).
Interface mirrors elo.build_features so wc.py / the dashboard barely change. Leak-free by
construction (every recorded rating/RD is pre-match).
"""

import math
from collections import defaultdict

import numpy as np
import pandas as pd

import confed
from elo import importance_weight, mov_multiplier   # shared, identical weighting

Q = math.log(10) / 400.0
PI2 = math.pi * math.pi

# tuned by coordinate descent on 2022-23 validation log-loss (scripts/glicko.py)
DEFAULT_PARAMS = dict(
    home_advantage=40.0,   # applied in the UPDATE only; prediction home-effect lives in the flag
    start_rating=1500.0,
    rd_init=320.0,         # initial AND max rating deviation (uncertainty ceiling)
    rd_floor=35.0,         # min RD - a team is never treated as perfectly known
    drift=2.0,             # RD growth per sqrt(day) of inactivity
    use_mov=True,
    k_confed=2.0,          # confederation-offset learning rate (cross-confed games only)
)
MOM_WINDOW = 10
REST_CAP_DAYS = 14


def gfac(RD):
    return 1.0 / math.sqrt(1.0 + 3.0 * Q * Q * RD * RD / PI2)


def compute_glicko(df, params=DEFAULT_PARAMS):
    """One causal pass. Adds pre-match columns home_elo/away_elo (effective rating = team +
    confederation offset), home_rd/away_rd (uncertainty), home_mom/away_mom, home_rest/away_rest.
    Returns (df_with_columns, final_effective_ratings, final_rd)."""
    sr, HA, RD0 = params["start_rating"], params["home_advantage"], params["rd_init"]
    rdmin, c, usemov, kc = params["rd_floor"], params["drift"], params["use_mov"], params["k_confed"]
    R = {}; RD = {}; conf = defaultdict(float); last = {}; hist = defaultdict(list)
    cols = {k: [] for k in ["home_elo", "away_elo", "home_rd", "away_rd",
                            "home_mom", "away_mom", "home_rest", "away_rest"]}
    for row in df.itertuples(index=False):
        h, a, d = row.home_team, row.away_team, row.date
        rh, ra = R.get(h, sr), R.get(a, sr)
        RDh, RDa = RD.get(h, RD0), RD.get(a, RD0)
        if h in last:
            RDh = min(math.sqrt(RDh * RDh + c * c * (d - last[h]).days), RD0)
        if a in last:
            RDa = min(math.sqrt(RDa * RDa + c * c * (d - last[a]).days), RD0)
        ch, ca = confed.confed_of(h), confed.confed_of(a)
        oh = conf[ch] if ch != "OTHER" else 0.0
        oa = conf[ca] if ca != "OTHER" else 0.0
        eff_h, eff_a = rh + oh, ra + oa
        ha = 0.0 if bool(row.neutral) else HA

        cols["home_elo"].append(eff_h); cols["away_elo"].append(eff_a)
        cols["home_rd"].append(RDh); cols["away_rd"].append(RDa)
        cols["home_rest"].append(min((d - last[h]).days, REST_CAP_DAYS) if h in last else REST_CAP_DAYS)
        cols["away_rest"].append(min((d - last[a]).days, REST_CAP_DAYS) if a in last else REST_CAP_DAYS)
        hh, aa = hist[h], hist[a]
        cols["home_mom"].append(eff_h - hh[-MOM_WINDOW] if len(hh) >= MOM_WINDOW else 0.0)
        cols["away_mom"].append(eff_a - aa[-MOM_WINDOW] if len(aa) >= MOM_WINDOW else 0.0)
        hh.append(eff_h); aa.append(eff_a)

        if pd.isna(row.home_score) or pd.isna(row.away_score):
            continue                                   # unplayed fixture -> record only, no state change
        s = 1.0 if row.home_score > row.away_score else (0.0 if row.home_score < row.away_score else 0.5)
        w = importance_weight(row.tournament) * (mov_multiplier(row.home_score - row.away_score) if usemov else 1.0)
        # home update (its rating carries +ha in the expectation)
        go = gfac(RDa); Es = 1.0 / (1 + 10 ** (-go * ((eff_h + ha) - eff_a) / 400.0))
        info = w * Q * Q * go * go * Es * (1 - Es); RD2 = 1.0 / (1.0 / (RDh * RDh) + info)
        R[h] = rh + Q * RD2 * w * go * (s - Es); RD[h] = max(math.sqrt(RD2), rdmin)
        # away update (opponent home carries +ha)
        go = gfac(RDh); Ea = 1.0 / (1 + 10 ** (-go * (eff_a - (eff_h + ha)) / 400.0))
        info = w * Q * Q * go * go * Ea * (1 - Ea); RD2 = 1.0 / (1.0 / (RDa * RDa) + info)
        R[a] = ra + Q * RD2 * w * go * ((1 - s) - Ea); RD[a] = max(math.sqrt(RD2), rdmin)
        # confederation offset (cross-confed games only, shared & zero-sum)
        if kc and ch != ca and ch != "OTHER" and ca != "OTHER":
            dc = kc * w * (s - Es); conf[ch] += dc; conf[ca] -= dc
        last[h] = d; last[a] = d

    out = df.copy()
    for k, v in cols.items():
        out[k] = v
    eff_final = {t: R[t] + (conf[confed.confed_of(t)] if confed.confed_of(t) != "OTHER" else 0.0) for t in R}
    out.attrs["confed_offsets"] = dict(conf)
    out.attrs["final_rd"] = dict(RD)
    return out, eff_final, dict(RD)


def build_features(df, params=DEFAULT_PARAMS):
    """Run the Glicko pass and assemble the model-ready feature table. Mirrors elo.build_features:
    returns (feature_df, final_ratings). feature_df.attrs['final_rd'] holds per-team uncertainty."""
    df = df.sort_values("date").reset_index(drop=True).copy()
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE")
    df, final_ratings, final_rd = compute_glicko(df, params)

    res = np.select([df.home_score > df.away_score, df.home_score < df.away_score],
                    ["home_win", "away_win"], default="draw")
    played = df.home_score.notna() & df.away_score.notna()
    df["result"] = pd.Series(res, index=df.index).where(played)

    df["home_adv_flag"] = (~df["neutral"]).astype(int)
    df["rating_gap"] = df.home_elo - df.away_elo
    df["abs_gap"] = df.rating_gap.abs()
    df["rd_sum"] = df.home_rd + df.away_rd                     # combined uncertainty -> hedge toward 50/50
    df["mom_diff"] = df.home_mom - df.away_mom                 # available but NOT in V1 (dropped: redundant w/ Glicko)
    df["rest_diff"] = df.home_rest - df.away_rest
    df.attrs["final_rd"] = final_rd
    return df, final_ratings


# the columns the model feeds in (forward-selected: momentum & rest dropped, uncertainty kept)
V1_FEATURES = ["rating_gap", "home_adv_flag", "abs_gap", "rd_sum"]


if __name__ == "__main__":
    feat, final = build_features(pd.read_csv("results.csv", parse_dates=["date"]))
    print("Top 10 Glicko ratings (rating ± uncertainty):")
    rd = feat.attrs["final_rd"]
    for t, r in sorted(final.items(), key=lambda kv: -kv[1])[:10]:
        print(f"   {t:<16} {r:6.0f} ± {rd.get(t, float('nan')):3.0f}")
    print("\nV1 features:", V1_FEATURES)
    print("confed offsets:", {k: round(v) for k, v in feat.attrs['confed_offsets'].items()})
