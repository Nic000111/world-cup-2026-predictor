"""
elo.py - V1 Elo rating engine + feature builder for the World Cup 2026 predictor.

ONE chronological (causal) pass over every match in history:
  - record each team's PRE-match rating          -> leak-free strength signal
  - record momentum  = Elo change over last N games (recent over-performance vs own rating)
  - record rest days = days since the team's last match (capped)
then update ratings from the result.

Design (V1):
  * Home advantage stays INSIDE the Elo update + the `elo_p_home` baseline ONLY.
    The MODEL features use the PURE rating gap + a home/neutral flag, so the model
    learns the home effect itself (Option A).
  * Elo is computed over the FULL history (burn-in); the 2010+ training slice is taken later.
  * Leak-free by construction: a match's features never see its own (or any later) result.

V1 model features:  rating_gap, home_adv_flag, abs_gap, mom_diff, rest_diff
Baseline (NOT fed): elo_p_home   (Elo's own home-win prob -> tuning target + benchmark)
"""

from collections import defaultdict

import numpy as np
import pandas as pd

import confed

# ---- Elo hyperparameters (ALL tunable later, on a validation time-slice) ----
DEFAULT_PARAMS = dict(
    k_base=20.0,           # base step size (k=15 was ~tied on val; kept 20 to avoid rescaling churn)
    home_advantage=70.0,   # tuned on 2022-23 val; matches the modern ~+0.5-goal home edge (was 100)
    start_rating=1500.0,   # initial rating for every team (washes out after burn-in)
    scale=400.0,           # logistic scale
    use_mov=True,          # scale K by margin of victory
    confed_elo=True,       # two-level Elo: per-confederation offset, pooled from cross-confed games
    k_confed=1.5,          # tuned on 2022-23 val (cross-confed log-loss); kept well below the raw
                           #   loss-minimum (k>=8) so offsets stay football-plausible, not OFC-overfit
    confed_use_friendly=True,   # friendlies DO move the offset. Importance weighting already damps them
                                #   3:1 vs World Cup games; EXCLUDING them only widened the CONMEBOL
                                #   lead (Copa America) for zero accuracy gain, so we keep them.
)

MOM_WINDOW = 10        # momentum = Elo change over a team's last N appearances
REST_CAP_DAYS = 14     # cap rest, so a months-long layoff before game 1 doesn't dominate


def importance_weight(tournament: str) -> float:
    """K-factor multiplier by match importance. Friendlies are noisy -> down-weight."""
    t = str(tournament).lower()
    if "world cup" in t and "qualification" not in t:
        return 3.0   # World Cup finals
    if "qualification" in t:
        return 2.0   # any qualifiers
    if any(x in t for x in ["euro", "copa", "african cup", "asian cup", "gold cup"]):
        return 2.5   # major continental finals
    if "nations league" in t:
        return 2.0
    if "friendly" in t:
        return 1.0
    return 1.5       # other competitive


def expected_home(r_home, r_away, home_adv, scale):
    """Elo expected score for the home side (win=1, draw=0.5, loss=0)."""
    return 1.0 / (1.0 + 10 ** (-((r_home + home_adv) - r_away) / scale))


def mov_multiplier(goal_diff: int) -> float:
    """Scale K up for bigger wins (eloratings.net-style)."""
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8.0


def compute_elo(df, params=DEFAULT_PARAMS, mom_window=MOM_WINDOW, rest_cap=REST_CAP_DAYS):
    """One causal pass. Adds the pre-match columns
        home_elo, away_elo, home_mom, away_mom, home_rest, away_rest
    and returns (df_with_columns, final_ratings, confed_offsets).

    Two-level option (params["confed_elo"]): alongside each team rating we maintain a
    per-CONFEDERATION offset that updates ONLY on cross-confederation games and is shared
    by every team in that confederation. A team's EFFECTIVE rating (what we record and what
    drives the expectation) is team + confed_offset, so:
      * within-confed games are unchanged — both sides carry the same offset, it cancels;
      * cross-confed games lift/lower whole confederations, pooling the sparse inter-continental
        signal that no single team plays enough of. The offset self-corrects toward the value
        that zeroes each confederation's mean cross-confed residual. Importance-weighted, so
        World Cup results move it 3x a friendly. `final_ratings` already includes the offset.

    `df` must already be sorted by date.
    """
    sr = params["start_rating"]
    use_confed = params.get("confed_elo", False)
    kc = params.get("k_confed", 0.0)
    use_fr = params.get("confed_use_friendly", False)
    ratings = {}                 # team -> raw team rating
    conf = defaultdict(float)    # confederation -> offset (moved by cross-confed games only)
    hist = defaultdict(list)     # team -> chronological list of pre-match EFFECTIVE ratings
    last_played = {}             # team -> date of previous match (for rest)
    cols = {c: [] for c in
            ["home_elo", "away_elo", "home_mom", "away_mom", "home_rest", "away_rest"]}

    def confed_off(team):
        c = confed.confed_of(team)
        return c, (conf[c] if (use_confed and c != "OTHER") else 0.0)

    for row in df.itertuples(index=False):
        h, a = row.home_team, row.away_team
        rh = ratings.get(h, sr)
        ra = ratings.get(a, sr)
        ch, oh = confed_off(h)
        ca, oa = confed_off(a)
        eff_h, eff_a = rh + oh, ra + oa       # effective = team rating + confederation offset

        # --- record PRE-match features (effective ratings keep it leak-free) ---
        cols["home_elo"].append(eff_h)
        cols["away_elo"].append(eff_a)

        hh, ah = hist[h], hist[a]
        cols["home_mom"].append(eff_h - hh[-mom_window] if len(hh) >= mom_window else 0.0)
        cols["away_mom"].append(eff_a - ah[-mom_window] if len(ah) >= mom_window else 0.0)
        hh.append(eff_h)
        ah.append(eff_a)

        cols["home_rest"].append(min((row.date - last_played[h]).days, rest_cap) if h in last_played else rest_cap)
        cols["away_rest"].append(min((row.date - last_played[a]).days, rest_cap) if a in last_played else rest_cap)
        last_played[h] = row.date
        last_played[a] = row.date

        # --- update ratings (skip unplayed/future fixtures: no score yet) ---
        if pd.isna(row.home_score) or pd.isna(row.away_score):
            continue

        ha = 0.0 if bool(row.neutral) else params["home_advantage"]
        e_home = expected_home(eff_h, eff_a, ha, params["scale"])
        if row.home_score > row.away_score:
            s_home = 1.0
        elif row.home_score < row.away_score:
            s_home = 0.0
        else:
            s_home = 0.5

        mult = importance_weight(row.tournament)
        if params["use_mov"]:
            mult *= mov_multiplier(row.home_score - row.away_score)
        resid = s_home - e_home

        delta = params["k_base"] * mult * resid     # zero-sum team update (vs the effective expectation)
        ratings[h] = rh + delta
        ratings[a] = ra - delta

        # confederation offset: cross-confed COMPETITIVE games (friendlies excluded by default ->
        # the neutral/high-stakes games most like a World Cup), real confeds, shared & zero-sum
        if (use_confed and kc and ch != ca and ch != "OTHER" and ca != "OTHER"
                and (use_fr or "friendly" not in str(row.tournament).lower())):
            dc = kc * mult * resid
            conf[ch] += dc
            conf[ca] -= dc

    out = df.copy()
    for c, v in cols.items():
        out[c] = v
    eff_final = {t: ratings[t] + (conf[confed.confed_of(t)]
                                  if (use_confed and confed.confed_of(t) != "OTHER") else 0.0)
                 for t in ratings}
    return out, eff_final, dict(conf)


def build_features(df, params=DEFAULT_PARAMS):
    """Run the Elo pass and assemble the V1 model-ready feature table.

    Option A: the model gets the PURE rating gap + a home/neutral flag and learns the
    home effect itself; the home boost is never baked into a feature.
    Returns (feature_df, final_ratings).
    """
    df = df.sort_values("date").reset_index(drop=True).copy()
    # robust boolean for the neutral flag (handles bool or "TRUE"/"FALSE")
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE")

    df, final_ratings, confed_offsets = compute_elo(df, params)
    df.attrs["confed_offsets"] = confed_offsets

    # ---- label (only where the match has actually been played) ----
    res = np.select(
        [df.home_score > df.away_score, df.home_score < df.away_score],
        ["home_win", "away_win"], default="draw",
    )
    played = df.home_score.notna() & df.away_score.notna()
    df["result"] = pd.Series(res, index=df.index).where(played)   # future fixtures -> NaN

    # ---- V1 model features (all PRE-match, leak-free) ----
    df["home_adv_flag"] = (~df["neutral"]).astype(int)   # 2: is anyone truly at home?
    df["rating_gap"]    = df.home_elo - df.away_elo        # 1: pure strength gap (NO home boost)
    df["abs_gap"]       = df.rating_gap.abs()              # 3: closeness -> draw propensity
    df["mom_diff"]      = df.home_mom - df.away_mom        # 4: recent over-performance vs own Elo
    df["rest_diff"]     = df.home_rest - df.away_rest      # 6: freshness gap (capped)

    # ---- baseline ONLY (not a model feature): Elo's own home-win prob, home boost included ----
    ha = np.where(df.neutral, 0.0, params["home_advantage"])
    df["elo_p_home"] = 1.0 / (1.0 + 10 ** (-((df.home_elo + ha) - df.away_elo) / params["scale"]))

    return df, final_ratings


# the columns we actually feed the model in V1
V1_FEATURES = ["rating_gap", "home_adv_flag", "abs_gap", "mom_diff", "rest_diff"]


if __name__ == "__main__":
    # ---- quick self-test / sanity checks ----
    df = pd.read_csv("results.csv", parse_dates=["date"])
    feat, final = build_features(df)

    print("=" * 64)
    print("Top 10 current Elo ratings (should resemble the FIFA top):")
    for team, r in sorted(final.items(), key=lambda kv: -kv[1])[:10]:
        print(f"   {team:<16} {r:6.0f}")

    train = feat[feat.date >= "2010-01-01"]
    played_rows = train[train.result.notna()]
    print("\n" + "=" * 64)
    print(f"Training rows (>=2010): {len(train)} | played: {len(played_rows)} | future: {len(train) - len(played_rows)}")
    print("V1 features:", V1_FEATURES)

    show = ["date", "home_team", "away_team", *V1_FEATURES, "elo_p_home", "result"]
    print("\nSample of recent PLAYED matches (2024+):")
    print(played_rows[played_rows.date >= "2024-01-01"][show].head(6).to_string(index=False))

    print("\nA few 2026 World Cup fixtures (features present, result = NaN -> leak-free):")
    wc = feat[(feat.tournament == "FIFA World Cup") & (feat.date.dt.year == 2026)]
    print(wc[show].head(4).to_string(index=False))

    print("\nNaN check in V1 features (played training rows) -- expect all 0:")
    print(played_rows[V1_FEATURES].isna().sum().to_string())
