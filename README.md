# World Cup 2026 — Match Outcome Predictor

A calibrated, leak-free model that forecasts every 2026 World Cup match — and the entire tournament — from international results since 1872, with a self-correcting cross-continental adjustment that learns from the data alone.

**Live dashboard:** [nic-world-cup-2026.streamlit.app](https://nic-world-cup-2026.streamlit.app)

---

## What it is

A pure-results predictor for the 48-team 2026 World Cup. No transfer values, no FIFA rankings, no betting odds in the loop — just every international result, fed through a self-computed Elo with an honest train / validate / test split.

The dashboard exposes six tabs:

- **Predict a match** — any two teams, neutral or home, W/D/L probabilities + expected goals + most-likely score
- **Group forecast** — all 72 group-stage fixtures with probabilities and likely scorelines
- **Ratings** — current Elo for every national team, plus the per-confederation adjustment
- **Enter result** — record actual scores as they happen; the Elo absorbs them and every prediction updates (password-gated on the deployed app)
- **Title odds** — 20,000-simulation Monte-Carlo of the whole tournament with rating uncertainty
- **Bracket** — the single most-likely knockout path, R16 → final

---

## How it works

### 1. Self-computed Elo from raw results

One causal pass over every international match ever played. The rating each team carries *into* a match is what gets recorded — by construction, no result can leak into its own features. Tunables: `k_base = 20`, `home_advantage = 70` (zeroed for neutral venues), importance-weighted (World Cup 3×, qualifiers 2×, friendlies 1×), margin-of-victory multiplier.

### 2. Per-confederation offset (the cross-continental fix)

Only ~14% of international games cross confederations, and each top European team plays barely 10 of them in 8 years. That's not enough data to pin each confederation's strength against the others — ordinary Elo silently compresses the gap between UEFA / CONMEBOL and the weaker confederations.

The fix: maintain a per-confederation rating alongside the team ratings. Cross-continental games update *both* the team Elo and a shared confederation offset. A team's effective rating = `team + offset[its confederation]`. Within-confederation games leave it alone (the offset cancels). The offset learns from pooled signal — the 758 CONMEBOL cross-confed games, not the 11 Spain plays.

Tuned by held-out cross-confed log-loss; deliberately set below the raw loss-minimum to keep offsets football-plausible rather than overfit to tiny confederations.

### 3. Two complementary models on the same Elo features

- **Elo-logistic** — logistic regression on rating gap, home flag, absolute gap, momentum, rest. The W/D/L probabilities.
- **Elo-Poisson + Dixon-Coles** — a Poisson goals model on the same Elo features, with the Dixon-Coles low-score correction (rho fit by MLE). Produces both 1X2 probabilities and full scoreline distributions.

### 4. Full tournament Monte-Carlo

20,000 simulations of the 48-team format (12 groups, top 2 + 8 best thirds, R32 → R16 → QF → SF → final). Each simulation perturbs every team's Elo by N(0, 125) to model rating uncertainty — so favourites don't over-concentrate at the top. Group-stage tiebreakers, third-place slot assignment, and knockout penalty-shootout fallbacks all match the real tournament rules.

---

## Validation

Time-based splits, no peeking — train < 2022, validate 2022–23, test 2024–25.

| Metric | All games | Cross-continental |
|---|---|---|
| Log-loss (held-out test) | **0.860** | **0.896** |
| Accuracy | **60.4%** | **59.2%** |

The cross-continental log-loss improved ~3% after shipping the confederation offset (was 0.926). The structural ceiling for results-only models on international football is around 0.86 log-loss and 60% accuracy — draws (~24% of games) are rarely the modal pick, so a hard floor of mispredicted games is unavoidable in any model.

### Against the sharp betting market

| | ours | de-vigged market |
|---|---|---|
| Correlation across all 48 teams | **0.90** | — |
| Mean absolute divergence | **0.9 percentage points** | — |
| Continental bias (UEFA, sum of diffs) | **−6.5 pts** (was −12.5 pre-confed-fix) | 0 |

The continental tilt against bookmakers roughly halved after the confederation offset. The residual UEFA gap is the **squad-depth premium** — the market knows Europe's bench is deeper than their starting XI suggests, which a results-only model structurally cannot see (we tried squad market value; it failed a leak-free test).

### What we believe vs the market

- **Higher than market on recent winners** — Spain (+6.2 pts), Argentina (+3.7), Croatia/Ecuador/Colombia
- **Lower than market on name-brands with weak form** — Brazil (−5.1), Portugal (−3.8), England (−3.8), France (−3.8)

That's the honest signature of a results-only model: it rewards what's been done, not who's traditionally good.

---

## Running locally

```bash
git clone https://github.com/Nic000111/world-cup-2026-predictor.git
cd world-cup-2026-predictor
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run dashboard.py
```

Dashboard opens at `http://localhost:8501`. No password gate locally — the `ENTER_PWD` secret is only set in the deployed app.

### Reproducing the analysis

Each script in `scripts/` is standalone. Run from project root:

```bash
python scripts/market_comparison.py    # bookmaker comparison + continental tilt
python scripts/tune_confed.py          # the k_confed sweep that fixed the cross-confed gap
python scripts/tune_elo.py             # k_base and home_advantage tuning
python scripts/final_test.py           # held-out test set evaluation
python scripts/group_predictions.py    # generates the WC predictions CSV
```

---

## Project structure

```
.
├── dashboard.py            Streamlit app (the live URL)
├── wc.py                   WorldCupModel — Elo + logistic + Poisson + Monte-Carlo
├── elo.py                  Two-level Elo engine (team + confederation)
├── confed.py               Country → confederation lookup
├── results.csv             ~49k international results (1872 – present, martj42)
├── requirements.txt
├── scripts/                Analysis & tuning scripts (see scripts/README.md)
└── notebooks/              Early exploration notebook
```

---

## Methodology highlights

- **Time-based splits** — random k-folds would leak future information into past predictions. Validate 2022–23, hold out 2024–25, never peek.
- **Honest experiment register** — we tried cross-confed K-boost (failed), squad market value as Elo anchor (failed a leak-control), and confederation offset (passed every robustness check). Only what survived is shipped.
- **No magic numbers** — every hyperparameter (`k_base`, `home_advantage`, `k_confed`) is tuned by held-out log-loss with the search documented in `scripts/`.
- **No copying the market** — when we discovered our model under-rates UEFA vs bookmakers by ~13 percentage points, the goal was *not* to match the market but to find the structural fix in our own data. The per-confederation offset closed about half of that gap from results alone.

---

## Data

[martj42 / international_results](https://github.com/martj42/international_results) — every international football result from 1872 to today (~49k matches), updated continuously by the maintainer. Re-pull anytime to refresh the Elo with the latest games.

## Acknowledgments

- Elo formulation inspired by [eloratings.net](https://www.eloratings.net/) (MoV multiplier shape)
- Dixon-Coles low-score correction: Dixon & Coles, *Modelling Association Football Scores and Inefficiencies in the Football Betting Market* (1997)
- Streamlit & scikit-learn for the application stack

---

## Limitations

A results-only model genuinely cannot see:
- Squad depth (Europe's 20th-best player vs CONMEBOL's)
- Injuries, lineup rotations between rating-update and match
- In-tournament fatigue, travel, weather
- Knockout-stage variance — single matches at the margin are coin-flips

These are baked into bookmaker prices but not into ours. That's not a bug to fix; it's a feature of being a transparent, interpretable, results-honest model. When we disagree with the market, the disagreements are interpretable — and that's the whole point.
