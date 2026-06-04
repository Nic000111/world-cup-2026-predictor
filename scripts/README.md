# Scripts

Standalone analysis, tuning, and diagnostic scripts. Each one bootstraps the project root onto `sys.path` so you can run them from anywhere (`python scripts/<name>.py` or `cd scripts && python <name>.py`).

## Active

| script | what it does |
|---|---|
| `glicko.py` | **The head-to-head that picked the shipped engine.** Builds Glicko (rating + uncertainty), coordinate-descent-tunes every parameter, forward-selects features, and beats Elo on held-out log-loss. Why `glicko_engine.py` is the engine. |
| `tune_elo.py` | Sweeps `k_base` × `home_advantage` × MoV for the *legacy* Elo engine (the Glicko baseline). |
| `tune_confed.py` | Sweeps `k_confed` (confederation learning rate). The tuning that produced the cross-continental fix. |
| `final_test.py` | Single-shot evaluation on the held-out 2024+ test set. Run this once after every methodology change to update the "shipped" numbers. |
| `calibration.py` | Reliability diagram + ECE for the W/D/L model; writes `docs/calibration.png` (the plot in the main README). Evidence that the probabilities are honest, not a calibration fix. |
| `group_predictions.py` | Generates the WC group-stage predictions CSV + readable text version. Saved to `~/Downloads/`. |
| `market_comparison.py` | De-vigs bookmaker odds, runs the full Monte-Carlo, compares our title odds vs the market team-by-team and confederation-by-confederation. |
| `diag.py` | Quick Elo sanity check — top ratings, training-row counts, label balance. |
| `diag_confed_bias.py` | Computes per-confederation cross-confed residuals (the diagnostic that proved we needed the confederation offset). |
| `diagnostics.py` | Three model sanity checks in one: draw behaviour, headline-score↔W/D/L consistency (0/3540 disagreements), and the joint-mode-vs-outcome illustration (why the likeliest *score* can be a draw while the likeliest *outcome* is a win). |

## Historical / rejected experiments

Kept for transparency — these are the experiments that didn't make it into the shipped model.

| script | what it found |
|---|---|
| `validate_boost.py` | Cross-confed K-boost: shrinks the residual but log-loss change is within noise. **Rejected** — information limit, not a weighting bug. |
| `validate_confed.py` | Earlier confederation validation, before the two-level Elo design. |
| `simulate_wc.py` | Earlier WC simulation work (superseded by `wc.simulate_tournament`). |
| `train_models.py` | Initial model-training script with GridSearchCV (superseded by `wc.py` baking in the chosen pipelines). |
| `goals_model.py` | Initial goals-model exploration (now lives inside `wc.py` as the Elo-Poisson + Dixon-Coles fit). |
