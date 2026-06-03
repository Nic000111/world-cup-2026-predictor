"""
train_models.py - V1 model bake-off for the World Cup 2026 predictor.

Flow:
  time split (no random CV) -> per-model sklearn Pipeline
  -> hyperparameter GridSearchCV on a single PredefinedSplit (train->val)
  -> honest test evaluation
  -> decision-threshold grid search (recovers under-predicted draws)
  -> calibration check on the winner.

Metrics: log-loss (selection target), RPS (football-native, ordinal), accuracy (reported).
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import elo
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)   # only mute convergence chatter, nothing else
CLASSES = ["home_win", "draw", "away_win"]   # ordinal order for RPS (home - draw - away)


# ---------------- metrics ----------------
def ordered_proba(model, X):
    """predict_proba, columns reordered to CLASSES."""
    p = model.predict_proba(X)
    idx = [list(model.classes_).index(c) for c in CLASSES]
    return p[:, idx]


def rps_score(y_true, proba):
    """Ranked Probability Score (lower = better). proba aligned to CLASSES."""
    pos = {c: i for i, c in enumerate(CLASSES)}
    onehot = np.zeros_like(proba)
    for r, lab in enumerate(y_true):
        onehot[r, pos[lab]] = 1.0
    cp, ct = np.cumsum(proba, axis=1), np.cumsum(onehot, axis=1)
    return float(np.mean(((cp - ct) ** 2).sum(axis=1) / (len(CLASSES) - 1)))


def evaluate(model, X, y):
    p = ordered_proba(model, X)                    # columns in CLASSES order
    pred = np.array(CLASSES)[p.argmax(1)]
    # log-loss computed straight from the true-class probability -> immune to any label-order quirk
    pos = np.array([CLASSES.index(t) for t in y])
    p_true = np.clip(p[np.arange(len(y)), pos], 1e-15, 1.0)
    return dict(log_loss=float(-np.mean(np.log(p_true))),
                rps=rps_score(y, p),
                accuracy=accuracy_score(y, pred))


def draw_recall(y, pred):
    m = (y == "draw")
    return float((pred[m] == "draw").mean()) if m.sum() else 0.0


# ---------------- data + time split ----------------
feat, _ = elo.build_features(pd.read_csv("results.csv", parse_dates=["date"]))
d = feat[feat.result.notna() & (feat.date >= "2010-01-01")].copy().reset_index(drop=True)
year = d.date.dt.year
masks = {
    "train": (year <= 2021).values,
    "val":   ((year >= 2022) & (year <= 2023)).values,
    "test":  (year >= 2024).values,
}
FULL = elo.V1_FEATURES                  # 5 features
ELO = ["rating_gap", "home_adv_flag"]   # Elo-only bar


def XY(feats, which):
    m = masks[which]
    return d.loc[m, feats], d.loc[m, "result"].values


print("Split sizes:", {k: int(v.sum()) for k, v in masks.items()})
for k in masks:
    yy = d.loc[masks[k], "result"]
    print(f"  {k:5} outcome%: {(yy.value_counts(normalize=True) * 100).round(1).to_dict()}")


# ---------------- baseline: pick the higher Elo (accuracy only, never draws) ----------------
print("\n--- baseline: pick-higher-Elo ---")
for which in ["val", "test"]:
    m = masks[which]
    pred = np.where(d.loc[m, "elo_p_home"].values >= 0.5, "home_win", "away_win")
    print(f"  {which} accuracy: {accuracy_score(d.loc[m, 'result'].values, pred):.3f}")


# ---------------- model grid search on the single train->val fold ----------------
GS, FEATS = {}, {}


def run(name, pipe, grid, feats):
    Xtr, ytr = XY(feats, "train")
    Xva, yva = XY(feats, "val")
    Xtv = pd.concat([Xtr, Xva]).reset_index(drop=True)
    ytv = np.concatenate([ytr, yva])
    fold = np.r_[np.full(len(Xtr), -1), np.zeros(len(Xva), int)]   # -1=train always, 0=val fold
    gs = GridSearchCV(pipe, grid, cv=PredefinedSplit(fold),
                      scoring="neg_log_loss", refit=True, n_jobs=-1)
    gs.fit(Xtv, ytv)
    Xte, yte = XY(feats, "test")
    res = evaluate(gs.best_estimator_, Xte, yte)
    res.update(name=name, val_logloss=-gs.best_score_)
    GS[name], FEATS[name] = gs, feats
    print(f"  {name:24} best params: {gs.best_params_}")
    return res


def lin(C_grid):
    return (Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))]),
            {"clf__C": C_grid, "clf__class_weight": [None, "balanced"]})

print("\n--- grid search (selecting on val log-loss) ---")
results = []
p, g = lin([0.01, 0.1, 1, 10, 100]);            results.append(run("logistic (Elo-only)", p, g, ELO))
p, g = lin([1e-3, 1e-2, 0.1, 1, 10, 100]);      results.append(run("logistic (full)", p, g, FULL))

results.append(run("hist gradient boosting",
    Pipeline([("clf", HistGradientBoostingClassifier(random_state=0))]),
    {"clf__learning_rate": [0.05, 0.1, 0.2], "clf__max_iter": [300, 800],
     "clf__max_leaf_nodes": [15, 31, 63], "clf__min_samples_leaf": [20, 100],
     "clf__l2_regularization": [0.0, 1.0], "clf__class_weight": [None, "balanced"]}, FULL))

results.append(run("random forest",
    Pipeline([("clf", RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=1))]),
    {"clf__max_depth": [None, 8, 16], "clf__min_samples_leaf": [5, 20, 50],
     "clf__max_features": ["sqrt", None], "clf__class_weight": [None, "balanced"]}, FULL))


# ---------------- comparison table ----------------
tbl = pd.DataFrame(results)[["name", "val_logloss", "log_loss", "rps", "accuracy"]]
tbl.columns = ["model", "val_logloss", "test_logloss", "test_rps", "test_acc"]
print("\n" + "=" * 70)
print("MODEL COMPARISON (sorted by val log-loss):")
print(tbl.sort_values("val_logloss").round(4).to_string(index=False))


# ---------------- threshold grid search on the val-honest winner ----------------
best_name = min(results, key=lambda r: r["val_logloss"])["name"]
feats = FEATS[best_name]
Xtr, ytr = XY(feats, "train"); Xva, yva = XY(feats, "val"); Xte, yte = XY(feats, "test")
m_train = clone(GS[best_name].best_estimator_).fit(Xtr, ytr)   # train-only -> honest val probs
pva = ordered_proba(m_train, Xva)

best_acc, best_w = -1.0, (1.0, 1.0, 1.0)
for wd in np.round(np.arange(1.0, 2.55, 0.1), 2):
    for wa in np.round(np.arange(0.7, 1.55, 0.1), 2):
        w = np.array([1.0, wd, wa])
        a = accuracy_score(yva, np.array(CLASSES)[(pva * w).argmax(1)])
        if a > best_acc:
            best_acc, best_w = a, (1.0, float(wd), float(wa))

pte = ordered_proba(GS[best_name].best_estimator_, Xte)   # train+val model on test
pred_argmax = np.array(CLASSES)[pte.argmax(1)]
pred_tuned = np.array(CLASSES)[(pte * np.array(best_w)).argmax(1)]
print("\n" + "=" * 70)
print(f"THRESHOLD SEARCH on winner: {best_name}")
print(f"  weights (home,draw,away) = {best_w}   (val acc {best_acc:.3f})")
print(f"  TEST accuracy   argmax {accuracy_score(yte, pred_argmax):.3f}  ->  tuned {accuracy_score(yte, pred_tuned):.3f}")
print(f"  TEST draw recall argmax {draw_recall(yte, pred_argmax):.3f}  ->  tuned {draw_recall(yte, pred_tuned):.3f}")


# ---------------- calibration check on the winner (home-win probability) ----------------
phome = pte[:, CLASSES.index("home_win")]
cal = pd.DataFrame({"p_home": phome, "is_home": (yte == "home_win").astype(int)})
cal["bin"] = pd.cut(cal.p_home, np.linspace(0, 1, 11))
out = cal.groupby("bin", observed=True).agg(n=("is_home", "size"),
                                            predicted=("p_home", "mean"),
                                            actual=("is_home", "mean"))
print("\n" + "=" * 70)
print(f"CALIBRATION of {best_name} (home-win prob: predicted vs actual):")
print(out.round(3).to_string())
