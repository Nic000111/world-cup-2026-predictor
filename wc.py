"""
wc.py - consolidated 2026 World Cup prediction engine.

Builds the Elo ratings once and fits both models:
  - Elo-logistic            -> 1X2 (home/draw/away win) probabilities
  - Elo-Poisson + Dixon-Coles -> goals: 1X2 + expected goals + full scoreline distribution

Exposes predict_match(home, away, neutral) for ANY two teams. Backs the Streamlit dashboard.
"""
import itertools
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import confed
import glicko_engine as engine     # uncertainty-aware (Glicko) ratings — replaced Elo (beat it on held-out log-loss)

CLASSES = ["home_win", "draw", "away_win"]
MAXG = 10
GFEATS = ["scorer_elo", "opp_elo", "is_home", "is_friendly"]
_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS = os.path.join(_DIR, "results.csv")
MANUAL_RESULTS = os.path.join(_DIR, "manual_results.csv")
MANUAL_COLS = ["date", "home_team", "away_team", "home_score", "away_score", "tournament", "city", "country", "neutral"]


def load_results(results_path=DEFAULT_RESULTS):
    """results.csv merged with manually-entered results. A manual result fills a matching
    unplayed fixture if one exists (so it both updates Elo and leaves the 'upcoming' list);
    otherwise it is appended as a brand-new match."""
    df = pd.read_csv(results_path, parse_dates=["date"])
    if os.path.exists(MANUAL_RESULTS):
        man = (pd.read_csv(MANUAL_RESULTS, parse_dates=["date"])
               .dropna(subset=["home_score", "away_score"])
               .drop_duplicates(subset=["home_team", "away_team", "date"], keep="last"))
        extra = []
        for _, m in man.iterrows():
            hit = (df.home_team == m.home_team) & (df.away_team == m.away_team) & df.home_score.isna()
            if hit.any():
                i = df.index[hit][0]
                df.loc[i, "home_score"], df.loc[i, "away_score"] = m.home_score, m.away_score
            else:
                extra.append(m)
        if extra:
            df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
    return df.sort_values("date").reset_index(drop=True)


def record_result(home, away, home_score, away_score, date, tournament="FIFA World Cup", neutral=True):
    """Append a manually-entered result to manual_results.csv."""
    row = {"date": pd.to_datetime(date), "home_team": home, "away_team": away,
           "home_score": int(home_score), "away_score": int(away_score),
           "tournament": tournament, "city": "", "country": "", "neutral": bool(neutral)}
    man = pd.read_csv(MANUAL_RESULTS) if os.path.exists(MANUAL_RESULTS) else pd.DataFrame(columns=MANUAL_COLS)
    pd.concat([man, pd.DataFrame([row])], ignore_index=True).to_csv(MANUAL_RESULTS, index=False)


class WorldCupModel:
    def __init__(self, results_path=DEFAULT_RESULTS):
        feat, ratings = engine.build_features(load_results(results_path))
        self.feat = feat
        self.team_elo = dict(ratings)
        self.team_rd = dict(feat.attrs.get("final_rd", {}))          # per-team uncertainty (Glicko RD)
        pl = feat[feat.result.notna() & (feat.date >= "2010-01-01")].copy()
        pl["is_friendly"] = pl.tournament.str.contains("friendly", case=False).astype(float)

        # model 1: Glicko-logistic (1X2)
        self.F = engine.V1_FEATURES
        self.lr = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(C=0.01, max_iter=2000))]).fit(pl[self.F], pl["result"])

        # model 2: Elo-Poisson + Dixon-Coles (goals)
        stk = self._stack(pl)
        self.gm = Pipeline([("sc", StandardScaler()), ("po", PoissonRegressor(alpha=1e-4, max_iter=5000))]).fit(stk[GFEATS], stk["goals"])
        lh = self._mu(pl.home_elo, pl.away_elo, np.where(pl.neutral, 0., 1.), pl.is_friendly)
        la = self._mu(pl.away_elo, pl.home_elo, np.zeros(len(pl)), pl.is_friendly)
        xt, yt = pl.home_score.values.astype(int), pl.away_score.values.astype(int)
        self.rho = float(minimize_scalar(lambda r: -np.sum(np.log(np.clip(self._tau(xt, yt, lh, la, r), 1e-9, None))),
                                         bounds=(-0.2, 0.2), method="bounded").x)

        self.teams = sorted(self.team_elo)
        self.confed_offsets = dict(feat.attrs.get("confed_offsets", {}))   # per-confederation cross-continental offset

    # ---- internals ----
    def _stack(self, r):
        h = pd.DataFrame(dict(goals=r.home_score.values, scorer_elo=r.home_elo.values, opp_elo=r.away_elo.values,
                              is_home=np.where(r.neutral, 0., 1.), is_friendly=r.is_friendly.values))
        a = pd.DataFrame(dict(goals=r.away_score.values, scorer_elo=r.away_elo.values, opp_elo=r.home_elo.values,
                              is_home=0., is_friendly=r.is_friendly.values))
        return pd.concat([h, a], ignore_index=True)

    def _mu(self, scorer, opp, is_home, is_friendly):
        X = pd.DataFrame({"scorer_elo": np.asarray(scorer, float), "opp_elo": np.asarray(opp, float),
                          "is_home": np.asarray(is_home, float), "is_friendly": np.asarray(is_friendly, float)})
        return self.gm.predict(X[GFEATS])

    @staticmethod
    def _tau(x, y, lh, la, r):
        t = np.ones(len(x))
        m = (x == 0) & (y == 0); t[m] = 1 - lh[m] * la[m] * r
        m = (x == 0) & (y == 1); t[m] = 1 + lh[m] * r
        m = (x == 1) & (y == 0); t[m] = 1 + la[m] * r
        m = (x == 1) & (y == 1); t[m] = 1 - r
        return t

    def _matrix(self, lh, la):
        g = np.arange(MAXG + 1)
        M = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
        M[0, 0] *= 1 - lh * la * self.rho; M[0, 1] *= 1 + lh * self.rho
        M[1, 0] *= 1 + la * self.rho; M[1, 1] *= 1 - self.rho
        return np.clip(M, 0, None) / np.clip(M, 0, None).sum()

    # ---- public API ----
    def predict_match(self, home, away, neutral=True):
        eh, ea = self.team_elo[home], self.team_elo[away]
        gap = eh - ea
        rd0 = engine.DEFAULT_PARAMS["rd_init"]
        feats = pd.DataFrame([{"rating_gap": gap, "home_adv_flag": int(not neutral), "abs_gap": abs(gap),
                               "rd_sum": self.team_rd.get(home, rd0) + self.team_rd.get(away, rd0)}])[self.F]
        lp = self.lr.predict_proba(feats)[0][[list(self.lr.classes_).index(c) for c in CLASSES]]
        lh = float(self._mu([eh], [ea], [0. if neutral else 1.], [0.])[0])
        la = float(self._mu([ea], [eh], [0.], [0.])[0])
        M = self._matrix(lh, la)
        gh, gd, ga = float(np.tril(M, -1).sum()), float(np.trace(M)), float(np.triu(M, 1).sum())
        # Headline scoreline = most-likely score WITHIN the most-likely outcome, so it can never
        # contradict the win/draw/loss pick (we never show a 1-1 next to a home-favoured game — we
        # show the favourite's most-likely *winning* score). Rows = home goals, cols = away goals.
        ii, jj = np.indices(M.shape)
        region = [ii > jj, ii == jj, ii < jj][int(np.argmax([gh, gd, ga]))]
        si, sj = np.unravel_index(np.where(region, M, -1.0).argmax(), M.shape)
        flat = M.ravel()
        top = [(f"{i}-{j}", float(flat[t])) for t in np.argsort(flat)[::-1][:6] for (i, j) in [np.unravel_index(t, M.shape)]]
        under25 = float(M[(ii + jj) <= 2].sum())          # P(total goals <= 2)  -> Under 2.5
        btts_yes = float(M[(ii >= 1) & (jj >= 1)].sum())  # P(both teams score)
        return {"home": home, "away": away, "neutral": neutral, "rating": (float(eh), float(ea)),
                "rd": (float(self.team_rd.get(home, rd0)), float(self.team_rd.get(away, rd0))),
                "logistic": {"home": float(lp[0]), "draw": float(lp[1]), "away": float(lp[2])},
                "goals": {"home": gh, "draw": gd, "away": ga}, "xg": (lh, la),
                "markets": {"under25": under25, "btts_yes": btts_yes},
                "likely_score": f"{si}-{sj}", "top_scores": top, "matrix": M}

    def group_fixtures(self):
        wc = self.feat[(self.feat.tournament == "FIFA World Cup") & (self.feat.date.dt.year == 2026) & self.feat.result.isna()]
        rows = []
        for _, r in wc.iterrows():
            p = self.predict_match(r.home_team, r.away_team, neutral=bool(r.neutral))
            rows.append(dict(date=r.date.strftime("%Y-%m-%d"), home=r.home_team, away=r.away_team,
                             home_win=p["goals"]["home"], draw=p["goals"]["draw"], away_win=p["goals"]["away"],
                             xg_home=p["xg"][0], xg_away=p["xg"][1],
                             under25=p["markets"]["under25"], btts_yes=p["markets"]["btts_yes"],
                             likely_score=p["likely_score"]))
        return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    def ratings_table(self):
        rd0 = engine.DEFAULT_PARAMS["rd_init"]
        df = pd.DataFrame({"team": list(self.team_elo),
                           "confederation": [confed.confed_of(t) for t in self.team_elo],
                           "rating": [round(v) for v in self.team_elo.values()],
                           "uncertainty": [round(self.team_rd.get(t, rd0)) for t in self.team_elo]})
        df = df[df.uncertainty <= 85]      # drop barely-tracked non-national sides (high RD) from the display
        return df.sort_values("rating", ascending=False).reset_index(drop=True)

    # ---- tournament Monte-Carlo (with rating uncertainty) ----
    def _adv(self, la, lb):
        """Neutral knockout P(team a advances) from expected goals la,lb (DC + penalty split)."""
        g = np.arange(MAXG + 1)
        M = np.outer(poisson.pmf(g, la), poisson.pmf(g, lb))
        M[0, 0] *= 1 - la * lb * self.rho; M[0, 1] *= 1 + la * self.rho
        M[1, 0] *= 1 + lb * self.rho; M[1, 1] *= 1 - self.rho
        M = np.clip(M, 0, None); h, d, a = np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()
        tot = h + d + a; h, a = h / tot, a / tot
        return h + (1 - h - a) * (h / (h + a)) if (h + a) > 0 else 0.5

    def simulate_tournament(self, n_sim=20000, rd_scale=1.5, seed=0):
        """Monte-Carlo the 2026 World Cup. Each run perturbs every team's rating by its OWN Glicko
        uncertainty N(0, rd_scale * team_RD) -- so genuinely data-poor / volatile teams spread out
        more than well-known ones (real per-team uncertainty, not a flat fudge). Returns per-team
        probabilities of winning the group / advancing / reaching each round / lifting the cup."""
        rng = np.random.default_rng(seed)
        sc, po = self.gm.named_steps["sc"], self.gm.named_steps["po"]
        mu, sg, cf, b0 = sc.mean_, sc.scale_, po.coef_, po.intercept_

        def lam(se, oe, ih):  # vectorized expected goals (is_friendly = 0)
            z = cf[0] * (se - mu[0]) / sg[0] + cf[1] * (oe - mu[1]) / sg[1] + cf[2] * (ih - mu[2]) / sg[2] + cf[3] * (0.0 - mu[3]) / sg[3]
            return np.exp(b0 + z)

        wc = self.feat[(self.feat.tournament == "FIFA World Cup") & (self.feat.date.dt.year == 2026)]  # all 72 group fixtures (played or not)
        teams = sorted(set(wc.home_team) | set(wc.away_team)); ti = {t: i for i, t in enumerate(teams)}
        base = np.array([self.team_elo[t] for t in teams])
        rd0 = engine.DEFAULT_PARAMS["rd_init"]
        sd = np.array([self.team_rd.get(t, rd0) for t in teams]) * rd_scale   # per-team uncertainty
        adj = defaultdict(set)
        for _, r in wc.iterrows():
            adj[r.home_team].add(r.away_team); adj[r.away_team].add(r.home_team)
        ANCH = {"Mexico": "A", "Canada": "B", "Brazil": "C", "United States": "D", "Germany": "E", "Netherlands": "F",
                "Belgium": "G", "Spain": "H", "France": "I", "Argentina": "J", "Portugal": "K", "England": "L"}
        grp, seen = {}, set()
        for t in teams:
            if t in seen: continue
            g = {t} | adj[t]; grp[next(ANCH[x] for x in g if x in ANCH)] = [ti[x] for x in g]; seen |= g
        LETTERS = sorted(grp)
        fixtures = [(ti[r.home_team], ti[r.away_team], 0.0 if r.neutral else 1.0, r.home_score, r.away_score) for _, r in wc.iterrows()]

        gaps = np.arange(-1400, 1401, 20.0); ref = 1900.0
        adv_tab = np.array([self._adv(lam(ref + gp / 2, ref - gp / 2, 0.0), lam(ref - gp / 2, ref + gp / 2, 0.0)) for gp in gaps])

        SLOTS = [("M74", set("ABCDF")), ("M77", set("CDFGH")), ("M79", set("CEFHI")), ("M80", set("EHIJK")),
                 ("M81", set("BEFIJ")), ("M82", set("AEHIJ")), ("M85", set("EFGIJ")), ("M87", set("DEIJL"))]
        def match3(q):
            asn, used = {}, set()
            def bt(i):
                if i == 8: return True
                sid, al = SLOTS[i]
                for gg in al:
                    if gg in q and gg not in used:
                        used.add(gg); asn[sid] = gg
                        if bt(i + 1): return True
                        used.discard(gg); del asn[sid]
                return False
            return {gg: sid for sid, gg in asn.items()} if bt(0) else None
        THIRD = {frozenset(c): match3(frozenset(c)) for c in itertools.combinations("ABCDEFGHIJKL", 8)}
        R32 = {73: ("2A", "2B"), 74: ("1E", "M74"), 75: ("1F", "2C"), 76: ("1C", "2F"), 77: ("1I", "M77"), 78: ("2E", "2I"),
               79: ("1A", "M79"), 80: ("1L", "M80"), 81: ("1D", "M81"), 82: ("1G", "M82"), 83: ("2K", "2L"), 84: ("1H", "2J"),
               85: ("1B", "M85"), 86: ("1J", "2H"), 87: ("1K", "M87"), 88: ("2D", "2G")}
        R16 = {89: (74, 77), 90: (73, 75), 91: (76, 78), 92: (79, 80), 93: (83, 84), 94: (81, 82), 95: (86, 88), 96: (85, 87)}
        QF = {97: (89, 90), 98: (93, 94), 99: (91, 92), 100: (95, 96)}; SF = {101: (97, 98), 102: (99, 100)}

        pe = base[None, :] + rng.normal(0, 1, (n_sim, 48)) * sd[None, :]
        HG, AG = [], []
        for h, a, ih, hs, as_ in fixtures:
            if pd.notna(hs):                                       # already played -> use the real score
                HG.append([int(hs)] * n_sim); AG.append([int(as_)] * n_sim)
            else:                                                  # not yet played -> simulate it
                HG.append(rng.poisson(lam(pe[:, h], pe[:, a], ih)).tolist())
                AG.append(rng.poisson(lam(pe[:, a], pe[:, h], 0.0)).tolist())
        KO = rng.random((n_sim, 31))
        reach = {k: np.zeros(48, int) for k in ["win_group", "advance", "reach_QF", "reach_SF", "final", "champion"]}
        for s in range(n_sim):
            pts = [0.0] * 48; gd = [0.0] * 48; gf = [0.0] * 48
            for fi, (h, a, ih, hs, as_) in enumerate(fixtures):
                hg, ag = HG[fi][s], AG[fi][s]
                gf[h] += hg; gf[a] += ag; gd[h] += hg - ag; gd[a] += ag - hg
                if hg > ag: pts[h] += 3
                elif hg < ag: pts[a] += 3
                else: pts[h] += 1; pts[a] += 1
            ps = pe[s]; pos = {}
            for L in LETTERS:
                st = sorted(grp[L], key=lambda t: (pts[t], gd[t], gf[t], ps[t]), reverse=True)
                pos[L] = st; reach["win_group"][st[0]] += 1; reach["advance"][st[0]] += 1; reach["advance"][st[1]] += 1
            thirds = sorted(((L, pos[L][2]) for L in LETTERS), key=lambda x: (pts[x[1]], gd[x[1]], gf[x[1]], ps[x[1]]), reverse=True)[:8]
            for _, t in thirds: reach["advance"][t] += 1
            gmap = THIRD[frozenset(L for L, _ in thirds)]; third_team = {gmap[L]: t for L, t in thirds}
            def team_of(c): return pos[c[1]][0] if c[0] == "1" else pos[c[1]][1] if c[0] == "2" else third_team[c]
            win = {}; k = 0
            for m, (c1, c2) in R32.items():
                t1, t2 = team_of(c1), team_of(c2)
                win[m] = t1 if KO[s, k] < np.interp(ps[t1] - ps[t2], gaps, adv_tab) else t2; k += 1
            for stage, nxt in [(R16, "reach_QF"), (QF, "reach_SF"), (SF, "final")]:
                for m, (x, y) in stage.items():
                    t1, t2 = win[x], win[y]
                    win[m] = t1 if KO[s, k] < np.interp(ps[t1] - ps[t2], gaps, adv_tab) else t2
                    reach[nxt][win[m]] += 1; k += 1
            t1, t2 = win[101], win[102]
            reach["champion"][t1 if KO[s, k] < np.interp(ps[t1] - ps[t2], gaps, adv_tab) else t2] += 1
        out = pd.DataFrame({"team": teams, "rating": base.round().astype(int)})
        for kk in reach:
            out[kk] = reach[kk] / n_sim
        return out.sort_values("champion", ascending=False).reset_index(drop=True)

    def project_bracket(self):
        """Single most-likely knockout bracket from current ratings + entered results.
        Group order = projected points (actual where played, else expected); knockouts: favourite advances.
        Returns {R32,R16,QF,SF: [(a,b,winner,p)...], final:(a,b,w,p), third:(...), champion:(team,p)}."""
        wc = self.feat[(self.feat.tournament == "FIFA World Cup") & (self.feat.date.dt.year == 2026)]
        teams = sorted(set(wc.home_team) | set(wc.away_team))
        adj = defaultdict(set)
        for _, r in wc.iterrows():
            adj[r.home_team].add(r.away_team); adj[r.away_team].add(r.home_team)
        ANCH = {"Mexico": "A", "Canada": "B", "Brazil": "C", "United States": "D", "Germany": "E", "Netherlands": "F",
                "Belgium": "G", "Spain": "H", "France": "I", "Argentina": "J", "Portugal": "K", "England": "L"}
        grp, seen = {}, set()
        for t in teams:
            if t in seen: continue
            g = {t} | adj[t]; grp[next(ANCH[x] for x in g if x in ANCH)] = list(g); seen |= g
        pts, gd = defaultdict(float), defaultdict(float)
        for _, r in wc.iterrows():
            if pd.notna(r.home_score):
                hp, ap = (3, 0) if r.home_score > r.away_score else (0, 3) if r.home_score < r.away_score else (1, 1)
                pts[r.home_team] += hp; pts[r.away_team] += ap
                gd[r.home_team] += r.home_score - r.away_score; gd[r.away_team] += r.away_score - r.home_score
            else:
                g = self.predict_match(r.home_team, r.away_team, neutral=bool(r.neutral))["goals"]
                pts[r.home_team] += 3 * g["home"] + g["draw"]; pts[r.away_team] += 3 * g["away"] + g["draw"]
        key = lambda t: (pts[t], gd[t], self.team_elo[t])
        pos = {L: sorted(grp[L], key=key, reverse=True) for L in sorted(grp)}
        thirds = sorted([(L, pos[L][2]) for L in pos], key=lambda x: key(x[1]), reverse=True)[:8]
        SLOTS = [("M74", set("ABCDF")), ("M77", set("CDFGH")), ("M79", set("CEFHI")), ("M80", set("EHIJK")),
                 ("M81", set("BEFIJ")), ("M82", set("AEHIJ")), ("M85", set("EFGIJ")), ("M87", set("DEIJL"))]
        qset = {L for L, _ in thirds}; asn, used = {}, set()
        def bt(i):
            if i == 8: return True
            sid, al = SLOTS[i]
            for gg in al:
                if gg in qset and gg not in used:
                    used.add(gg); asn[sid] = gg
                    if bt(i + 1): return True
                    used.discard(gg); del asn[sid]
            return False
        bt(0); tmap = {sid: dict(thirds)[g] for sid, g in asn.items()}
        def slot(c):
            return pos[c[1]][0] if c[0] == "1" else pos[c[1]][1] if c[0] == "2" else tmap[c]
        def winner(a, b):
            g = self.predict_match(a, b, neutral=True)["goals"]; h, aw = g["home"], g["away"]
            pa = h + (1 - h - aw) * (h / (h + aw)) if (h + aw) > 0 else 0.5
            return (a, pa) if pa >= 0.5 else (b, 1 - pa)
        R32 = {73: ("2A", "2B"), 74: ("1E", "M74"), 75: ("1F", "2C"), 76: ("1C", "2F"), 77: ("1I", "M77"), 78: ("2E", "2I"),
               79: ("1A", "M79"), 80: ("1L", "M80"), 81: ("1D", "M81"), 82: ("1G", "M82"), 83: ("2K", "2L"), 84: ("1H", "2J"),
               85: ("1B", "M85"), 86: ("1J", "2H"), 87: ("1K", "M87"), 88: ("2D", "2G")}
        R16 = {89: (74, 77), 90: (73, 75), 91: (76, 78), 92: (79, 80), 93: (83, 84), 94: (81, 82), 95: (86, 88), 96: (85, 87)}
        QF = {97: (89, 90), 98: (93, 94), 99: (91, 92), 100: (95, 96)}; SF = {101: (97, 98), 102: (99, 100)}
        out = {"R32": [], "R16": [], "QF": [], "SF": []}; win, loser = {}, {}
        for m, (c1, c2) in R32.items():
            a, b = slot(c1), slot(c2); w, p = winner(a, b); win[m] = w; out["R32"].append((a, b, w, p))
        for lab, stage in [("R16", R16), ("QF", QF), ("SF", SF)]:
            for m, (x, y) in stage.items():
                a, b = win[x], win[y]; w, p = winner(a, b); win[m] = w; loser[m] = b if w == a else a
                out[lab].append((a, b, w, p))
        a, b = win[101], win[102]; w, p = winner(a, b); out["final"] = (a, b, w, p); out["champion"] = (w, p)
        la, lb = loser[101], loser[102]; out["third"] = (la, lb, *winner(la, lb))
        return out

    def groups(self):
        """{group letter: [team names]} for the 2026 group stage (derived from fixtures + seeded anchors)."""
        wc = self.feat[(self.feat.tournament == "FIFA World Cup") & (self.feat.date.dt.year == 2026)]
        teams = sorted(set(wc.home_team) | set(wc.away_team))
        adj = defaultdict(set)
        for _, r in wc.iterrows():
            adj[r.home_team].add(r.away_team); adj[r.away_team].add(r.home_team)
        ANCH = {"Mexico": "A", "Canada": "B", "Brazil": "C", "United States": "D", "Germany": "E", "Netherlands": "F",
                "Belgium": "G", "Spain": "H", "France": "I", "Argentina": "J", "Portugal": "K", "England": "L"}
        grp, seen = {}, set()
        for t in teams:
            if t in seen: continue
            g = {t} | adj[t]; grp[next(ANCH[x] for x in g if x in ANCH)] = sorted(g); seen |= g
        return {L: grp[L] for L in sorted(grp)}

    def data_through(self):
        """Date of the most recent played match in the data — what the ratings reflect."""
        played = self.feat[self.feat.result.notna()]
        return played.date.max() if len(played) else None


if __name__ == "__main__":
    m = WorldCupModel()
    print(f"{len(m.teams)} teams loaded.  Dixon-Coles rho = {m.rho:+.3f}\nSample predictions:")
    for h, a, n in [("Spain", "Brazil", True), ("Mexico", "South Africa", False), ("Argentina", "France", True)]:
        r = m.predict_match(h, a, n)
        print(f"  {h:9} vs {a:13} (neutral={n}): "
              f"logistic {r['logistic']['home']:.0%}/{r['logistic']['draw']:.0%}/{r['logistic']['away']:.0%} | "
              f"goals {r['goals']['home']:.0%}/{r['goals']['draw']:.0%}/{r['goals']['away']:.0%} | "
              f"xG {r['xg'][0]:.1f}-{r['xg'][1]:.1f} | likely {r['likely_score']}")
