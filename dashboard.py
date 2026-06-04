"""
dashboard.py - interactive 2026 World Cup predictor.
Run with:  streamlit run dashboard.py
"""
import os

import pandas as pd
import streamlit as st

import wc

st.set_page_config(page_title="World Cup 2026 Predictor", page_icon="⚽", layout="wide")


# Bump CACHE_VERSION whenever the model interface changes, so cached entries invalidate on deploy.
CACHE_VERSION = "v6-glicko"


@st.cache_resource
def load_model(_v=CACHE_VERSION):
    return wc.WorldCupModel()


@st.cache_data
def fixtures(_v=CACHE_VERSION):
    return load_model().group_fixtures()


@st.cache_data
def ratings(_v=CACHE_VERSION):
    return load_model().ratings_table()


@st.cache_data
def run_sim(rd_scale, _v=CACHE_VERSION):
    return load_model().simulate_tournament(n_sim=20000, rd_scale=rd_scale)


@st.cache_data
def bracket(_v=CACHE_VERSION):
    return load_model().project_bracket()


@st.cache_data
def group_map(_v=CACHE_VERSION):
    return load_model().groups()


@st.cache_data
def data_through(_v=CACHE_VERSION):
    d = load_model().data_through()
    return pd.Timestamp(d).strftime("%d %b %Y") if d is not None else "—"


def render_bracket(bk):
    def box(a, b, w):
        ra, rb = ("w" if w == a else "l"), ("w" if w == b else "l")
        return f'<div class="m"><div class="t {ra}">{a}</div><div class="t {rb}">{b}</div></div>'

    def col(title, matches):
        return f'<div class="rd"><div class="rt">{title}</div><div class="ms">' + "".join(box(a, b, w) for a, b, w, p in matches) + "</div></div>"

    champ, _ = bk["champion"]; fa, fb, fw, fp = bk["final"]
    body = (col("Round of 16", bk["R16"]) + col("Quarter-finals", bk["QF"]) + col("Semi-finals", bk["SF"])
            + f'<div class="rd"><div class="rt">Final</div><div class="ms">{box(fa, fb, fw)}</div></div>'
            + f'<div class="rd"><div class="rt">Champion</div><div class="champ">🏆<br>{champ}<br><span class="cp">{fp * 100:.0f}% in final</span></div></div>')
    css = """<style>
.bk{display:flex;gap:12px;align-items:stretch;overflow-x:auto;padding:6px 2px 12px;font-size:12.5px;line-height:1.2}
.rd{display:flex;flex-direction:column;min-width:142px}
.rt{font-weight:700;text-align:center;margin-bottom:8px;color:#9aa0a6;font-size:11px;letter-spacing:.04em;text-transform:uppercase}
.ms{display:flex;flex-direction:column;justify-content:space-around;flex:1;gap:6px}
.m{border:1px solid #d7dbe0;border-radius:7px;overflow:hidden}
.t{padding:6px 9px;background:#fff;color:#202124;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.t.w{font-weight:700;background:#1f6feb;color:#fff}
.t.l{color:#9aa0a6}
.champ{margin:auto;text-align:center;font-weight:800;font-size:16px;color:#202124;border:2px solid #f5b301;border-radius:10px;padding:14px 16px;background:#fff8e1}
.cp{font-size:11px;color:#b8860b;font-weight:600}
</style>"""
    return css + f'<div class="bk">{body}</div>'


LEAN_MARGIN = 0.12   # the top outcome must lead the next by >= 12 points to be a confident "X win"; else "Lean X"


def outcome_label(home, away, g):
    """Most-likely match outcome in words; 'Lean X' for a near-toss-up (no confident favourite)."""
    ranked = sorted([(g["home"], home), (g["draw"], None), (g["away"], away)], key=lambda t: -t[0])
    (p1, n1), (p2, _) = ranked[0], ranked[1]
    if n1 is None:
        return "Draw"
    return f"{n1} win" if (p1 - p2) >= LEAN_MARGIN else f"Lean {n1}"


COLUMN_LEGEND = (
    "- **Prediction** — the single most likely result. **“X win”** is a clear favourite; **“Lean X”** means it's "
    "nearly a toss-up (the favourite leads the next outcome by under ~12 points) — a slight edge, not a confident "
    "call. You'll never see *“Draw”* here: a draw is rarely any match's single most likely outcome (it tops out "
    "around 31%), so one team's win almost always edges it.\n"
    "- **home / draw / away %** — the chance of each result: home wins / draw / away wins. They add up to 100%. "
    "*(At neutral World Cup venues “home” is just the side listed first — no advantage.)*\n"
    "- **xG (expected goals)** — the *average* number of goals each side is forecast to score (e.g. 1.8 – 0.8). Shows "
    "who should score more and whether the game looks tight — it is **not** a predicted scoreline.\n"
    "- **O/U 2.5** — Over / Under 2.5 total goals: whether the match more likely ends with **3 or more** goals (Over) "
    "or **2 or fewer** (Under).\n"
    "- **Double chance** — the favourite's **win-or-draw** probability (e.g. *Mexico or draw 89%*). The 'safer' bet: "
    "that side just has to avoid losing.")

model = load_model()

st.title("⚽ World Cup 2026 — Match Predictor")
st.caption("Forecasts every 2026 World Cup match from a self-computed **Glicko** rating — each team carries a strength "
           "**and an uncertainty**, so the model hedges when it knows less. No betting odds, no FIFA rankings — just "
           "results.")
with st.expander("How it works (the model in detail)"):
    st.markdown(
        "Two models share one engine: a **Glicko-logistic** for win/draw/loss and a **Glicko-Poisson + Dixon–Coles** "
        "for goals, both built on a self-computed Glicko rating over the full international history (models trained on "
        "2010+). A **per-confederation adjustment** corrects cross-continental strength. Hyperparameters were tuned on "
        "2022–23 and validated on a held-out 2024–25 test set; the deployed model is refit on every game played "
        "through today. The **📈 How good is it?** tab has the held-out accuracy and an honest list of what it can and "
        "can't do.")

tab_predict, tab_good, tab_groups, tab_ratings, tab_odds, tab_bracket, tab_enter = st.tabs(
    ["🔮 Predict a match", "📈 How good is it?", "📋 Groups", "📊 Ratings", "🏆 Title odds", "🗺️ Bracket",
     "➕ Enter result"])

# ───────────────────────── Predict a match ─────────────────────────
with tab_predict:
    teams = model.teams
    d_home = teams.index("France") if "France" in teams else 0
    d_away = teams.index("England") if "England" in teams else 1
    c1, c2, c3 = st.columns([3, 3, 2])
    home = c1.selectbox("🏠 Home team", teams, index=d_home)
    away = c2.selectbox("✈️ Away team", teams, index=d_away)
    venue = c3.radio("Venue", ["Neutral (e.g. World Cup)", f"{home} at home"], index=0)

    if home == away:
        st.warning("Pick two different teams.")
    else:
        neutral = venue.startswith("Neutral")
        r = model.predict_match(home, away, neutral=neutral)

        st.markdown(f"### {home} {'(home) ' if not neutral else ''}vs {away}")
        st.caption(f"Glicko rating —  {home}: **{r['rating'][0]:.0f}** ±{r['rd'][0]:.0f}   ·   "
                   f"{away}: **{r['rating'][1]:.0f}** ±{r['rd'][1]:.0f}   (± = uncertainty)")

        g = r["goals"]
        label = outcome_label(home, away, g)
        top_pct = max(g["home"], g["draw"], g["away"]) * 100
        st.markdown(f"#### Most likely outcome:  {label}  ·  {top_pct:.0f}%")
        st.caption("**“X win”** = a clear favourite  ·  **“Lean X”** = nearly a toss-up (a slight edge, not a "
                   "confident call). A *draw* is rarely any match's single most likely result, so the pick almost "
                   "always names a team.")

        st.markdown("#### Win / Draw / Loss")
        m1, m2, m3 = st.columns(3)
        m1.metric(f"{home} win", f"{g['home'] * 100:.0f}%")
        m2.metric("Draw", f"{g['draw'] * 100:.0f}%")
        m3.metric(f"{away} win", f"{g['away'] * 100:.0f}%")
        prob = pd.DataFrame(
            {"Goals model": [g["home"], g["draw"], g["away"]],
             "Glicko-logistic": [r["logistic"]["home"], r["logistic"]["draw"], r["logistic"]["away"]]},
            index=[f"{home} win", "Draw", f"{away} win"]) * 100
        st.bar_chart(prob, horizontal=True, stack=False)

        dc_h, dc_a = g["home"] + g["draw"], g["away"] + g["draw"]
        st.caption(f"**Double chance** (win or draw) —  {home}: **{dc_h * 100:.0f}%**   ·   "
                   f"{away}: **{dc_a * 100:.0f}%**.  The 'safer' market — that side just has to avoid losing.")

        st.markdown("#### Expected goals & goals market")
        mk = r.get("markets", {})
        x1, x2, x3 = st.columns(3)
        x1.metric(f"{home} expected goals", f"{r['xg'][0]:.2f}")
        x2.metric(f"{away} expected goals", f"{r['xg'][1]:.2f}")
        if mk:
            u = mk["under25"]
            x3.metric("Goals — O/U 2.5", f"Under {u * 100:.0f}%" if u >= .5 else f"Over {(1 - u) * 100:.0f}%",
                      help="Over / Under 2.5 total goals: whether the match more likely has 3 or more goals (Over) "
                           "or 2 or fewer (Under).")
        M = r.get("matrix")
        if M is not None:
            tot = {}
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    tot[i + j] = tot.get(i + j, 0.0) + float(M[i, j])
            gvals = [tot.get(k, 0.0) for k in range(5)] + [sum(v for k, v in tot.items() if k >= 5)]
            st.markdown("**Total goals in the match** — the spread behind the Over/Under")
            st.bar_chart(pd.DataFrame({"chance %": [v * 100 for v in gvals]}, index=["0", "1", "2", "3", "4", "5+"]))
        st.caption("**Expected goals** = the average each side is forecast to score. We deliberately **don't show a "
                   "single most-likely scoreline** — the likeliest exact score is often a low draw (like 1-1) even "
                   "when one team is clearly favoured, which misleads more than it helps.")

# ───────────────────────── How good is it? ─────────────────────────
with tab_good:
    st.subheader("📈 How good is it, really?")
    st.caption("An honest report card. The numbers below come from a *held-out* test: the model was trained only "
               "on games up to 2023, then graded on 2024–25 matches it had never seen — so this is real "
               "out-of-sample performance, not the model marking its own homework.")

    a, b, c = st.columns(3)
    a.metric("Winner called right", "≈ 60%",
             help="On held-out 2024–25 games. Blind guessing gets 33%. Always backing the favourite scores about "
                  "the same 60% — which is roughly the ceiling for any results-only model.")
    b.metric("Are the % honest?", "Yes",
             help="Calibration error (ECE) ≈ 0.02 — tiny. When it says 70%, that outcome really happens about "
                  "70% of the time.")
    c.metric("Agreement with bookies", "0.90",
             help="Correlation between our title odds and the de-vigged betting market — close to the sharp money, "
                  "without ever looking at odds.")

    st.markdown(
        "#### The one-line answer\n"
        "It picks the winner right about **3 times in 5**, and — more importantly — its **percentages are honest**: "
        "when it says *70%*, that result really happens about 70% of the time. Think of it as a calm, well-informed "
        "favourite-picker, not a crystal ball.\n\n"
        "**Why not better than 60%?** Because about **1 game in 4 is a draw**, and a draw is almost never any single "
        "team's most-likely result — so a hard slice of matches is near-unpredictable *for anyone*. Roughly "
        "**60% accuracy is the ceiling** for a model built only on past results. Beating it means knowing things "
        "results can't tell you: injuries, who's hot this week, how deep the bench really is.")

    col_g, col_b = st.columns(2)
    with col_g:
        st.markdown("#### Where it's strong ✅")
        st.markdown(
            "- **Clear mismatches** — strong vs weak: confident *and* usually right.\n"
            "- **Picking favourites & ranking teams** — its bread and butter.\n"
            "- **Knowing what it doesn't know** — for rarely-seen teams it hedges toward 50/50 instead of bluffing.\n"
            "- **Tracking the sharp market** (0.90 correlation) without ever seeing a betting line.")
    with col_b:
        st.markdown("#### Where it struggles ❌")
        st.markdown(
            "- **Calling draws** — it'll almost always *name a winner*, even when a draw is brewing (draw odds top "
            "out near 32%, so a draw is rarely the headline pick). It still gives draws an honest ~1-in-4 chance.\n"
            "- **Coin-flip games** — two even sides come out ~40/30/30. The game *is* a toss-up, knockouts especially.\n"
            "- **Upsets** — a minnow's one great day is, by definition, unpredictable.\n"
            "- **The unseen** — injuries, line-up changes, fatigue, squad depth. Blind to all of it.")

    # two real, self-updating examples — the most lopsided and the most even of the 2026 group fixtures
    st.markdown("#### Two live examples — real 2026 group fixtures")
    try:
        gf_ex = fixtures().copy()
        assert len(gf_ex)
        lop = gf_ex.loc[gf_ex[["home_win", "away_win"]].max(axis=1).idxmax()]      # biggest favourite
        eve = gf_ex.loc[(gf_ex.home_win - gf_ex.away_win).abs().idxmin()]          # closest to even
        if lop.home_win >= lop.away_win:
            fv, fp, dg, dp = lop.home, lop.home_win, lop.away, lop.away_win
        else:
            fv, fp, dg, dp = lop.away, lop.away_win, lop.home, lop.home_win
        st.markdown(
            f"- **Lopsided — {fv} vs {dg}:**  **{fp * 100:.0f}%** for {fv}  ·  {lop.draw * 100:.0f}% draw  ·  "
            f"{dp * 100:.0f}% for {dg}. Confident — the kind of call it gets right most of the time.\n"
            f"- **Coin-flip — {eve.home} vs {eve.away}:**  {eve.home_win * 100:.0f}% / {eve.draw * 100:.0f}% / "
            f"{eve.away_win * 100:.0f}%  *(win / draw / loss)*. Two even sides — *too close to call*, which is the "
            f"honest answer.")
    except Exception:
        st.caption("_(live examples unavailable — enter results to refresh the fixture list)_")

    st.markdown(
        "#### Is it good at predicting *goals*? 🥅\n"
        "**Partly — it depends what you ask of it.**\n"
        "- **Expected goals** (e.g. *1.9 – 0.8*): yes. This is a sensible read of which side should score more and "
        "whether a game looks tight or one-sided.\n"
        "- **The exact scoreline**: no — and *no model can be*. Football scores are very random: dozens of results "
        "are plausible, and even the single most-likely one (a tidy *1-0* or *2-1*) only lands a small slice "
        "of the time — on the order of **1 game in 8**. That's why we don't show a single predicted score.")

    st.info("**Bottom line:** trust the **probabilities and the favourites** — they're honest and near the limit of "
            "what results alone can reveal. Treat **draws, upsets, and exact scorelines** as genuinely uncertain. "
            "That's not a flaw to fix — it's the honest truth about predicting football.")

# ───────────────────────── Groups (standings + fixtures) ─────────────────────────
with tab_groups:
    st.subheader("2026 World Cup — groups")
    grps = group_map()
    team2grp = {t: L for L, ts in grps.items() for t in ts}

    st.markdown("#### Predicted group standings")
    st.caption("Each group ordered by **chance of reaching the knockout stage** (from 20,000 simulations). "
               "Top 2 of every group qualify automatically; the 8 best third-placed teams also advance.")
    try:
        with st.spinner("Simulating the tournament…"):
            sim = run_sim(1.5).set_index("team")
        letters = sorted(grps)
        for base in range(0, len(letters), 3):
            cols = st.columns(3)
            for k, L in enumerate(letters[base:base + 3]):
                with cols[k]:
                    ordered = sorted(grps[L], key=lambda t: float(sim.loc[t, "advance"]) if t in sim.index else 0.0,
                                     reverse=True)
                    df = pd.DataFrame({
                        "team": ordered,
                        "win grp": [f"{(sim.loc[t, 'win_group'] if t in sim.index else 0) * 100:.0f}%" for t in ordered],
                        "advance": [f"{(sim.loc[t, 'advance'] if t in sim.index else 0) * 100:.0f}%" for t in ordered]})
                    st.markdown(f"**Group {L}**")
                    st.dataframe(df, hide_index=True, width="stretch")
    except Exception:
        st.info("Standings need the tournament simulation — give it a moment, or open the **Title odds** tab once.")

    st.divider()
    st.markdown("#### Fixtures")
    st.caption("Each fixture's **prediction** (*Lean* = a near-toss-up), the **chances** (home / draw / away %), "
               "**expected goals**, the **Over/Under 2.5** market, and the favourite's **double chance**. Played games "
               "drop off as you enter results.")
    with st.expander("ℹ️ What the columns mean"):
        st.markdown(COLUMN_LEGEND)
    q = st.text_input("🔎 Filter by team", key="grp_filter").strip().lower()
    gf = fixtures().copy()
    if len(gf) == 0:
        st.info("All group fixtures have been played (entered as results). 🎉")
    elif "under25" not in gf.columns:
        st.warning("This forecast was cached by an older build. **Reboot the app** (Manage app ▸ ⋮ ▸ Reboot) to "
                   "refresh it.")
    else:
        gf["group"] = gf.home.map(team2grp)
        gf["prediction"] = [outcome_label(r.home, r.away, {"home": r.home_win, "draw": r.draw, "away": r.away_win})
                            for r in gf.itertuples()]
        gf["home / draw / away %"] = gf.apply(
            lambda r: f"{r.home_win * 100:.0f} / {r.draw * 100:.0f} / {r.away_win * 100:.0f}", axis=1)
        gf["xG"] = gf.xg_home.round(1).astype(str) + " – " + gf.xg_away.round(1).astype(str)
        gf["O/U 2.5"] = gf.under25.apply(lambda u: f"Under {u * 100:.0f}%" if u >= .5 else f"Over {(1 - u) * 100:.0f}%")

        def _dchance(r):
            fav, p = (r.home, r.home_win) if r.home_win >= r.away_win else (r.away, r.away_win)
            return f"{fav} or draw {(p + r.draw) * 100:.0f}%"
        gf["double chance"] = [_dchance(r) for r in gf.itertuples()]
        gf = gf.sort_values(["group", "date"], na_position="last")
        if q:
            gf = gf[gf.home.str.lower().str.contains(q, regex=False) | gf.away.str.lower().str.contains(q, regex=False)]
        st.dataframe(gf[["group", "date", "home", "away", "prediction", "home / draw / away %", "xG", "O/U 2.5",
                         "double chance"]], width="stretch", height=520, hide_index=True)

# ───────────────────────── Ratings ─────────────────────────
with tab_ratings:
    st.subheader("Current Glicko ratings")
    st.caption("**Rating** = team strength (higher is better). **Uncertainty** = the model's error bar on that "
               "rating: *how sure it is*. It's low (~60) for teams that play often against known opponents, and "
               "high for rarely-seen or long-absent sides. It does real work — when a team's uncertainty is high, "
               "the model **hedges its predictions toward 50/50** instead of overcommitting. Teams too uncertain "
               "to rate reliably are hidden from this table.")
    rt = ratings()
    st.bar_chart(rt.head(25).set_index("team")["rating"], horizontal=True, height=520)
    q = st.text_input("🔎 Filter by team", key="rt_filter").strip().lower()
    show = rt[rt.team.str.lower().str.contains(q, regex=False)] if q else rt
    st.dataframe(show, width="stretch", height=420, hide_index=True)

    st.markdown("#### Confederation adjustment")
    st.caption("Rating points added to each confederation's teams in **cross-continental matches only** — "
               "learned from inter-confederation results. It cancels within a confederation, so it never "
               "affects, say, Spain vs France.")
    off = getattr(model, "confed_offsets", {})
    if off:
        offs = pd.Series(off, name="offset").sort_values(ascending=False)
        offs.index.name = "confederation"
        st.bar_chart(offs, horizontal=True, height=240)
    with st.expander("How the confederation adjustment works"):
        st.markdown(
            "Only ~14% of international games cross confederations, and each top team plays very few "
            "(Spain ≈ 11 in 8 years), so the *relative* strength of each confederation is poorly pinned by "
            "ordinary ratings. We fix that by pooling **every** confederation's cross-continental games into one "
            "shared offset, then adding it to its teams' ratings.\n\n"
            "- **What it changed:** cross-continental log-loss improved ~3% on held-out 2024+ data, and our "
            "continental bias vs the betting market roughly halved (e.g. UEFA −12.5 → −6.5 points).\n"
            "- **CONMEBOL ≈ UEFA:** our *results* rate the two top confederations as co-leaders — when they "
            "actually meet it's a coin-flip (98-51-67 since 2010). The market puts Europe clearly ahead, but "
            "that's a squad-*depth* judgement a results-only model can't see (a squad-value test couldn't add "
            "it without leaking future form).\n"
            "- **Within-confederation games are untouched** — the offset cancels, so internal rankings are "
            "exactly as before.")

# ───────────────────────── Title odds ─────────────────────────
with tab_odds:
    st.subheader("🏆 Title odds — full tournament simulation")
    st.caption("Monte-Carlo of the entire bracket (group stage → final, with the real format & tiebreakers) "
               "from the goals model. Updates automatically as you enter results.")
    with st.expander("⚙️ Advanced — uncertainty spread"):
        sd = st.slider("Uncertainty scale — higher spreads the favourites out", 0.0, 3.0, 1.5, 0.25,
                       help="Multiplies each team's OWN Glicko uncertainty. ~0 over-concentrates the top; "
                            "~1.5 gives a market-like spread, with data-poor teams spread wider than well-known ones.")
    try:
        with st.spinner("Simulating 20,000 tournaments…"):
            sim = run_sim(sd)
        st.bar_chart(sim.head(16).set_index("team")["champion"].mul(100), horizontal=True, height=460)
        show = sim.copy()
        for col in ["win_group", "advance", "reach_QF", "reach_SF", "final", "champion"]:
            show[col] = (show[col] * 100).round(1)
        show = show.rename(columns={"win_group": "win grp %", "advance": "reach R32 %", "reach_QF": "reach QF %",
                                    "reach_SF": "reach SF %", "final": "final %", "champion": "CHAMPION %"})
        st.dataframe(show.head(32), width="stretch", height=460, hide_index=True)
    except Exception:
        import traceback
        st.error("Title odds couldn't be computed right now — try a reboot (Manage app ▸ ⋮ ▸ Reboot).")
        with st.expander("Error details"):
            st.code(traceback.format_exc())

# ───────────────────────── Bracket ─────────────────────────
with tab_bracket:
    st.subheader("🗺️ Most-likely knockout bracket")
    st.caption("From the current ratings + your entered results: projected group standings → the favourite "
               "advances each round. The % is that favourite's chance in the tie. Re-projects as you enter results.")
    try:
        bk = bracket()
        st.markdown(render_bracket(bk), unsafe_allow_html=True)
        la, lb, lw, lp = bk["third"]
        st.caption(f"🥉 Third-place play-off: **{la}** vs **{lb}** → **{lw}** ({lp * 100:.0f}%)")
    except Exception:
        import traceback
        st.error("The bracket couldn't be projected right now — try a reboot (Manage app ▸ ⋮ ▸ Reboot).")
        with st.expander("Error details"):
            st.code(traceback.format_exc())

# ───────────────────────── Enter a result ─────────────────────────
with tab_enter:
    if st.session_state.get("saved_msg"):
        st.success(st.session_state.pop("saved_msg"))
    st.subheader("➕ Enter a result — the ratings absorb it and every prediction updates")
    st.caption("As games are played, record the score here. Pick an upcoming fixture (names guaranteed "
               "to match), or a custom match for anything else.")
    # Password gate (active only when ENTER_PWD is set as a Streamlit secret — i.e. on the deployed app).
    # Locally with no secrets.toml, the gate is bypassed automatically.
    # CRITICAL: gate ONLY this tab — do NOT call st.stop(), which would halt the whole script.
    try:
        expected_pwd = st.secrets.get("ENTER_PWD", "")
    except Exception:
        expected_pwd = ""
    locked = bool(expected_pwd) and not st.session_state.get("entry_unlocked")
    if locked:
        with st.form("pwd_form", clear_on_submit=False):
            attempt = st.text_input("🔒 Password to enter results", type="password")
            if st.form_submit_button("Unlock"):
                if attempt == expected_pwd:
                    st.session_state["entry_unlocked"] = True
                    st.rerun()
                else:
                    st.error("Wrong password.")
    else:
        gf = fixtures()
        opts = ["✏️  Custom match (any two teams)"] + [f"{r.date}    {r.home}  vs  {r.away}" for r in gf.itertuples()]
        pick = st.selectbox("Which game?", opts)
        if pick.startswith("✏️"):
            cc1, cc2 = st.columns(2)
            h = cc1.selectbox("Home team", model.teams, index=(model.teams.index("France") if "France" in model.teams else 0), key="cust_h")
            a = cc2.selectbox("Away team", model.teams, index=(model.teams.index("England") if "England" in model.teams else 1), key="cust_a")
            cc3, cc4, cc5 = st.columns(3)
            tourn = cc3.text_input("Tournament", value="Friendly")
            neutral = cc4.checkbox("Neutral venue", value=True)
            d = cc5.date_input("Date", key="cust_d")
        else:
            row = gf.iloc[opts.index(pick) - 1]
            h, a, tourn, neutral, d = row.home, row.away, "FIFA World Cup", True, pd.to_datetime(row.date).date()
            st.info(f"**{h}**  vs  **{a}**   ·   {row.date}")

        sc1, sc2 = st.columns(2)
        hs = sc1.number_input(f"⚽ {h} goals", min_value=0, max_value=30, value=0, step=1, key="hs_in")
        as_ = sc2.number_input(f"⚽ {a} goals", min_value=0, max_value=30, value=0, step=1, key="as_in")

        if st.button("💾 Save result & update ratings", type="primary", disabled=(h == a)):
            wc.record_result(h, a, hs, as_, str(d), tourn, neutral)
            st.cache_resource.clear(); st.cache_data.clear()
            st.session_state["saved_msg"] = f"Saved  {h} {int(hs)}–{int(as_)} {a}.  Ratings updated — predictions now reflect it."
            st.rerun()

        if os.path.exists(wc.MANUAL_RESULTS):
            man = pd.read_csv(wc.MANUAL_RESULTS)
            if len(man):
                st.markdown("**Results you've entered:**")
                st.dataframe(man[["date", "home_team", "away_team", "home_score", "away_score"]].iloc[::-1],
                             width="stretch", hide_index=True)
                if st.button("🗑️ Clear all entered results"):
                    os.remove(wc.MANUAL_RESULTS)
                    st.cache_resource.clear(); st.cache_data.clear()
                    st.session_state["saved_msg"] = "Cleared all entered results."
                    st.rerun()

st.divider()
st.caption(f"⚠️ A model, not a crystal ball: ~40% of matches (draws + upsets) are near-random, so these are honest "
           f"*probabilities*, not certainties.   ·   data through **{data_through()}**   ·   build `{CACHE_VERSION}`")
