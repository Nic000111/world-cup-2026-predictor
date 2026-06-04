"""
dashboard.py - interactive 2026 World Cup predictor.
Run with:  streamlit run dashboard.py
"""
import os

import pandas as pd
import streamlit as st

import wc

st.set_page_config(page_title="World Cup 2026 Predictor", page_icon="⚽", layout="wide")


# Bump CACHE_VERSION whenever the model interface changes, so every cached entry
# invalidates automatically on next deploy (avoids stale-pickle bugs on Streamlit Cloud).
CACHE_VERSION = "v3-glicko"


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


model = load_model()

st.title("⚽ World Cup 2026 — Match Predictor")
st.caption("Two models on a self-computed **Glicko** rating (full international history; models trained on 2010+) — "
           "each team carries a rating **and an uncertainty**, which beat plain Elo on held-out log-loss. "
           "With a **per-confederation adjustment** for cross-continental strength: "
           "**Glicko-logistic** for win/draw/loss · **Glicko-Poisson + Dixon–Coles** for goals & scorelines. "
           "Hyperparameters tuned on 2022–23 and validated on a held-out 2024–25 test set; "
           "deployed model refit on all played games through today.")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["🔮 Predict a match", "📋 Group forecast", "📊 Ratings", "➕ Enter result", "🏆 Title odds", "🗺️ Bracket"])

# ───────────────────────── Tab 1: predict a match ─────────────────────────
with tab1:
    teams = model.teams
    d_home = teams.index("France") if "France" in teams else 0
    d_away = teams.index("England") if "England" in teams else 1
    c1, c2, c3 = st.columns([3, 3, 2])
    home = c1.selectbox("🏠 Home team", teams, index=d_home)
    away = c2.selectbox("✈️ Away team", teams, index=d_away)
    venue = c3.radio("Venue", ["Neutral (e.g. World Cup)", f"{home} at home"], index=0)
    c3.date_input("Match date (context only — model uses current strength)")

    if home == away:
        st.warning("Pick two different teams.")
    else:
        neutral = venue.startswith("Neutral")
        r = model.predict_match(home, away, neutral=neutral)

        st.markdown(f"### {home} {'(home) ' if not neutral else ''}vs {away}")
        rt = r.get("rating", r.get("elo", (0.0, 0.0)))          # tolerate a stale cached model during redeploys
        rdv = r.get("rd")
        if rdv:
            st.caption(f"Glicko rating —  {home}: **{rt[0]:.0f}** ±{rdv[0]:.0f}   ·   "
                       f"{away}: **{rt[1]:.0f}** ±{rdv[1]:.0f}   (± = uncertainty)")
        else:
            st.caption(f"Rating —  {home}: **{rt[0]:.0f}**   ·   {away}: **{rt[1]:.0f}**")

        st.markdown("#### Win / Draw / Loss")
        m1, m2, m3 = st.columns(3)
        m1.metric(f"{home} win", f"{r['goals']['home'] * 100:.0f}%")
        m2.metric("Draw", f"{r['goals']['draw'] * 100:.0f}%")
        m3.metric(f"{away} win", f"{r['goals']['away'] * 100:.0f}%")
        prob = pd.DataFrame(
            {"Glicko-logistic": [r["logistic"]["home"], r["logistic"]["draw"], r["logistic"]["away"]],
             "Goals model": [r["goals"]["home"], r["goals"]["draw"], r["goals"]["away"]]},
            index=[f"{home} win", "Draw", f"{away} win"]) * 100
        st.bar_chart(prob, horizontal=True)

        st.markdown("#### Goals & most-likely scorelines")
        g1, g2, g3 = st.columns(3)
        g1.metric(f"{home} expected goals", f"{r['xg'][0]:.2f}")
        g2.metric(f"{away} expected goals", f"{r['xg'][1]:.2f}")
        g3.metric("Most-likely score", r["likely_score"])
        sc = pd.DataFrame(r["top_scores"], columns=["score", "probability %"])
        sc["probability %"] = (sc["probability %"] * 100).round(1)
        st.bar_chart(sc.set_index("score"))
        st.caption("Each exact score is individually unlikely — the rest of the probability spreads "
                   "across all other scorelines, which is why the single most-likely score is always tidy/low.")

# ───────────────────────── Tab 2: 2026 group forecast ─────────────────────────
with tab2:
    st.subheader("2026 World Cup — group-stage forecast")
    st.caption("Win/draw/loss from the goals model · expected goals (xG) · most-likely scoreline. "
               "Played games drop off as you enter results.")
    gf = fixtures().copy()
    if len(gf):
        for col in ["home_win", "draw", "away_win"]:
            gf[col] = (gf[col] * 100).round(0).astype(int)
        gf["xG"] = gf.xg_home.round(1).astype(str) + " – " + gf.xg_away.round(1).astype(str)
        st.dataframe(gf[["date", "home", "away", "home_win", "draw", "away_win", "xG", "likely_score"]],
                     width="stretch", height=560, hide_index=True)
    else:
        st.info("All group fixtures have been played (entered as results). 🎉")

# ───────────────────────── Tab 3: ratings ─────────────────────────
with tab3:
    st.subheader("Current Glicko ratings")
    rt = ratings()
    st.bar_chart(rt.head(25).set_index("team")["rating"], horizontal=True, height=520)
    st.dataframe(rt, width="stretch", height=420, hide_index=True)

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

# ───────────────────────── Tab 4: enter a result ─────────────────────────
with tab4:
    if st.session_state.get("saved_msg"):
        st.success(st.session_state.pop("saved_msg"))
    st.subheader("➕ Enter a result — the ratings absorb it and every prediction updates")
    st.caption("As games are played, record the score here. Pick an upcoming fixture (names guaranteed "
               "to match), or a custom match for anything else.")
    # Password gate (active only when ENTER_PWD is set as a Streamlit secret — i.e. on the deployed app).
    # Locally with no secrets.toml, the gate is bypassed automatically.
    # CRITICAL: gate ONLY this tab — do NOT call st.stop(), which would halt the whole script and
    # leave the Title odds & Bracket tabs blank.
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

# ───────────────────────── Tab 5: title odds ─────────────────────────
with tab5:
    st.write(f"_build {CACHE_VERSION}_")   # sentinel: if you DON'T see this line, the deploy is stale
    st.subheader("🏆 Title odds — full tournament simulation")
    st.caption("Monte-Carlo of the entire bracket (group stage → final, with the real format & tiebreakers) "
               "from the goals model. Updates automatically as you enter results.")
    sd = st.slider("Uncertainty scale — higher spreads the favourites out", 0.0, 3.0, 1.5, 0.25,
                   help="Multiplies each team's OWN Glicko uncertainty. ~0 over-concentrates the top; "
                        "~1.5 gives a market-like spread, with data-poor teams spread wider than well-known ones.")
    try:
        with st.spinner("Simulating 20,000 tournaments…"):
            sim = run_sim(sd)
        st.bar_chart(sim.head(16).set_index("team")["champion"].mul(100), horizontal=True, height=460)
        show = sim.copy()
        for c in ["win_group", "advance", "reach_QF", "reach_SF", "final", "champion"]:
            show[c] = (show[c] * 100).round(1)
        show = show.rename(columns={"win_group": "win grp %", "advance": "reach R32 %", "reach_QF": "reach QF %",
                                    "reach_SF": "reach SF %", "final": "final %", "champion": "CHAMPION %"})
        st.dataframe(show.head(32), width="stretch", height=460, hide_index=True)
    except Exception as e:
        import traceback
        st.error(f"Title odds failed: **{type(e).__name__}: {e}**")
        st.code(traceback.format_exc())

# ───────────────────────── Tab 6: projected bracket ─────────────────────────
with tab6:
    st.write(f"_build {CACHE_VERSION}_")   # sentinel: if you DON'T see this line, the deploy is stale
    st.subheader("🗺️ Most-likely knockout bracket")
    st.caption("From the current ratings + your entered results: projected group standings → the favourite "
               "advances each round. The % is that favourite's chance in the tie. Re-projects as you enter results.")
    try:
        bk = bracket()
        st.markdown(render_bracket(bk), unsafe_allow_html=True)
        la, lb, lw, lp = bk["third"]
        st.caption(f"🥉 Third-place play-off: **{la}** vs **{lb}** → **{lw}** ({lp * 100:.0f}%)")
    except Exception as e:
        import traceback
        st.error(f"Bracket failed: **{type(e).__name__}: {e}**")
        st.code(traceback.format_exc())

st.divider()
st.caption("⚠️ A model, not a crystal ball: ~40% of matches (draws + upsets) are near-random, so these "
           "are honest *probabilities*, not certainties.")
