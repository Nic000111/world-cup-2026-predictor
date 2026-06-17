"""
live_results.py — pull FINISHED 2026 World Cup results from football-data.org, in our schema.

Stateless by design: results are re-fetched on each load, so they survive Streamlit Community
Cloud's ephemeral filesystem (sleep/wake/redeploy) — nothing is written to disk that can be lost.
Returns an empty frame on any failure or missing token, so the app degrades to manual/local data.
"""
import json
import ssl
import urllib.request

import pandas as pd

API_URL = "https://api.football-data.org/v4/competitions/WC/matches"

# Use certifi's CA bundle if present (fixes macOS Pythons that lack system certs); otherwise the
# platform default (fine on Streamlit Cloud / Linux).
try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _SSL_CTX = ssl.create_default_context()

# football-data.org team names -> our (martj42) names. Only the four that actually differ;
# the other 44 match exactly.
NAME_MAP = {
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Cape Verde Islands": "Cape Verde",
    "Congo DR": "DR Congo",
    "Czechia": "Czech Republic",
}


def fetch_wc_results(token, timeout=15):
    """DataFrame of FINISHED World Cup matches (date, home_team, away_team, home_score, away_score,
    tournament, city, country, neutral) with team names mapped to ours. Empty frame on any problem."""
    if not token:
        return pd.DataFrame()
    req = urllib.request.Request(API_URL, headers={"X-Auth-Token": token})
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return pd.DataFrame()

    rows = []
    for m in data.get("matches", []):
        if m.get("status") != "FINISHED":
            continue
        ft = (m.get("score") or {}).get("fullTime") or {}
        hs, as_ = ft.get("home"), ft.get("away")
        h = (m.get("homeTeam") or {}).get("name")
        a = (m.get("awayTeam") or {}).get("name")
        if hs is None or as_ is None or not h or not a:
            continue
        rows.append({
            "date": pd.Timestamp(m["utcDate"][:10]),
            "home_team": NAME_MAP.get(h, h),
            "away_team": NAME_MAP.get(a, a),
            "home_score": int(hs),
            "away_score": int(as_),
            "tournament": "FIFA World Cup",
            "city": "", "country": "", "neutral": True,
        })
    return pd.DataFrame(rows)
