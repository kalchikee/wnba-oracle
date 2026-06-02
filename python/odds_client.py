"""WNBA Vegas odds via The Odds API (https://the-odds-api.com).

Opt-in: requires THE_ODDS_API_KEY env var. When the key is unset or the
request fails, fetch_wnba_odds() returns an empty dict and the rest of
the pipeline proceeds without vegasProb (same as today). WNBA coverage
on The Odds API is patchier than NBA — sometimes only a few books quote
each game — so we de-vig by averaging across all available books rather
than picking a single sportsbook.

Free tier: 500 requests/month. One call per morning covers all games on
that date, so daily usage is ~30 calls/month.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import requests

ODDS_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY = "basketball_wnba"


def _decimal_to_implied(decimal_odds: float) -> float:
    """1.91 -> 0.524 (the raw implied probability before de-vig)."""
    if decimal_odds <= 1.0:
        return 0.5
    return 1.0 / decimal_odds


def _devig(home_implied: float, away_implied: float) -> tuple[float, float]:
    """Remove the bookmaker's overround by normalizing to sum=1."""
    total = home_implied + away_implied
    if total <= 0:
        return 0.5, 0.5
    return home_implied / total, away_implied / total


def fetch_wnba_odds(api_key: Optional[str] = None) -> dict[str, float]:
    """Return {game_key: home_vegas_prob} for every WNBA game with odds.

    game_key is "AWAY-HOME" using ESPN-style 3-letter abbreviations
    upper-cased (matches the gameId format used elsewhere in the
    pipeline). Returns {} if no key is set, the API errors, or no
    upcoming WNBA games are listed."""
    key = api_key or os.environ.get("THE_ODDS_API_KEY")
    if not key:
        return {}
    try:
        r = requests.get(
            f"{ODDS_BASE}/sports/{SPORT_KEY}/odds",
            params={
                "regions": "us",
                "markets": "h2h",
                "oddsFormat": "decimal",
                "apiKey": key,
            },
            timeout=10,
        )
        if r.status_code != 200:
            print(f"[odds] WNBA odds fetch failed: HTTP {r.status_code}",
                  file=sys.stderr)
            return {}
        events = r.json()
    except Exception as e:
        print(f"[odds] WNBA odds fetch error: {e}", file=sys.stderr)
        return {}

    # Team-name -> short abbreviation. The Odds API returns full names;
    # WNBA prediction files use ESPN abbreviations.
    name_to_abbr = {
        "Atlanta Dream": "ATL", "Chicago Sky": "CHI", "Connecticut Sun": "CONN",
        "Dallas Wings": "DAL", "Golden State Valkyries": "GSV",
        "Indiana Fever": "IND", "Las Vegas Aces": "LV",
        "Los Angeles Sparks": "LA", "Minnesota Lynx": "MIN",
        "New York Liberty": "NY", "Phoenix Mercury": "PHX",
        "Seattle Storm": "SEA", "Washington Mystics": "WSH",
    }

    out: dict[str, float] = {}
    for ev in events:
        home_name = ev.get("home_team", "")
        away_name = ev.get("away_team", "")
        home_abbr = name_to_abbr.get(home_name)
        away_abbr = name_to_abbr.get(away_name)
        if not home_abbr or not away_abbr:
            continue
        # Average de-vigged implied probability across all available books
        home_probs, away_probs = [], []
        for book in ev.get("bookmakers", []):
            for market in book.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                outcomes = {o.get("name"): float(o.get("price", 0))
                            for o in market.get("outcomes", [])}
                if home_name not in outcomes or away_name not in outcomes:
                    continue
                hi = _decimal_to_implied(outcomes[home_name])
                ai = _decimal_to_implied(outcomes[away_name])
                hp, _ = _devig(hi, ai)
                home_probs.append(hp)
                away_probs.append(1 - hp)
        if not home_probs:
            continue
        out[f"{away_abbr}-{home_abbr}"] = sum(home_probs) / len(home_probs)

    return out
