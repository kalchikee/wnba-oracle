#!/usr/bin/env python3
"""
WNBA Dataset Builder
Fetches historical WNBA game results from ESPN (2021-2025),
builds Elo ratings, and saves training_data.csv.

Usage: python python/build_dataset.py
"""
import sys, json, time, requests
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from datetime import date, timedelta

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data"
CACHE_DIR = ROOT / "cache"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV   = DATA_DIR / "training_data.csv"
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba"
HEADERS   = {"User-Agent": "WNBA-Oracle/4.1"}

# WNBA seasons run May–October each year
SEASONS = [2021, 2022, 2023, 2024, 2025]

INITIAL_ELO = 1500.0
K_FACTOR    = 20.0
HOME_ADV    = 60.0   # ~3.5 pts in WNBA


def espn_get(url: str) -> dict:
    for i in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if i == 2:
                print(f"  Failed {url}: {e}")
                return {}
            time.sleep(2 ** i)
    return {}


def fetch_season_games(year: int) -> list:
    cache = CACHE_DIR / f"wnba_{year}.json"
    if cache.exists():
        return json.loads(cache.read_text())

    games = []
    seen  = set()
    # WNBA regular season: May 1 – Sep 20; playoffs Sep 20 – Oct 20
    start = date(year, 5, 1)
    end   = date(year, 10, 25)
    current = start

    while current <= end:
        date_str = current.strftime("%Y%m%d")
        data = espn_get(f"{ESPN_BASE}/scoreboard?dates={date_str}&limit=20")
        for ev in data.get("events", []):
            eid = ev.get("id", "")
            if eid in seen:
                current += timedelta(days=1)
                continue
            seen.add(eid)
            status = ev.get("status", {}).get("type", {}).get("completed", False)
            if not status:
                continue
            comp = (ev.get("competitions") or [{}])[0]
            cs   = comp.get("competitors", [])
            home = next((c for c in cs if c.get("homeAway") == "home"), None)
            away = next((c for c in cs if c.get("homeAway") == "away"), None)
            if not home or not away:
                continue
            h_abbr = home.get("team", {}).get("abbreviation", "").upper()
            a_abbr = away.get("team", {}).get("abbreviation", "").upper()
            h_score = int(home.get("score", 0) or 0)
            a_score = int(away.get("score", 0) or 0)
            if not h_abbr or not a_abbr or (h_score == 0 and a_score == 0):
                continue
            neutral = int(comp.get("neutralSite", False))
            # Win% records
            h_wp = a_wp = 0.5
            for rec in home.get("records", []):
                if rec.get("type") == "total":
                    gp = sum([rec.get("wins",0), rec.get("losses",0)])
                    wins = rec.get("wins", 0)
                    if gp > 0:
                        h_wp = wins / gp
            for rec in away.get("records", []):
                if rec.get("type") == "total":
                    gp = sum([rec.get("wins",0), rec.get("losses",0)])
                    wins = rec.get("wins", 0)
                    if gp > 0:
                        a_wp = wins / gp

            games.append({
                "game_id":    eid,
                "game_date":  current.strftime("%Y-%m-%d"),
                "season":     year,
                "home_team":  h_abbr,
                "away_team":  a_abbr,
                "home_score": h_score,
                "away_score": a_score,
                "home_win":   1 if h_score > a_score else 0,
                "neutral":    neutral,
                "h_win_pct":  h_wp,
                "a_win_pct":  a_wp,
            })
        current += timedelta(days=1)

    # Deduplicate
    seen2 = set(); unique = []
    for g in games:
        if g["game_id"] not in seen2:
            seen2.add(g["game_id"]); unique.append(g)

    cache.write_text(json.dumps(unique, indent=2))
    print(f"  {year}: {len(unique)} games fetched")
    return unique


def build_elo(games: list) -> list:
    """Walk games chronologically, build Elo + rolling 10-game form features.

    The rolling features look only at the team's prior games (no leakage) and
    are zero-filled until each team has 3 games of history. Net point diff and
    recent win pct over the last 10 games capture form/injury context that
    season-long win% misses; these are the most lift-per-unit-effort features
    available without box scores."""
    elo = defaultdict(lambda: INITIAL_ELO)
    # Per-team deque of (date, point_diff, won) for the last 10 games.
    from collections import deque
    form: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
    rows = []

    def last10_stats(team: str) -> tuple[float, float]:
        d = form[team]
        if len(d) < 3:
            return 0.0, 0.5
        net = sum(x[1] for x in d) / len(d)
        win_pct = sum(x[2] for x in d) / len(d)
        return net, win_pct

    for g in sorted(games, key=lambda x: (x["season"], x["game_date"])):
        h, a    = g["home_team"], g["away_team"]
        rh, ra  = elo[h], elo[a]
        neutral = g.get("neutral", 0)
        ha      = 0.0 if neutral else HOME_ADV
        exp_h   = 1.0 / (1.0 + 10 ** ((ra - (rh + ha)) / 400.0))
        hw      = g["home_win"]
        delta   = K_FACTOR * (hw - exp_h)

        h_wp = g.get("h_win_pct", 0.5)
        a_wp = g.get("a_win_pct", 0.5)

        # Pull last-10 BEFORE this game so the features don't leak the result.
        h_net, h_recent_wp = last10_stats(h)
        a_net, a_recent_wp = last10_stats(a)

        rows.append({
            "season":             g["season"],
            "game_date":          g["game_date"],
            "home_team":          h,
            "away_team":          a,
            "home_score":         g["home_score"],
            "away_score":         g["away_score"],
            "home_win":           hw,
            "label":              hw,
            "neutral":            neutral,
            "elo_diff":           rh - ra,
            "win_pct_diff":       h_wp - a_wp,
            "log5_prob":          h_wp / (h_wp + a_wp) if (h_wp + a_wp) > 0 else 0.5,
            "pythagorean_diff":   h_wp - a_wp,
            "is_home":            1.0 - neutral,
            "recent_10_net_pts_diff":  h_net - a_net,
            "recent_10_win_pct_diff":  h_recent_wp - a_recent_wp,
            "home_last10_net_pts":     h_net,
            "away_last10_net_pts":     a_net,
        })

        # Update Elo + form AFTER recording the row (so the row's features
        # reflect pre-game state only).
        elo[h] = rh + delta
        elo[a] = ra - delta
        pt_diff = g["home_score"] - g["away_score"]
        form[h].append((g["game_date"], pt_diff,  hw))
        form[a].append((g["game_date"], -pt_diff, 1 - hw))

    return rows


def main():
    print("WNBA Dataset Builder")
    print("=" * 40)
    all_games = []
    for year in SEASONS:
        print(f"\nSeason {year}")
        games = fetch_season_games(year)
        all_games.extend(games)

    print(f"\nTotal games: {len(all_games)}")
    rows = build_elo(all_games)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(df)} rows to {OUT_CSV}")
    print(f"Home win rate: {df['home_win'].mean():.3f}")
    print(f"Seasons: {sorted(df['season'].unique().tolist())}")


if __name__ == "__main__":
    main()
