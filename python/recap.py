#!/usr/bin/env python3
"""WNBA recap — grade past predictions against ESPN final scores.

Reads predictions/<date>.json files that haven't been graded yet, fetches
ESPN's WNBA scoreboard for each date, matches each pick to its final
score, and appends a graded row to data/grading_history.json:

    {
      "lastUngradedDate": "2026-05-09",      # earliest date with open picks
      "graded": [
        {
          "date": "2026-05-02",
          "gameId": "wnba-2026-05-02-NIGERIA-IND",
          "away": "NIGERIA",
          "home": "IND",
          "pickedTeam": "IND",
          "modelProb": 0.6479,
          "actualWinner": "IND",
          "correct": true,
          "homeScore": 105,
          "awayScore": 57,
          "gradedAt": "2026-05-10T04:00:00Z"
        }
      ]
    }

Idempotent: re-running on the same date is a no-op for already-graded
games. Skips games not yet completed (will pick them up next run).

Usage:
    python python/recap.py                 # grade everything ungraded
    python python/recap.py --date 20260502 # grade just that one day
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
PREDICTIONS_DIR = ROOT / "predictions"
HISTORY_FILE = ROOT / "data" / "grading_history.json"
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard"

# Abbreviation aliases — WNBA Oracle's prediction files sometimes spell out
# the full name ("NIGERIA") while ESPN uses a 5-char abbreviation ("NIGER").
# Normalize both sides to whatever ESPN returns so matching works.
ABBR_ALIASES: dict[str, str] = {
    "NIGERIA": "NIGER",
}


def normalize_abbr(s: str) -> str:
    """Map a prediction-side abbreviation to its ESPN equivalent."""
    return ABBR_ALIASES.get(s, s)


def load_history() -> dict:
    if not HISTORY_FILE.exists():
        return {"lastUngradedDate": None, "graded": []}
    try:
        return json.loads(HISTORY_FILE.read_text())
    except Exception as e:
        # Don't silently overwrite — preserve the file and start a fresh
        # in-memory history, matching the safety pattern in the kalshi-safety
        # paperTradeGate fix.
        backup = HISTORY_FILE.with_suffix(f".corrupt-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.json")
        try:
            HISTORY_FILE.rename(backup)
            print(f"[recap] CORRUPT history file — preserved as {backup}; starting fresh: {e}")
        except Exception:
            print(f"[recap] CORRUPT history file — could not back up; starting fresh: {e}")
        return {"lastUngradedDate": None, "graded": []}


def save_history(history: dict) -> None:
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = HISTORY_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(history, indent=2))
    tmp.replace(HISTORY_FILE)


def fetch_espn_scoreboard(iso_date: str) -> list[dict]:
    """Returns a list of completed games for the given ISO date, each shaped:
       {away, home, awayScore, homeScore, winner}
       Skips games not yet completed."""
    yyyymmdd = iso_date.replace("-", "")
    url = f"{ESPN_BASE}?dates={yyyymmdd}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[recap] ESPN fetch failed for {iso_date}: {e}")
        return []

    out: list[dict] = []
    for ev in data.get("events", []):
        comp = (ev.get("competitions") or [{}])[0]
        if not comp.get("status", {}).get("type", {}).get("completed"):
            continue  # not yet final — pick up next run
        home = next((c for c in comp.get("competitors", []) if c.get("homeAway") == "home"), None)
        away = next((c for c in comp.get("competitors", []) if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        home_abbr = (home.get("team") or {}).get("abbreviation")
        away_abbr = (away.get("team") or {}).get("abbreviation")
        try:
            home_score = int(home.get("score", 0))
            away_score = int(away.get("score", 0))
        except (TypeError, ValueError):
            continue
        if not home_abbr or not away_abbr:
            continue
        winner = home_abbr if home_score > away_score else away_abbr if away_score > home_score else None
        if winner is None:
            continue  # tie — rare in WNBA but skip
        out.append({
            "away": away_abbr,
            "home": home_abbr,
            "awayScore": away_score,
            "homeScore": home_score,
            "winner": winner,
        })
    return out


def grade_date(iso_date: str, history: dict) -> int:
    """Grade all predictions for this date. Returns count of newly-graded picks."""
    pred_file = PREDICTIONS_DIR / f"{iso_date}.json"
    if not pred_file.exists():
        return 0
    try:
        preds = json.loads(pred_file.read_text())
    except Exception as e:
        print(f"[recap] could not parse {pred_file}: {e}")
        return 0

    picks = preds.get("picks", [])
    if not picks:
        return 0

    # Build set of already-graded gameIds for this date so we don't double-record
    already_graded = {g["gameId"] for g in history["graded"] if g.get("date") == iso_date}

    espn_games = fetch_espn_scoreboard(iso_date)
    if not espn_games:
        return 0
    espn_by_matchup = {(g["away"], g["home"]): g for g in espn_games}

    newly_graded = 0
    for pick in picks:
        game_id = pick.get("gameId")
        if game_id in already_graded:
            continue
        away_n = normalize_abbr(str(pick.get("away", "")))
        home_n = normalize_abbr(str(pick.get("home", "")))
        espn = espn_by_matchup.get((away_n, home_n))
        if not espn:
            # Game not found in ESPN scoreboard — could be postponed, an
            # exhibition that ESPN doesn't track, or the abbreviation is
            # something we haven't aliased yet. Skip without recording.
            print(f"[recap] {iso_date} {away_n}@{home_n} not found in ESPN scoreboard — skipping")
            continue
        picked_team_n = normalize_abbr(str(pick.get("pickedTeam", "")))
        actual_winner_n = espn["winner"]
        correct = picked_team_n == actual_winner_n
        history["graded"].append({
            "date": iso_date,
            "gameId": game_id,
            "away": pick.get("away"),
            "home": pick.get("home"),
            "pickedTeam": pick.get("pickedTeam"),
            "modelProb": pick.get("modelProb"),
            "actualWinner": actual_winner_n,
            "correct": correct,
            "homeScore": espn["homeScore"],
            "awayScore": espn["awayScore"],
            "gradedAt": datetime.now(timezone.utc).isoformat(),
        })
        newly_graded += 1
    return newly_graded


def grade_all_ungraded(history: dict) -> int:
    """Walk every prediction file and grade dates that have ungraded picks.
       Returns total count of newly-graded picks."""
    if not PREDICTIONS_DIR.exists():
        return 0
    total = 0
    # Don't try to grade today's predictions — games haven't happened yet.
    today_iso = datetime.now().strftime("%Y-%m-%d")
    for f in sorted(PREDICTIONS_DIR.glob("*.json")):
        iso = f.stem
        if iso >= today_iso:
            continue
        total += grade_date(iso, history)
    return total


def compute_season_stats(history: dict) -> dict:
    """Returns {total, correct, accuracy} over all graded picks."""
    graded = history.get("graded", [])
    total = len(graded)
    correct = sum(1 for g in graded if g.get("correct"))
    accuracy = (correct / total) if total > 0 else 0.0
    return {"total": total, "correct": correct, "accuracy": accuracy}


# Confidence buckets shown in the Discord embed so the user can see how
# the model's declared probability tracks reality at each tier. Each
# bucket is half-open [lo, hi). modelProb is the picked-side probability
# so the lowest possible value is 0.5.
CONFIDENCE_BUCKETS = [
    (0.50, 0.60, "50-60%"),
    (0.60, 0.70, "60-70%"),
    (0.70, 0.80, "70-80%"),
    (0.80, 0.90, "80-90%"),
    (0.90, 1.01, "90%+"),
]


def compute_confidence_buckets(history: dict) -> list[dict]:
    """For each confidence bucket return {label, total, correct, accuracy}.

    Only buckets with at least one graded pick are returned, so the
    embed doesn't carry empty rows when the season is young."""
    graded = history.get("graded", [])
    out = []
    for lo, hi, label in CONFIDENCE_BUCKETS:
        rows = [g for g in graded
                if g.get("modelProb") is not None
                and lo <= float(g["modelProb"]) < hi]
        if not rows:
            continue
        correct = sum(1 for r in rows if r.get("correct"))
        out.append({
            "label":    label,
            "total":    len(rows),
            "correct":  correct,
            "accuracy": correct / len(rows),
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="YYYYMMDD or YYYY-MM-DD; grade only this date")
    args = parser.parse_args()

    history = load_history()

    if args.date:
        iso = args.date if "-" in args.date else f"{args.date[:4]}-{args.date[4:6]}-{args.date[6:8]}"
        newly = grade_date(iso, history)
    else:
        newly = grade_all_ungraded(history)

    if newly > 0:
        save_history(history)

    stats = compute_season_stats(history)
    pct = stats["accuracy"] * 100
    print(f"[recap] newly graded: {newly} picks")
    print(f"[recap] season: {stats['correct']}/{stats['total']} correct ({pct:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
