"""Writes today's WNBA predictions to predictions/YYYY-MM-DD.json.

The kalshi-safety service fetches this file via GitHub raw URL to
decide which picks to back on Kalshi. This module only emits the
JSON — it does not place any bets.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
PREDICTIONS_DIR = ROOT / "predictions"

MIN_PROB = float(os.environ.get("KALSHI_MIN_PROB", "0.58"))


# Mirrors NBA Oracle's confidence ladder (src/kalshi/predictionsFile.ts +
# src/features/marketEdge.ts) so kalshi-safety can read the same tier field
# from either sport and the Discord embed uses the same emoji ladder.
def confidence_tier(prob: float) -> str:
    p = max(prob, 1.0 - prob)
    if p >= 0.72: return "extreme"
    if p >= 0.67: return "high"
    if p >= 0.62: return "medium"
    if p >= 0.57: return "low"
    return "none"


def _normalize_date(date_str: str) -> str:
    """Return an ISO YYYY-MM-DD date from either YYYYMMDD or YYYY-MM-DD input."""
    s = date_str.strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    return s


def write_predictions_file(date: str, results: list[dict]) -> str:
    """Write predictions/<date>.json in the kalshi-safety schema.

    `results` is a list of dicts shaped like the in-memory records used by
    predict.py/discord_alert.py — i.e. each entry has at minimum
    home_abbr, away_abbr, home_prob, away_prob.
    """
    iso_date = _normalize_date(date)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PREDICTIONS_DIR / f"{iso_date}.json"

    picks: list[dict] = []
    for r in results:
        home_prob = float(r.get("home_prob", 0.0))
        away_prob = float(r.get("away_prob", 1.0 - home_prob))
        favored_home = home_prob >= away_prob
        model_prob = max(home_prob, away_prob)
        if model_prob < MIN_PROB:
            continue
        home = str(r.get("home_abbr", ""))
        away = str(r.get("away_abbr", ""))
        picks.append({
            "gameId": f"wnba-{iso_date}-{away}-{home}",
            "home": home,
            "away": away,
            "pickedTeam": home if favored_home else away,
            "pickedSide": "home" if favored_home else "away",
            "modelProb": round(model_prob, 4),
            "confidenceTier": confidence_tier(model_prob),
            "extra": {
                "homeProb": round(home_prob, 4),
                "awayProb": round(away_prob, 4),
            },
        })

    payload = {
        "sport": "WNBA",
        "date": iso_date,
        "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "picks": picks,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    return str(out_path)
