#!/usr/bin/env python3
"""
WNBA Discord Alert
Runs the prediction pipeline and sends today's picks to Discord.

Usage:
  python python/discord_alert.py            # today's games
  python python/discord_alert.py --date 20260801
"""
import argparse, os, sys, json, requests
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Import prediction logic from predict.py
sys.path.insert(0, str(Path(__file__).parent))
from predict import (
    load_model, build_elo_from_history, fetch_games, fetch_team_record,
    build_features, predict_proba,
)
from predictions_file import write_predictions_file
import time
from datetime import timedelta

WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL", "")


def send_discord(embed: dict) -> None:
    if not WEBHOOK:
        print("[discord] No DISCORD_WEBHOOK_URL configured — printing embed instead:")
        print(json.dumps(embed, indent=2))
        return
    r = requests.post(WEBHOOK, json={"embeds": [embed]}, timeout=10)
    if r.status_code not in (200, 204):
        raise RuntimeError(f"Discord webhook returned {r.status_code}: {r.text[:200]}")
    print("[discord] Message sent.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
    args = parser.parse_args()
    date_str = args.date

    model = load_model(date_str)
    if not model:
        print("ERROR: no model found — run train_model.py first")
        sys.exit(1)

    elo = build_elo_from_history()
    games = fetch_games(date_str)

    if not games:
        # Off-season: don't spam Discord
        print(f"No WNBA games on {date_str} (season May–October) — skipping Discord alert.")
        return

    scheduled = [g for g in games if "SCHEDULED" in g["status"]] or games
    results = []
    for game in scheduled:
        h_rec = fetch_team_record(game["home_id"])
        a_rec = fetch_team_record(game["away_id"])
        time.sleep(0.1)
        fv = build_features(elo, game["home_abbr"], game["away_abbr"],
                            h_rec, a_rec, game.get("neutral", 0))
        home_p = predict_proba(model, fv)
        results.append({
            "home_abbr": game["home_abbr"],
            "away_abbr": game["away_abbr"],
            "home_prob": home_p,
            "away_prob": 1.0 - home_p,
        })

    if not results:
        print("No predictable games — skipping Discord alert.")
        return

    # Emit predictions JSON for kalshi-safety to consume.
    try:
        out_path = write_predictions_file(date_str, results)
        print(f"[kalshi] Wrote predictions file: {out_path}")
    except Exception as e:
        print(f"[kalshi] Failed to write predictions file: {e}")

    # Build embed
    results.sort(key=lambda x: -max(x["home_prob"], x["away_prob"]))
    field_lines = []
    hc_count = 0
    for r in results:
        pick = r["home_abbr"] if r["home_prob"] >= r["away_prob"] else r["away_abbr"]
        pick_prob = max(r["home_prob"], r["away_prob"])
        star = " ⭐" if pick_prob >= 0.65 else ""
        hc_count += 1 if pick_prob >= 0.65 else 0
        field_lines.append(
            f"**{r['away_abbr']} @ {r['home_abbr']}** → **{pick}** "
            f"({pick_prob*100:.1f}%){star}"
        )

    date_display = datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")
    embed = {
        "title": f"🏀 WNBA Oracle — {date_display}",
        "description": (
            f"**{len(results)} game(s)** · **{hc_count}** high-conviction (65%+)"
        ),
        "color": 0xFF6600,  # WNBA orange
        "fields": [{
            "name": "📋 All Games",
            "value": "\n".join(field_lines)[:1000],  # Discord 1024 char limit
            "inline": False,
        }],
        "footer": {"text": "WNBA Oracle v4.1 | ESPN data + logistic regression"},
        "timestamp": datetime.now().isoformat(),
    }

    send_discord(embed)


if __name__ == "__main__":
    main()
