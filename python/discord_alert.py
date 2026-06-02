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
    fetch_team_recent_form, build_features, predict_proba,
)
from predictions_file import write_predictions_file, confidence_tier
from recap import load_history, compute_season_stats
import time
from datetime import timedelta


# Mirrors NBA Oracle's emoji ladder (src/alerts/discord.ts confidenceBar) so
# WNBA and NBA Discord embeds use the same visual scale instead of WNBA's
# previous single-threshold ⭐ at 65%.
def confidence_bar(prob: float) -> str:
    p = max(prob, 1.0 - prob)
    if p >= 0.72: return "🔥🔥🔥"
    if p >= 0.67: return "🔥🔥"
    if p >= 0.60: return "🔥"
    if p >= 0.55: return "✅"
    return "🪙"

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
        h_form = fetch_team_recent_form(game["home_id"])
        a_form = fetch_team_recent_form(game["away_id"])
        time.sleep(0.1)
        fv = build_features(elo, game["home_abbr"], game["away_abbr"],
                            h_rec, a_rec, game.get("neutral", 0),
                            h_form=h_form, a_form=a_form)
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
    # NBA-parity tier counters — same buckets as src/alerts/discord.ts.
    tier_counts = {"extreme": 0, "high": 0, "medium": 0, "low": 0, "none": 0}
    bet_count = 0  # tier >= high (≥67%), matches NBA's shouldBet threshold
    for r in results:
        pick = r["home_abbr"] if r["home_prob"] >= r["away_prob"] else r["away_abbr"]
        pick_prob = max(r["home_prob"], r["away_prob"])
        tier = confidence_tier(pick_prob)
        tier_counts[tier] += 1
        if tier in ("extreme", "high"):
            bet_count += 1
        bar = confidence_bar(pick_prob)
        field_lines.append(
            f"{bar} **{r['away_abbr']} @ {r['home_abbr']}** → **{pick}** "
            f"({pick_prob*100:.1f}%)"
        )

    # Season accuracy — pulled from data/grading_history.json (populated by
    # the nightly recap workflow). Field is hidden until the season has at
    # least one graded prediction.
    fields = []
    try:
        season = compute_season_stats(load_history())
        if season["total"] > 0:
            fields.append({
                "name": "📊 Season Accuracy",
                "value": (
                    f"**{season['accuracy'] * 100:.1f}%** · "
                    f"{season['correct']}/{season['total']} predictions correct this season"
                ),
                "inline": False,
            })
    except Exception as e:
        print(f"[discord] season-stats read failed (non-blocking): {e}")

    fields.append({
        "name": "📋 All Games",
        "value": "\n".join(field_lines)[:1000],  # Discord 1024 char limit
        "inline": False,
    })

    date_display = datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")
    embed = {
        "title": f"🏀 WNBA Oracle — {date_display}",
        "description": (
            f"**{len(results)} game(s)** · **{bet_count}** to bet (≥67%) · "
            f"🔥🔥🔥 {tier_counts['extreme']} · 🔥🔥 {tier_counts['high']} · "
            f"🔥 {tier_counts['medium']} · ✅ {tier_counts['low']} · "
            f"🪙 {tier_counts['none']}"
        ),
        "color": 0xFF6600,  # WNBA orange
        "fields": fields,
        "footer": {"text": "WNBA Oracle v4.1 | ESPN data + logistic regression"},
        "timestamp": datetime.now().isoformat(),
    }

    send_discord(embed)


if __name__ == "__main__":
    main()
