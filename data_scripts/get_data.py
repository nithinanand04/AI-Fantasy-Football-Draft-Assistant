import requests
import json
from pathlib import Path
from time import sleep

BASE_V1 = "https://api.sleeper.app/v1"
BASE = "https://api.sleeper.app"  # for projections/stats/schedule

OUT = Path("sleeper_export_first")
OUT.mkdir(exist_ok=True)

def get_json(url, params=None):
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def save_json(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2))

def fetch_beginning_of_season(season: int):
    # Players directory
    players = get_json(f"{BASE_V1}/players/nfl")
    save_json(players, OUT / f"players_nfl.json")

    # Week 1 projections
    params = {
        "season_type": "regular",
        "position[]": ["QB","RB","WR","TE","K","DEF","FLEX"],
    }
    proj_w1 = get_json(f"{BASE}/projections/nfl/{season}/1", params=params)
    save_json(proj_w1, OUT / f"projections_{season}_week1.json")

    # Schedule (byes)
    sched = get_json(f"{BASE}/schedule/nfl/regular/{season}")
    save_json(sched, OUT / f"schedule_regular_{season}.json")

def fetch_end_of_season_weekly_stats(season: int, player_ids: list[str], sleep_s=0.05):
    # NOTE: per-player endpoint; start with a subset (e.g., top 300 by ADP/proj)
    stats = {}
    for i, pid in enumerate(player_ids, 1):
        url = f"{BASE}/stats/nfl/player/{pid}"
        params = {"season_type": "regular", "season": season, "grouping": "week"}
        try:
            stats[pid] = get_json(url, params=params)
        except requests.HTTPError as e:
            # Some ids might fail; just skip
            stats[pid] = {"error": str(e)}
        if i % 50 == 0:
            print(f"Fetched {i}/{len(player_ids)} players...")
        sleep(sleep_s)

    save_json(stats, OUT / f"stats_weekly_{season}_players_{len(player_ids)}.json")

if __name__ == "__main__":
    season = 2024
    fetch_beginning_of_season(season)

    # Example: pick a subset of player_ids to start
    players = json.loads((OUT / "players_nfl.json").read_text())
    some_player_ids = list(players.keys())[:500]  # replace with top-N by projection/ADP later

    fetch_end_of_season_weekly_stats(season, some_player_ids)
    print("Done.")
