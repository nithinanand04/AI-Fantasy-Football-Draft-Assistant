#!/usr/bin/env python3
"""
Sleeper data exporter (V1)

What it does:
1) Downloads Sleeper player directory (/v1/players/nfl)
2) Downloads Week 1 projections (/projections/nfl/{season}/1) and selects the TOP-N
   fantasy-relevant players (QB/RB/WR/TE) by projected fantasy points (PPR by default)
3) For those TOP-N player_ids, downloads end-of-season WEEKLY stats
   (/stats/nfl/player/{player_id}?season=...&grouping=week)
4) Saves everything to ./sleeper_exports/

Notes:
- Projections/stats endpoints are not strongly documented on docs.sleeper.com, but are
  commonly used by the community.
- Weekly stats are raw stat categories; "boom/bust" is something YOU compute later.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import requests

BASE_V1 = "https://api.sleeper.app/v1"
BASE = "https://api.sleeper.app"

OUT_DIR = Path("sleeper_exports")


# -----------------------------
# HTTP helpers
# -----------------------------
def get_json(url: str, params: Optional[dict] = None, timeout_s: int = 30) -> Any:
    r = requests.get(url, params=params, timeout=timeout_s)
    r.raise_for_status()
    return r.json()


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


# -----------------------------
# Fantasy points helpers (projections)
# -----------------------------
def compute_ppr_points_from_components(stats: Dict[str, Any], ppr: float) -> float:
    """
    Compute a rough fantasy points estimate from common stat components if a direct
    pts_* field isn't present in Sleeper projections.

    This is a *reasonable* fallback. You can adjust to match your league.
    """
    def f(key: str) -> float:
        v = stats.get(key)
        return float(v) if isinstance(v, (int, float)) else 0.0

    # Offense (common)
    pass_yd = f("pass_yd")
    pass_td = f("pass_td")
    pass_int = f("pass_int")

    rush_yd = f("rush_yd")
    rush_td = f("rush_td")

    rec = f("rec")
    rec_yd = f("rec_yd")
    rec_td = f("rec_td")

    fum_lost = f("fum_lost")

    # Very standard-ish scoring
    pts = 0.0
    pts += pass_yd * 0.04
    pts += pass_td * 4.0
    pts += pass_int * -2.0

    pts += rush_yd * 0.1
    pts += rush_td * 6.0

    pts += rec_yd * 0.1
    pts += rec_td * 6.0
    pts += rec * ppr

    pts += fum_lost * -2.0

    return pts


def projected_points_from_projection_row(row: Dict[str, Any], scoring: str) -> float:
    """
    Try to pull a direct fantasy points field if available; otherwise compute from components.
    scoring: "ppr" | "half" | "std"
    """
    stats = row.get("stats") or {}
    scoring = scoring.lower()

    # Try common direct keys first (these vary across datasets/years)
    direct_keys_by_scoring = {
        "ppr": ["pts_ppr", "fantasy_pts_ppr", "fpts_ppr", "pts"],
        "half": ["pts_half_ppr", "fantasy_pts_half_ppr", "fpts_half_ppr", "pts"],
        "std": ["pts_std", "fantasy_pts_std", "fpts_std", "pts"],
    }

    for k in direct_keys_by_scoring.get(scoring, []):
        v = stats.get(k)
        if isinstance(v, (int, float)):
            return float(v)

    # Fallback: compute from components
    ppr_val = 1.0 if scoring == "ppr" else (0.5 if scoring == "half" else 0.0)
    return compute_ppr_points_from_components(stats, ppr=ppr_val)


# -----------------------------
# Data pull steps
# -----------------------------
def fetch_players_nfl(use_cache: bool) -> Dict[str, Any]:
    path = OUT_DIR / "players_nfl.json"
    if use_cache and path.exists():
        return load_json(path)

    players = get_json(f"{BASE_V1}/players/nfl")
    save_json(players, path)
    return players


def fetch_week1_projections(season: int, use_cache: bool) -> List[Dict[str, Any]]:
    path = OUT_DIR / f"projections_{season}_week1.json"
    if use_cache and path.exists():
        return load_json(path)

    # Include K and DEF so draft universe has all starter positions
    params = {
        "season_type": "regular",
        "position[]": ["QB", "RB", "WR", "TE", "K", "DEF"],
    }
    proj = get_json(f"{BASE}/projections/nfl/{season}/1", params=params)
    save_json(proj, path)
    return proj


def select_top_n_player_ids(
    players_map: Dict[str, Any],
    proj_week1: List[Dict[str, Any]],
    top_n: int,
    scoring: str,
) -> List[str]:
    valid_pos = {"QB", "RB", "WR", "TE", "K", "DEF"}

    scored: List[Tuple[str, float]] = []
    for row in proj_week1:
        pid = row.get("player_id")
        if pid is None:
            continue
        pid = str(pid)

        meta = players_map.get(pid)
        if not meta:
            continue

        pos = meta.get("position")
        if pos not in valid_pos:
            continue

        pts = projected_points_from_projection_row(row, scoring=scoring)
        # Filter out obvious zeros if desired (often injured/irrelevant)
        scored.append((pid, float(pts)))

    scored.sort(key=lambda x: x[1], reverse=True)

    # Some positions might not have projections; if list is short, just return what we have
    chosen = [pid for pid, _ in scored[:top_n]]
    return chosen


def fetch_end_of_season_weekly_stats(
    season: int,
    player_ids: List[str],
    sleep_s: float = 0.05,
    use_cache: bool = True,
) -> Dict[str, Any]:
    path = OUT_DIR / f"stats_weekly_{season}_players_{len(player_ids)}.json"
    if use_cache and path.exists():
        return load_json(path)

    stats: Dict[str, Any] = {}
    for i, pid in enumerate(player_ids, 1):
        url = f"{BASE}/stats/nfl/player/{pid}"
        params = {"season_type": "regular", "season": season, "grouping": "week"}

        try:
            stats[pid] = get_json(url, params=params)
        except requests.HTTPError as e:
            # Some ids might not exist in stats provider; store error and continue
            stats[pid] = {"error": f"HTTPError: {str(e)}"}
        except requests.RequestException as e:
            stats[pid] = {"error": f"RequestException: {str(e)}"}

        if i % 50 == 0 or i == len(player_ids):
            print(f"Fetched weekly stats: {i}/{len(player_ids)}")

        time.sleep(sleep_s)

    save_json(stats, path)
    return stats


def export_selected_player_list(
    season: int,
    player_ids: List[str],
    players_map: Dict[str, Any],
    proj_week1: List[Dict[str, Any]],
    scoring: str,
) -> None:
    """
    Save a compact list so you can inspect what you're fetching (names/pos/team/proj).
    """
    proj_by_pid = {}
    for row in proj_week1:
        pid = row.get("player_id")
        if pid is None:
            continue
        proj_by_pid[str(pid)] = row

    out_rows = []
    for pid in player_ids:
        meta = players_map.get(pid, {})
        proj_row = proj_by_pid.get(pid, {})
        pts = projected_points_from_projection_row(proj_row, scoring=scoring) if proj_row else 0.0

        out_rows.append(
            {
                "player_id": pid,
                "full_name": meta.get("full_name"),
                "position": meta.get("position"),
                "team": meta.get("team"),
                "proj_week1_pts": pts,
            }
        )

    save_json(out_rows, OUT_DIR / f"selected_top_{len(player_ids)}_{season}_week1_{scoring}.json")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Export Sleeper NFL players + projections + weekly stats for top N.")
    parser.add_argument("--season", type=int, required=True, help="NFL season year, e.g. 2024")
    parser.add_argument("--top-n", type=int, default=500, help="How many draft-relevant players to keep (default 500)")
    parser.add_argument("--scoring", type=str, default="ppr", choices=["ppr", "half", "std"], help="Scoring for projection sorting")
    parser.add_argument("--sleep", type=float, default=0.05, help="Sleep between per-player stats calls (default 0.05s)")
    parser.add_argument("--no-cache", action="store_true", help="Ignore cached files and re-fetch")
    args = parser.parse_args()

    use_cache = not args.no_cache
    OUT_DIR.mkdir(exist_ok=True)

    print("1) Fetching players directory...")
    players = fetch_players_nfl(use_cache=use_cache)

    print("2) Fetching week 1 projections...")
    proj_w1 = fetch_week1_projections(season=args.season, use_cache=use_cache)

    print(f"3) Selecting top {args.top_n} QB/RB/WR/TE by week1 projected points ({args.scoring})...")
    top_ids = select_top_n_player_ids(players, proj_w1, top_n=args.top_n, scoring=args.scoring)
    print(f"Selected {len(top_ids)} players.")

    print("4) Exporting selected player list for sanity-checking...")
    export_selected_player_list(args.season, top_ids, players, proj_w1, scoring=args.scoring)

    print("5) Fetching end-of-season weekly stats for selected players (this can take a bit)...")
    fetch_end_of_season_weekly_stats(
        season=args.season,
        player_ids=top_ids,
        sleep_s=args.sleep,
        use_cache=use_cache,
    )

    print("\nDone. Files written to ./sleeper_exports/")
    print(f"- players_nfl.json")
    print(f"- projections_{args.season}_week1.json")
    print(f"- selected_top_{len(top_ids)}_{args.season}_week1_{args.scoring}.json")
    print(f"- stats_weekly_{args.season}_players_{len(top_ids)}.json")


if __name__ == "__main__":
    main()
