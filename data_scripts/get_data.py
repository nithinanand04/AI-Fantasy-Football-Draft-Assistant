#!/usr/bin/env python3
"""
Sleeper data collection utility.

Modes:
1) Full mode (default for seasons < 2026):
   - players/nfl
   - week1 projections
   - top-N selection with injury/availability filter via weekly stats fetch
   - writes stats_weekly_* and selected_top_*

2) Projections-only mode (default for seasons >= 2026):
   - players/nfl
   - week1 projections
   - top-N by projection only
   - writes selected_top_*
   - skips weekly stats fetch (future seasons typically do not have complete weekly stats yet)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

BASE_V1 = "https://api.sleeper.app/v1"
BASE = "https://api.sleeper.app"
OUT_DIR = Path("sleeper_exports")


def get_json(url: str, params: Optional[dict] = None, timeout_s: int = 30) -> Any:
    r = requests.get(url, params=params, timeout=timeout_s)
    r.raise_for_status()
    return r.json()


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def compute_ppr_points_from_components(stats: Dict[str, Any], ppr: float) -> float:
    def f(key: str) -> float:
        v = stats.get(key)
        return float(v) if isinstance(v, (int, float)) else 0.0

    pts = 0.0
    pts += f("pass_yd") * 0.04
    pts += f("pass_td") * 4.0
    pts += f("pass_int") * -2.0
    pts += f("rush_yd") * 0.1
    pts += f("rush_td") * 6.0
    pts += f("rec_yd") * 0.1
    pts += f("rec_td") * 6.0
    pts += f("rec") * ppr
    pts += f("fum_lost") * -2.0
    return pts


def projected_points_from_projection_row(row: Dict[str, Any], scoring: str) -> float:
    stats = row.get("stats") or {}
    scoring = scoring.lower()
    direct_keys_by_scoring = {
        "ppr": ["pts_ppr", "fantasy_pts_ppr", "fpts_ppr", "pts"],
        "half": ["pts_half_ppr", "fantasy_pts_half_ppr", "fpts_half_ppr", "pts"],
        "std": ["pts_std", "fantasy_pts_std", "fpts_std", "pts"],
    }
    for k in direct_keys_by_scoring.get(scoring, []):
        v = stats.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    ppr_val = 1.0 if scoring == "ppr" else (0.5 if scoring == "half" else 0.0)
    return compute_ppr_points_from_components(stats, ppr=ppr_val)


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
    params = {
        "season_type": "regular",
        "position[]": ["QB", "RB", "WR", "TE", "K", "DEF"],
    }
    proj = get_json(f"{BASE}/projections/nfl/{season}/1", params=params)
    save_json(proj, path)
    return proj


def build_sorted_candidates(
    players_map: Dict[str, Any],
    proj_week1: List[Dict[str, Any]],
    scoring: str,
) -> List[Tuple[str, float]]:
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
        scored.append((pid, float(pts)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def week_played(stats: Dict[str, Any]) -> bool:
    """Partial snaps / any real line counts as played."""
    if not stats:
        return False
    gp = stats.get("gp")
    gs = stats.get("gs")
    off = stats.get("off_snp")
    def_snp = stats.get("def_snp") or stats.get("tm_def_snp")
    st_snp = stats.get("st_snp") or stats.get("tm_st_snp")
    for v in (gp, gs, off, def_snp, st_snp):
        if isinstance(v, (int, float)) and v > 0:
            return True
    for k in ("pts_ppr", "pts_std", "pts_half_ppr", "pts"):
        v = stats.get(k)
        if isinstance(v, (int, float)) and v != 0:
            return True
    return False


def count_missed_weeks(
    weekly_by_week: Dict[str, Any],
    bye_week: int,
    season_weeks: int,
) -> int:
    """
    Weeks 1..season_weeks: count non-bye weeks where a row exists and player did not
    meaningfully play. Missing week rows are leniently ignored.
    """
    n = 0
    for w in range(1, season_weeks + 1):
        if bye_week and w == bye_week:
            continue
        wd = weekly_by_week.get(str(w))
        if not isinstance(wd, dict):
            continue
        stats = wd.get("stats") or {}
        if week_played(stats):
            continue
        n += 1
    return n


def fetch_player_weekly(season: int, player_id: str) -> Any:
    url = f"{BASE}/stats/nfl/player/{player_id}"
    params = {"season_type": "regular", "season": season, "grouping": "week"}
    return get_json(url, params=params)


def save_selected_list(
    season: int,
    scoring: str,
    selected: List[str],
    players: Dict[str, Any],
    proj_rows: List[Dict[str, Any]],
) -> None:
    proj_by_pid = {str(r.get("player_id")): r for r in proj_rows if r.get("player_id") is not None}
    rows = []
    for pid in selected:
        meta = players.get(pid, {})
        pr = proj_by_pid.get(pid, {})
        rows.append(
            {
                "player_id": pid,
                "full_name": meta.get("full_name"),
                "position": meta.get("position"),
                "team": meta.get("team"),
                "proj_week1_pts": projected_points_from_projection_row(pr, scoring) if pr else 0.0,
            }
        )
    save_json(rows, OUT_DIR / f"selected_top_{len(selected)}_{season}_week1_{scoring}.json")


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect Sleeper players/projections and optional weekly stats.")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--top-n", type=int, default=500)
    ap.add_argument("--max-injury-weeks", type=int, default=1, help="Exclude if missed-week count exceeds this.")
    ap.add_argument("--season-weeks", type=int, default=14)
    ap.add_argument("--scoring", type=str, default="ppr", choices=["ppr", "half", "std"])
    ap.add_argument("--sleep", type=float, default=0.05)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument(
        "--projections-only",
        action="store_true",
        help="Skip weekly stats fetch and injury filtering; select top-N by projection only.",
    )
    args = ap.parse_args()

    use_cache = not args.no_cache
    OUT_DIR.mkdir(exist_ok=True)

    auto_projections_only = args.season >= 2026
    projections_only = args.projections_only or auto_projections_only

    print("1) Players...")
    players = fetch_players_nfl(use_cache=use_cache)
    print("2) Week 1 projections...")
    proj = fetch_week1_projections(args.season, use_cache=use_cache)
    print("3) Sorted candidate list by projection...")
    ranked = build_sorted_candidates(players, proj, args.scoring)
    print(f"   {len(ranked)} candidates with position + projection.")

    if projections_only:
        selected = [pid for pid, _ in ranked[: args.top_n]]
        save_selected_list(args.season, args.scoring, selected, players, proj)
        save_json(
            {
                "season": args.season,
                "top_n_target": args.top_n,
                "selected_count": len(selected),
                "mode": "projections_only",
                "reason": "manual --projections-only" if args.projections_only else "auto season>=2026",
                "scoring_sort": args.scoring,
            },
            OUT_DIR / "meta.json",
        )
        print("\nDone (projections-only mode).")
        print(f"  Selected: {len(selected)} (target {args.top_n})")
        print(f"  Wrote: {OUT_DIR / f'selected_top_{len(selected)}_{args.season}_week1_{args.scoring}.json'}")
        return

    stats_out: Dict[str, Any] = {}
    selected: List[str] = []
    skipped_injury = 0
    skipped_error = 0

    print(
        f"4) Fetching weekly stats in order; keeping up to {args.top_n} "
        f"with missed_weeks <= {args.max_injury_weeks}..."
    )
    for pid, _proj_pts in ranked:
        if len(selected) >= args.top_n:
            break
        try:
            raw = fetch_player_weekly(args.season, pid)
        except requests.RequestException as e:
            skipped_error += 1
            print(f"   skip {pid}: request error {e}")
            time.sleep(args.sleep)
            continue

        if isinstance(raw, dict) and raw.get("error"):
            skipped_error += 1
            time.sleep(args.sleep)
            continue

        meta = players.get(pid, {}) or {}
        bye_week = _safe_int((meta.get("metadata") or {}).get("bye_week", 0), 0)
        missed = count_missed_weeks(raw, bye_week=bye_week, season_weeks=args.season_weeks)
        if missed > args.max_injury_weeks:
            skipped_injury += 1
            time.sleep(args.sleep)
            continue

        stats_out[pid] = raw
        selected.append(pid)
        if len(selected) % 25 == 0:
            print(f"   ... {len(selected)} / {args.top_n} selected")
        time.sleep(args.sleep)

    out_path = OUT_DIR / f"stats_weekly_{args.season}_players_{len(selected)}.json"
    save_json(stats_out, out_path)
    save_selected_list(args.season, args.scoring, selected, players, proj)

    save_json(
        {
            "season": args.season,
            "top_n_target": args.top_n,
            "selected_count": len(selected),
            "max_injury_weeks_allowed": args.max_injury_weeks,
            "season_weeks": args.season_weeks,
            "filter": "missed weeks = non-bye week with stat row and not week_played(); missing rows lenient; status not used",
            "skipped_injury": skipped_injury,
            "skipped_error": skipped_error,
            "scoring_sort": args.scoring,
            "mode": "full",
        },
        OUT_DIR / "meta.json",
    )

    print("\nDone.")
    print(f"  Selected: {len(selected)} (target {args.top_n})")
    print(f"  Skipped (missed-weeks rule): {skipped_injury}")
    print(f"  Skipped (errors): {skipped_error}")
    print(f"  Wrote: {out_path}")


if __name__ == "__main__":
    main()

