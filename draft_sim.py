#!/usr/bin/env python3
"""
Run local draft simulation: model team (team 0) vs ADP bots.

- Uses the same feature/model pipeline as the live server.
- Simulates one or many drafts with randomized snake order.
- Prints standings and aggregate metrics for the model team.
"""

from __future__ import annotations

import argparse
import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from data_scripts.label_data import DraftState, LeagueSettings, Prefs, Player, load_json, roster_counts, weekly_points_ppr
from draft_server.ranking import ensure_model_loaded, load_players_map, load_universe, recommend_top

FLEX_POS = {"RB", "WR", "TE"}
AI_TEAM_IDX = 0


# Build randomized snake draft order for one simulation run.
def build_snake_order(settings: LeagueSettings, rng: random.Random) -> List[int]:
    base = list(range(settings.num_teams))
    rng.shuffle(base)
    order: List[int] = []
    for r in range(settings.rounds):
        order.extend(base if r % 2 == 0 else list(reversed(base)))
    return order


# ADP bot that still guarantees all required starter slots can be filled by draft end.
def adp_pick_with_roster_constraints(state: DraftState, team_idx: int) -> str:
    s = state.settings
    roster = state.rosters[team_idx]
    counts = roster_counts(roster, state.available)
    needed = {
        "QB": max(0, s.slot_qb - counts["QB"]),
        "RB": max(0, s.slot_rb - counts["RB"]),
        "WR": max(0, s.slot_wr - counts["WR"]),
        "TE": max(0, s.slot_te - counts["TE"]),
        "K": max(0, s.slot_k - counts["K"]),
        "DEF": max(0, s.slot_def - counts["DEF"]),
    }
    overflow = (
        max(0, counts["RB"] - s.slot_rb)
        + max(0, counts["WR"] - s.slot_wr)
        + max(0, counts["TE"] - s.slot_te)
    )
    flex_needed = max(0, s.slot_flex - overflow)

    missing_required = sum(needed.values()) + flex_needed
    picks_left_including_this = s.rounds - len(roster)
    picks_left_after_this = picks_left_including_this - 1

    must_fill_now = missing_required > picks_left_after_this
    if must_fill_now:
        forced: List[Player] = []
        missing_pos = {pos for pos, n in needed.items() if n > 0}
        if missing_pos:
            forced.extend([p for p in state.available.values() if p.pos in missing_pos])
        if flex_needed > 0:
            forced.extend([p for p in state.available.values() if p.pos in FLEX_POS])
        if forced:
            by_id = {p.player_id: p for p in forced}
            return min(by_id.values(), key=lambda p: p.adp).player_id

    return min(state.available.values(), key=lambda p: p.adp).player_id


# Apply one pick to mutable draft state.
def apply_pick(state: DraftState, team_idx: int, pid: str) -> None:
    p = state.available.get(pid)
    if p is None:
        raise ValueError(f"Player {pid} not available")
    state.rosters[team_idx].append(pid)
    state.pick_history_pos.append(p.pos)
    del state.available[pid]
    state.pick_no += 1


# Return PPR points for one player/week from stats_weekly dict.
def player_week_points(pid: str, week: int, stats_weekly: Dict[str, Any]) -> float:
    per = stats_weekly.get(pid, {})
    wd = per.get(str(week))
    if not isinstance(wd, dict):
        return 0.0
    return weekly_points_ppr(wd.get("stats") or {})


# Compute best-starting-lineup points for one roster/week.
def weekly_best_lineup_points(roster: List[str], universe: Dict[str, Player], stats_weekly: Dict[str, Any], week: int, s: LeagueSettings) -> float:
    by_pos: Dict[str, List[float]] = {p: [] for p in ("QB", "RB", "WR", "TE", "K", "DEF")}
    for pid in roster:
        p = universe.get(pid)
        if not p:
            continue
        by_pos[p.pos].append(player_week_points(pid, week, stats_weekly))
    for p in by_pos:
        by_pos[p].sort(reverse=True)
    total = 0.0
    total += sum(by_pos["QB"][: s.slot_qb])
    total += sum(by_pos["RB"][: s.slot_rb])
    total += sum(by_pos["WR"][: s.slot_wr])
    total += sum(by_pos["TE"][: s.slot_te])
    total += sum(by_pos["K"][: s.slot_k])
    total += sum(by_pos["DEF"][: s.slot_def])
    flex_pool = by_pos["RB"][s.slot_rb :] + by_pos["WR"][s.slot_wr :] + by_pos["TE"][s.slot_te :]
    flex_pool.sort(reverse=True)
    total += sum(flex_pool[: s.slot_flex])
    return total


# One 12-team round-robin schedule (11 weeks), reused cyclically.
def build_round_robin_12() -> List[List[Tuple[int, int]]]:
    matchups_by_week: List[List[Tuple[int, int]]] = []
    for r in range(11):
        opp = (r + 1) % 11 + 1
        rest = [i for i in range(1, 12) if i != opp]
        pairs = [(rest[i], rest[i + 5]) for i in range(5)]
        matchups_by_week.append([(0, opp)] + pairs)
    return matchups_by_week


# Simulate season outcomes from completed draft state.
def run_season(state: DraftState, universe: Dict[str, Player], stats_weekly: Dict[str, Any], settings: LeagueSettings) -> Tuple[Dict[int, float], Dict[int, int], Dict[int, int]]:
    rr = build_round_robin_12()
    total_pts = {i: 0.0 for i in range(settings.num_teams)}
    wins = {i: 0 for i in range(settings.num_teams)}
    losses = {i: 0 for i in range(settings.num_teams)}
    for week in range(1, settings.season_weeks + 1):
        matchups = rr[(week - 1) % len(rr)]
        week_pts: Dict[int, float] = {}
        for t in range(settings.num_teams):
            week_pts[t] = weekly_best_lineup_points(state.rosters[t], universe, stats_weekly, week, settings)
            total_pts[t] += week_pts[t]
        for a, b in matchups:
            if week_pts[a] > week_pts[b]:
                wins[a] += 1
                losses[b] += 1
            elif week_pts[b] > week_pts[a]:
                wins[b] += 1
                losses[a] += 1
    return total_pts, wins, losses


# Rank teams by points (wins as tiebreaker).
def rank_by_points(total_pts: Dict[int, float], wins: Dict[int, int], num_teams: int) -> Dict[int, int]:
    order = list(range(num_teams))
    order.sort(key=lambda t: (-total_pts[t], -wins[t]))
    return {tid: i + 1 for i, tid in enumerate(order)}


# Rank teams by record (wins desc, losses asc, points desc).
def rank_by_record(total_pts: Dict[int, float], wins: Dict[int, int], losses: Dict[int, int], num_teams: int) -> Dict[int, int]:
    order = list(range(num_teams))
    order.sort(key=lambda t: (-wins[t], losses[t], -total_pts[t]))
    return {tid: i + 1 for i, tid in enumerate(order)}


@dataclass
class SimMetrics:
    rank_pts: int
    rank_record: int
    wins: int
    losses: int
    points: float


# Run one full draft where team 0 uses the model and others use ADP policy.
def run_one_draft(universe: Dict[str, Player], settings: LeagueSettings, prefs: Prefs, order: List[int], candidates_k: int) -> DraftState:
    state = DraftState(
        settings=settings,
        rosters={i: [] for i in range(settings.num_teams)},
        available=copy.deepcopy(universe),
        pick_no=1,
        pick_history_pos=[],
        order=order,
    )
    ensure_model_loaded()
    for team_idx in order:
        if team_idx == AI_TEAM_IDX:
            recs = recommend_top(state, AI_TEAM_IDX, universe, prefs, candidates_k=candidates_k, limit=1)
            pid = recs[0]["player_id"] if recs else next(iter(state.available.keys()))
        else:
            pid = adp_pick_with_roster_constraints(state, team_idx)
        apply_pick(state, team_idx, pid)
    return state


# CLI entrypoint for single-run or multi-run simulations.
def main() -> None:
    ap = argparse.ArgumentParser(description="Run draft simulation with model team vs ADP bots.")
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--top-n", type=int, default=500)
    ap.add_argument("--candidates-k", type=int, default=60)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--pref-risk", type=float, default=0.5)
    ap.add_argument("--w-qb", type=float, default=4.0)
    ap.add_argument("--w-rb", type=float, default=6.0)
    ap.add_argument("--w-wr", type=float, default=5.0)
    ap.add_argument("--w-te", type=float, default=3.0)
    ap.add_argument("--w-k", type=float, default=2.0)
    ap.add_argument("--w-def", type=float, default=1.0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    settings = LeagueSettings()
    universe = load_universe(args.season, args.top_n, settings)
    stats_weekly = load_json(Path("sleeper_exports") / f"stats_weekly_{args.season}_players_500.json")
    prefs = Prefs(
        risk_factor=args.pref_risk,
        w_qb=args.w_qb,
        w_rb=args.w_rb,
        w_wr=args.w_wr,
        w_te=args.w_te,
        w_k=args.w_k,
        w_def=args.w_def,
    )

    metrics: List[SimMetrics] = []
    for i in range(args.runs):
        order = build_snake_order(settings, random.Random(rng.getrandbits(32)))
        state = run_one_draft(universe, settings, prefs, order, args.candidates_k)
        total_pts, wins, losses = run_season(state, universe, stats_weekly, settings)
        rp = rank_by_points(total_pts, wins, settings.num_teams)[AI_TEAM_IDX]
        rr = rank_by_record(total_pts, wins, losses, settings.num_teams)[AI_TEAM_IDX]
        m = SimMetrics(
            rank_pts=rp,
            rank_record=rr,
            wins=wins[AI_TEAM_IDX],
            losses=losses[AI_TEAM_IDX],
            points=total_pts[AI_TEAM_IDX],
        )
        metrics.append(m)
        if args.verbose:
            print(
                f"run {i + 1}/{args.runs}: rank_pts={m.rank_pts} rank_rec={m.rank_record} "
                f"record={m.wins}-{m.losses} pts={m.points:.2f}"
            )

    if args.runs == 1:
        m = metrics[0]
        print(f"Model team: rank_pts={m.rank_pts}/12 rank_rec={m.rank_record}/12 record={m.wins}-{m.losses} pts={m.points:.2f}")
        return

    n = len(metrics)
    print("\n=== Aggregate (team 0 / model) ===")
    print(f"Runs: {n}")
    print(f"Average finish position by total points (1=best): {sum(m.rank_pts for m in metrics)/n:.3f}")
    print(f"Average finish position by record (W-L, pts tiebreak): {sum(m.rank_record for m in metrics)/n:.3f}")
    print(f"Average record: {sum(m.wins for m in metrics)/n:.3f}-{sum(m.losses for m in metrics)/n:.3f}")
    print(f"Average season points: {sum(m.points for m in metrics)/n:.2f}")


if __name__ == "__main__":
    main()

