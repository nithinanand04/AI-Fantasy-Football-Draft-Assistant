#!/usr/bin/env python3
"""
Single draft + season simulation.

- Team 0 = AI (uses trained MLP ranker; top 200 by ADP as candidates).
- Teams 1..11 = ADP bots (pick by ADP, fill all positions by end).
- Draft position for team 0 is randomized each run.
- Season: 17 weeks, round-robin weeks 1–11, repeat for 12–17.
- Scoring: sleeper_exports/stats_weekly_2024_players_500.json (pts_ppr or recompute).
- Output: W-L, total points, league rank for each team.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

# Import draft mechanics and featurize from label_data
from data_scripts.get_data import compute_ppr_points_from_components
from data_scripts.label_data import (
    LeagueSettings,
    DraftState,
    Player,
    Prefs,
    apply_pick,
    build_player_universe,
    choose_pick_adp_need_noise,
    featurize,
    get_candidates,
    trim_universe_top_n_by_adp,
)
from model import MLPScorer

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "sleeper_exports"
DEFAULT_MODEL_PATH = ROOT / "model_mlp.pt"
DEFAULT_STATS_WEEKLY = OUT_DIR / "stats_weekly_2024_players_500.json"
SEASON = 2024
AI_TEAM_IDX = 0
CANDIDATES_K = 200
UNIVERSE_TOP_N = 300


def build_random_snake_order(settings: LeagueSettings, rng: random.Random) -> List[int]:
    """Snake order with randomized team positions (so AI can be any draft slot)."""
    base = list(range(settings.num_teams))
    rng.shuffle(base)
    order: List[int] = []
    for r in range(settings.rounds):
        order.extend(base if r % 2 == 0 else base[::-1])
    return order


def run_draft(
    universe: Dict[str, Player],
    settings: LeagueSettings,
    prefs: Prefs,
    draft_order: List[int],
    model: MLPScorer,
    feature_index: List[str],
    device: torch.device,
    rng: random.Random,
) -> DraftState:
    """Run one full draft. Team 0 uses model; others use ADP + need (no noise)."""
    state = DraftState(
        settings=settings,
        rosters={i: [] for i in range(settings.num_teams)},
        available=copy.deepcopy(universe),
        pick_no=1,
        pick_history_pos=[],
    )

    def adp_bot(st: DraftState, team_idx: int) -> str:
        return choose_pick_adp_need_noise(st, team_idx, universe, noise_std=0.0)

    for team_idx in draft_order:
        if team_idx == AI_TEAM_IDX:
            cand_ids = get_candidates(state, k=CANDIDATES_K)
            if not cand_ids:
                pid = next(iter(state.available.keys()))
            else:
                vecs: List[List[float]] = []
                for cid in cand_ids:
                    _, vec = featurize(state, AI_TEAM_IDX, cid, universe, prefs, order=draft_order)
                    vecs.append(vec)
                X = torch.tensor(vecs, dtype=torch.float32, device=device)
                with torch.no_grad():
                    scores = model(X)
                best_idx = int(scores.argmax().item())
                pid = cand_ids[best_idx]
        else:
            pid = adp_bot(state, team_idx)
        apply_pick(state, team_idx, pid)

    return state


def get_player_week_pts(
    player_id: str,
    week: int,
    stats_weekly: Dict[str, Any],
    players_map: Dict[str, Any],
) -> float:
    """PPR points for one player in one week. Use pts_ppr if present else recompute."""
    data = stats_weekly.get(str(player_id))
    if not data:
        return 0.0
    week_data = data.get(str(week))
    if not week_data:
        return 0.0
    st = week_data.get("stats") or {}
    pts = st.get("pts_ppr")
    if pts is not None and isinstance(pts, (int, float)):
        return float(pts)
    pts = st.get("pts")
    if pts is not None and isinstance(pts, (int, float)):
        return float(pts)
    return compute_ppr_points_from_components(st, ppr=1.0)


def best_starters_weekly_pts(
    roster: List[str],
    week: int,
    stats_weekly: Dict[str, Any],
    players_map: Dict[str, Any],
    pos_by_pid: Dict[str, str],
    s: LeagueSettings,
) -> float:
    """Total starter PPR points for this roster in this week (best lineup)."""
    by_pos: Dict[str, List[float]] = {pos: [] for pos in ("QB", "RB", "WR", "TE", "K", "DEF")}
    for pid in roster:
        pos = pos_by_pid.get(pid)
        if not pos or pos not in by_pos:
            continue
        pts = get_player_week_pts(pid, week, stats_weekly, players_map)
        by_pos[pos].append(pts)
    for pos in by_pos:
        by_pos[pos].sort(reverse=True)

    total = 0.0
    total += sum(by_pos["QB"][: s.slot_qb])
    total += sum(by_pos["RB"][: s.slot_rb])
    total += sum(by_pos["WR"][: s.slot_wr])
    total += sum(by_pos["TE"][: s.slot_te])
    total += sum(by_pos["K"][: s.slot_k])
    total += sum(by_pos["DEF"][: s.slot_def])
    flex_pool = []
    flex_pool += by_pos["RB"][s.slot_rb :]
    flex_pool += by_pos["WR"][s.slot_wr :]
    flex_pool += by_pos["TE"][s.slot_te :]
    flex_pool.sort(reverse=True)
    total += sum(flex_pool[: s.slot_flex])
    return total


def build_round_robin_11(num_teams: int = 12) -> List[List[Tuple[int, int]]]:
    """One full round-robin: 11 weeks, 6 matchups per week. Returns list of 11 weeks, each a list of (a, b) pairs."""
    matchups_by_week: List[List[Tuple[int, int]]] = []
    for r in range(11):
        opp = (r + 1) % 11 + 1  # 0 plays 1,2,...,11
        rest = [i for i in range(1, 12) if i != opp]
        pairs = [(rest[i], rest[i + 5]) for i in range(5)]
        week_matchups = [(0, opp)] + pairs
        matchups_by_week.append(week_matchups)
    return matchups_by_week


def run_season(
    state: DraftState,
    universe: Dict[str, Player],
    stats_weekly: Dict[str, Any],
    players_map: Dict[str, Any],
    settings: LeagueSettings,
) -> Tuple[Dict[int, float], Dict[int, int], Dict[int, int]]:
    """Returns (total_pts_by_team, wins_by_team, losses_by_team)."""
    all_drafted = [pid for roster in state.rosters.values() for pid in roster]
    pos_by_pid = {pid: universe[pid].pos for pid in all_drafted if pid in universe}

    total_pts: Dict[int, float] = {i: 0.0 for i in range(settings.num_teams)}
    wins: Dict[int, int] = {i: 0 for i in range(settings.num_teams)}
    losses: Dict[int, int] = {i: 0 for i in range(settings.num_teams)}

    schedule_weeks_1_11 = build_round_robin_11(settings.num_teams)
    # schedule_weeks_1_11[r] = list of (a, b) for week r+1
    # Weeks 12–17: repeat weeks 0–5
    for week in range(1, 18):
        week_idx = (week - 1) % 11
        matchups = schedule_weeks_1_11[week_idx]
        week_pts: Dict[int, float] = {}
        for team_idx in range(settings.num_teams):
            roster = state.rosters[team_idx]
            week_pts[team_idx] = best_starters_weekly_pts(
                roster, week, stats_weekly, players_map, pos_by_pid, settings
            )
            total_pts[team_idx] += week_pts[team_idx]
        for (a, b) in matchups:
            if week_pts[a] > week_pts[b]:
                wins[a] += 1
                losses[b] += 1
            elif week_pts[b] > week_pts[a]:
                wins[b] += 1
                losses[a] += 1

    return total_pts, wins, losses


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Run one draft + season simulation.")
    ap.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH), help="Path to model_mlp.pt")
    ap.add_argument("--stats", type=str, default=str(DEFAULT_STATS_WEEKLY), help="Path to stats_weekly JSON")
    ap.add_argument("--seed", type=int, default=None, help="Random seed (default: random)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    feature_index = ckpt["feature_index"]
    model = MLPScorer(input_dim=ckpt["input_dim"], hidden_dim=ckpt["hidden_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    # Universe and settings
    universe = build_player_universe(SEASON)
    universe = trim_universe_top_n_by_adp(universe, top_n=UNIVERSE_TOP_N)
    settings = LeagueSettings()
    prefs = Prefs()
    players_map = json.loads((OUT_DIR / "players_nfl.json").read_text())

    # Random draft order
    draft_order = build_random_snake_order(settings, rng)
    ai_draft_slot = draft_order.index(AI_TEAM_IDX) % 12 + 1  # 1-based "pick number in first round"
    print(f"AI (team 0) draft slot: {ai_draft_slot} (of 12)")
    print("Running draft...")

    state = run_draft(universe, settings, prefs, draft_order, model, feature_index, device, rng)

    # Load weekly stats
    stats_path = Path(args.stats)
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")
    stats_weekly = json.loads(stats_path.read_text())

    # Diagnostic: roster position counts (AI vs typical bot)
    def pos_counts(roster: List[str]) -> Dict[str, int]:
        c = {"QB": 0, "RB": 0, "WR": 0, "TE": 0, "K": 0, "DEF": 0}
        for pid in roster:
            if pid in universe:
                c[universe[pid].pos] = c.get(universe[pid].pos, 0) + 1
        return c

    print("\n--- Roster position counts (QB/RB/WR/TE/K/DEF) ---")
    for tid in range(settings.num_teams):
        counts = pos_counts(state.rosters[tid])
        note = " (AI)" if tid == AI_TEAM_IDX else ""
        print(f"  Team {tid}{note}: {counts}")

    # Diagnostic: how many of AI's drafted players appear in stats_weekly (missing = 0 pts all season)
    ai_pids = state.rosters[AI_TEAM_IDX]
    in_stats = sum(1 for pid in ai_pids if str(pid) in stats_weekly)
    print(f"\n  AI roster: {in_stats}/{len(ai_pids)} players have data in stats_weekly (missing => 0 pts).")

    print("\nRunning season (17 weeks)...")
    total_pts, wins, losses = run_season(state, universe, stats_weekly, players_map, settings)

    # Rank by total points (tiebreak: wins)
    standings = list(range(settings.num_teams))
    standings.sort(key=lambda t: (-total_pts[t], -wins[t]))

    print("\n--- League standings ---")
    print(f"{'Rank':<6} {'Team':<8} {'W':<4} {'L':<4} {'TP':<10} {'Note'}")
    print("-" * 50)
    for rank, team_idx in enumerate(standings, 1):
        note = " (AI)" if team_idx == AI_TEAM_IDX else ""
        print(f"{rank:<6} {team_idx:<8} {wins[team_idx]:<4} {losses[team_idx]:<4} {total_pts[team_idx]:<10.1f}{note}")
    ai_rank = standings.index(AI_TEAM_IDX) + 1
    print(f"\nAI team finished rank {ai_rank} with {wins[AI_TEAM_IDX]}-{losses[AI_TEAM_IDX]} record, {total_pts[AI_TEAM_IDX]:.1f} total points.")


if __name__ == "__main__":
    main()
