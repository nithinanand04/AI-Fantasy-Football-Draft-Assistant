#!/usr/bin/env python3
"""
labelgen_v1_mvp.py  (UPDATED)

Fixes applied vs prior version:
1) weekly_summaries(): worst_week_index_norm no longer defaults to week 1 when all weeks tie
   - if std == 0 -> worst_week_index_norm = 0.0 (neutral)
2) bench_spots_remaining: now correctly means "bench slots remaining"
   - also adds total_spots_remaining as a separate, useful feature

Everything else remains MVP:
- 12 teams, 15 rounds, snake
- Starters: 1QB/2RB/2WR/1TE/1FLEX/1K/1DEF, Bench 6
- y via rollout reward = projected starter season value (season_value = 17 * week1 pts_ppr)
- Bye-week features computed via weekly proxy + bye=0
- Volatility placeholders remain 0.0 (plug in later)

Inputs expected:
  sleeper_exports/players_nfl.json
  sleeper_exports/projections_{SEASON}_week1.json

Outputs:
  sleeper_exports/train_rows.jsonl
  sleeper_exports/feature_index.json
  sleeper_exports/meta.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

OUT_DIR = Path("sleeper_exports")

POS = ("QB", "RB", "WR", "TE", "K", "DEF")
FLEX_ELIGIBLE = {"RB", "WR", "TE"}


# -----------------------------
# Data structures
# -----------------------------
@dataclass(frozen=True)
class LeagueSettings:
    num_teams: int = 12
    rounds: int = 15
    ppr_points_per_reception: float = 1.0
    superflex: int = 0

    slot_qb: int = 1
    slot_rb: int = 2
    slot_wr: int = 2
    slot_te: int = 1
    slot_flex: int = 1
    slot_superflex: int = 0
    slot_k: int = 1
    slot_def: int = 1
    slot_bench: int = 6

    is_snake: int = 1
    season_weeks: int = 17


@dataclass(frozen=True)
class Prefs:
    bye_safety_weight: float = 0.6
    risk_tolerance: float = 0.5
    reach_penalty_weight: float = 0.5
    stack_weight: float = 0.0

    w_qb: float = 1.0
    w_rb: float = 1.0
    w_wr: float = 1.0
    w_te: float = 1.0
    w_k: float = 0.6
    w_def: float = 0.6


@dataclass(frozen=True)
class Player:
    player_id: str
    pos: str
    team: Optional[str]
    bye_week: int
    adp: float
    week1_pts_ppr: float
    season_value: float

    # placeholders for later
    weekly_std_proxy: float = 0.0
    boom_rate: float = 0.0
    bust_rate: float = 0.0
    pos_adp_rank: float = -1.0


@dataclass
class DraftState:
    settings: LeagueSettings
    rosters: Dict[int, List[str]]
    available: Dict[str, Player]
    pick_no: int  # 1-indexed overall pick before the next pick happens
    pick_history_pos: List[str]


# -----------------------------
# IO helpers
# -----------------------------
def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


# -----------------------------
# Universe build
# -----------------------------
def build_player_universe(season: int) -> Dict[str, Player]:
    players_map = load_json(OUT_DIR / "players_nfl.json")
    proj_rows = load_json(OUT_DIR / f"projections_{season}_week1.json")

    universe: Dict[str, Player] = {}

    for row in proj_rows:
        pid = row.get("player_id")
        if pid is None:
            continue
        pid = str(pid)

        embedded_player = row.get("player") or {}
        pos = embedded_player.get("position") or (players_map.get(pid, {}).get("position") if pid in players_map else None)
        if pos not in POS:
            continue

        stats = row.get("stats") or {}
        pts_ppr = float(stats.get("pts_ppr") or 0.0)

        adp_raw = stats.get("adp_dd_ppr") or stats.get("adp_dd_std")
        if adp_raw is None:
            # K/DEF often lack PPR ADP in Sleeper; include with late-round default so they're in the pool
            if pos == "K":
                adp_raw = 165.0
            elif pos == "DEF":
                adp_raw = 155.0
            else:
                continue
        adp = float(adp_raw)
        if not math.isfinite(adp):
            continue

        team = embedded_player.get("team") or players_map.get(pid, {}).get("team")

        meta = players_map.get(pid, {}).get("metadata") or {}
        bye_week = _safe_int(meta.get("bye_week", 0), 0)

        # drop pure zeros to reduce noise (if you want K/DEF included, they must have pts)
        if pts_ppr <= 0.0:
            continue

        universe[pid] = Player(
            player_id=pid,
            pos=pos,
            team=team,
            bye_week=bye_week,
            adp=adp,
            week1_pts_ppr=pts_ppr,
            season_value=17.0 * pts_ppr,
            pos_adp_rank=float(stats.get("pos_adp_dd_ppr") or -1.0),
        )

    return universe


def trim_universe_top_n_by_adp(universe: Dict[str, Player], top_n: int) -> Dict[str, Player]:
    items = sorted(universe.items(), key=lambda kv: kv[1].adp)[:top_n]
    return {pid: p for pid, p in items}


# -----------------------------
# Draft mechanics
# -----------------------------
def snake_order(settings: LeagueSettings) -> List[int]:
    order: List[int] = []
    for r in range(settings.rounds):
        if r % 2 == 0:
            order.extend(list(range(settings.num_teams)))
        else:
            order.extend(list(range(settings.num_teams - 1, -1, -1)))
    return order


def picks_until_next_turn(order: List[int], current_pick_no: int, user_team_idx: int) -> int:
    # current_pick_no is the pick we are about to take (1-indexed).
    # After we take it, next picks begin at index current_pick_no (0-based).
    start = current_pick_no
    for j in range(start, len(order)):
        if order[j] == user_team_idx:
            return j - (current_pick_no - 1)
    return 0


def roster_counts(roster: List[str], universe_all: Dict[str, Player]) -> Dict[str, int]:
    counts = {pos: 0 for pos in POS}
    for pid in roster:
        p = universe_all.get(pid)
        if p:
            counts[p.pos] += 1
    return counts


def drafted_totals(state: DraftState) -> Dict[str, int]:
    totals = {pos: 0 for pos in POS}
    for pos in state.pick_history_pos:
        if pos in totals:
            totals[pos] += 1
    return totals


def recent_rates(state: DraftState, window: int = 10) -> Dict[str, float]:
    recent = state.pick_history_pos[-window:]
    rates = {pos: 0.0 for pos in POS}
    if not recent:
        return rates
    for pos in recent:
        if pos in rates:
            rates[pos] += 1.0
    for pos in POS:
        rates[pos] /= float(len(recent))
    return rates


def choose_pick_adp_need_noise(
    state: DraftState,
    team_idx: int,
    universe_all: Dict[str, Player],
    noise_std: float = 2.0,
    need_weight: float = 3.0,
) -> str:
    s = state.settings
    roster = state.rosters[team_idx]
    counts = roster_counts(roster, universe_all)

    needs = {
        "QB": max(0, s.slot_qb - counts["QB"]),
        "RB": max(0, s.slot_rb - counts["RB"]),
        "WR": max(0, s.slot_wr - counts["WR"]),
        "TE": max(0, s.slot_te - counts["TE"]),
        "K": max(0, s.slot_k - counts["K"]),
        "DEF": max(0, s.slot_def - counts["DEF"]),
    }

    best_pid = None
    best_score = float("inf")

    for p in state.available.values():
        score = p.adp + random.gauss(0.0, noise_std)
        if needs.get(p.pos, 0) > 0:
            score -= need_weight
        if score < best_score:
            best_score = score
            best_pid = p.player_id

    return best_pid or next(iter(state.available.keys()))


def apply_pick(state: DraftState, team_idx: int, player_id: str) -> None:
    p = state.available.get(player_id)
    if p is None:
        raise ValueError(f"Player {player_id} not available.")
    state.rosters[team_idx].append(player_id)
    state.pick_history_pos.append(p.pos)
    del state.available[player_id]
    state.pick_no += 1


def simulate_one_draft(
    universe: Dict[str, Player],
    settings: LeagueSettings,
    user_team_idx: int,
    opponent_policy,
    user_policy,
) -> Tuple[DraftState, List[DraftState]]:
    state = DraftState(
        settings=settings,
        rosters={i: [] for i in range(settings.num_teams)},
        available=universe.copy(),
        pick_no=1,
        pick_history_pos=[],
    )

    order = snake_order(settings)
    user_snapshots: List[DraftState] = []

    for team_idx in order:
        if team_idx == user_team_idx:
            user_snapshots.append(DraftState(
                settings=state.settings,
                rosters={k: v.copy() for k, v in state.rosters.items()},
                available=state.available.copy(),
                pick_no=state.pick_no,
                pick_history_pos=state.pick_history_pos.copy(),
            ))
            pid = user_policy(state, team_idx)
        else:
            pid = opponent_policy(state, team_idx)

        apply_pick(state, team_idx, pid)

    return state, user_snapshots


# -----------------------------
# Reward
# -----------------------------
def best_starters_season_value(roster: List[str], universe_all: Dict[str, Player], s: LeagueSettings) -> float:
    by_pos: Dict[str, List[float]] = {pos: [] for pos in POS}
    for pid in roster:
        p = universe_all.get(pid)
        if p:
            by_pos[p.pos].append(p.season_value)
    for pos in POS:
        by_pos[pos].sort(reverse=True)

    total = 0.0
    total += sum(by_pos["QB"][:s.slot_qb])
    total += sum(by_pos["RB"][:s.slot_rb])
    total += sum(by_pos["WR"][:s.slot_wr])
    total += sum(by_pos["TE"][:s.slot_te])
    total += sum(by_pos["K"][:s.slot_k])
    total += sum(by_pos["DEF"][:s.slot_def])

    flex_pool = []
    flex_pool += by_pos["RB"][s.slot_rb:]
    flex_pool += by_pos["WR"][s.slot_wr:]
    flex_pool += by_pos["TE"][s.slot_te:]
    flex_pool.sort(reverse=True)
    total += sum(flex_pool[:s.slot_flex])
    return total


def reward(roster: List[str], universe_all: Dict[str, Player], settings: LeagueSettings) -> float:
    return best_starters_season_value(roster, universe_all, settings)


# -----------------------------
# Weekly summaries (bye safety)
# -----------------------------
def weekly_starter_points(roster: List[str], universe_all: Dict[str, Player], s: LeagueSettings, week: int) -> float:
    by_pos_pts: Dict[str, List[float]] = {pos: [] for pos in POS}
    for pid in roster:
        p = universe_all.get(pid)
        if not p:
            continue
        pts = p.week1_pts_ppr
        if p.bye_week == week and week != 0:
            pts = 0.0
        by_pos_pts[p.pos].append(pts)

    for pos in POS:
        by_pos_pts[pos].sort(reverse=True)

    total = 0.0
    total += sum(by_pos_pts["QB"][:s.slot_qb])
    total += sum(by_pos_pts["RB"][:s.slot_rb])
    total += sum(by_pos_pts["WR"][:s.slot_wr])
    total += sum(by_pos_pts["TE"][:s.slot_te])
    total += sum(by_pos_pts["K"][:s.slot_k])
    total += sum(by_pos_pts["DEF"][:s.slot_def])

    flex_pool = []
    flex_pool += by_pos_pts["RB"][s.slot_rb:]
    flex_pool += by_pos_pts["WR"][s.slot_wr:]
    flex_pool += by_pos_pts["TE"][s.slot_te:]
    flex_pool.sort(reverse=True)
    total += sum(flex_pool[:s.slot_flex])
    return total


def team_weekly_vector(roster: List[str], universe_all: Dict[str, Player], s: LeagueSettings) -> List[float]:
    return [weekly_starter_points(roster, universe_all, s, week=w) for w in range(1, s.season_weeks + 1)]


def percentile(vals: List[float], p: float) -> float:
    if not vals:
        return 0.0
    vs = sorted(vals)
    k = (len(vs) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return vs[int(k)]
    return vs[f] * (c - k) + vs[c] * (k - f)


def weekly_summaries(T: List[float]) -> Tuple[float, float, float, float, int, float, float]:
    if not T:
        return (0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0)

    mean = sum(T) / len(T)
    var = sum((x - mean) ** 2 for x in T) / len(T)
    std = math.sqrt(var)
    mn = min(T)
    p10 = percentile(T, 0.10)

    threshold = mean - 0.75 * std
    holes = sum(1 for x in T if x < threshold)

    worst_val = mn
    worst_idx_norm = 0.0 if std == 0.0 else (T.index(mn) + 1) / 17.0
    return (mean, std, mn, p10, holes, worst_val, worst_idx_norm)


# -----------------------------
# Candidate selection & scarcity
# -----------------------------
def get_candidates(state: DraftState, k: int = 60) -> List[str]:
    avail = sorted(state.available.values(), key=lambda p: p.adp)
    return [p.player_id for p in avail[:k]]


def pos_remaining_topN(state: DraftState, pos: str, top_n: int = 100) -> int:
    avail = sorted(state.available.values(), key=lambda p: p.adp)[:top_n]
    return sum(1 for p in avail if p.pos == pos)


def pos_dropoff_estimate(state: DraftState, pos: str, n_until_next: int) -> float:
    avail_sorted = sorted(state.available.values(), key=lambda p: p.adp)
    best_now = max((p.season_value for p in avail_sorted if p.pos == pos), default=0.0)

    remaining = avail_sorted[n_until_next:] if n_until_next < len(avail_sorted) else []
    best_after = max((p.season_value for p in remaining if p.pos == pos), default=0.0)

    return best_now - best_after


# -----------------------------
# Feature index
# -----------------------------
def build_feature_index() -> List[str]:
    names: List[str] = []

    # A) League settings
    names += [
        "num_teams", "num_rounds", "ppr_points_per_reception", "superflex",
        "slot_qb", "slot_rb", "slot_wr", "slot_te", "slot_flex", "slot_superflex",
        "slot_k", "slot_def", "slot_bench", "is_snake", "season_weeks"
    ]

    # B) Draft context
    names += [
        "pick_no", "round_no", "pick_in_round", "picks_until_next_turn", "drafted_total",
        "drafted_qb_total", "drafted_rb_total", "drafted_wr_total", "drafted_te_total", "drafted_k_total", "drafted_def_total",
        "recent_qb_rate", "recent_rb_rate", "recent_wr_rate", "recent_te_rate", "recent_k_rate", "recent_def_rate",
        "best_remaining_adp", "best_remaining_proj", "remaining_players_count",
    ]

    # C) Roster state
    names += [
        "my_qb_count", "my_rb_count", "my_wr_count", "my_te_count", "my_k_count", "my_def_count", "my_total_count",
        "qb_needed", "rb_needed", "wr_needed", "te_needed", "k_needed", "def_needed",
        "flex_needed_est", "superflex_needed_est",
        "bench_spots_remaining",
        "total_spots_remaining",
        "rb_wr_balance", "early_rounds_remaining",
    ]

    # D) Team weekly summaries
    names += [
        "team_weekly_mean", "team_weekly_std", "team_weekly_min", "team_weekly_p10",
        "team_bye_hole_count", "team_worst_week_value", "team_worst_week_index_norm",
    ]

    # E) Prefs
    names += [
        "pref_bye_safety_weight", "pref_risk_tolerance", "pref_reach_penalty_weight", "pref_stack_weight",
        "pref_w_qb", "pref_w_rb", "pref_w_wr", "pref_w_te", "pref_w_k", "pref_w_def",
    ]

    # F) Candidate identity
    names += [
        "cand_pos_is_qb", "cand_pos_is_rb", "cand_pos_is_wr", "cand_pos_is_te", "cand_pos_is_k", "cand_pos_is_def",
    ]

    # G) Candidate value + volatility
    names += [
        "cand_adp", "cand_pos_adp_rank",
        "cand_proj_week1_pts_ppr", "cand_proj_season_value",
        "cand_bye_week",
        "cand_weekly_std_proxy", "cand_boom_rate", "cand_bust_rate",
    ]

    # H) Interaction/scarcity/fit/bye deltas
    names += [
        "cand_adp_delta",
        "cand_fills_required_starter",
        "cand_pos_dropoff",
        "cand_pos_remaining_topN",
        "cand_delta_min_week",
        "cand_delta_bye_hole_count",
        "cand_delta_team_weekly_std",
    ]

    return names


FEATURE_INDEX = build_feature_index()
FEATURE_SET = set(FEATURE_INDEX)


# -----------------------------
# Featurize
# -----------------------------
def featurize(
    pre_pick_state: DraftState,
    user_team_idx: int,
    candidate_id: str,
    universe_all: Dict[str, Player],
    prefs: Prefs,
    order: Optional[List[int]] = None,
) -> Tuple[Dict[str, float], List[float]]:
    s = pre_pick_state.settings
    order = order if order is not None else snake_order(s)

    pick_no = pre_pick_state.pick_no
    round_no = (pick_no - 1) // s.num_teams + 1
    pick_in_round = (pick_no - 1) % s.num_teams + 1
    p_next = picks_until_next_turn(order, pick_no, user_team_idx)

    totals = drafted_totals(pre_pick_state)
    rates = recent_rates(pre_pick_state, window=10)

    best_remaining_adp = min((p.adp for p in pre_pick_state.available.values()), default=9999.0)
    best_remaining_proj = max((p.season_value for p in pre_pick_state.available.values()), default=0.0)
    remaining_players_count = float(len(pre_pick_state.available))

    roster = pre_pick_state.rosters[user_team_idx]
    counts = roster_counts(roster, universe_all)
    my_total = int(len(roster))

    qb_needed = max(0, s.slot_qb - counts["QB"])
    rb_needed = max(0, s.slot_rb - counts["RB"])
    wr_needed = max(0, s.slot_wr - counts["WR"])
    te_needed = max(0, s.slot_te - counts["TE"])
    k_needed = max(0, s.slot_k - counts["K"])
    def_needed = max(0, s.slot_def - counts["DEF"])

    overflow = max(0, counts["RB"] - s.slot_rb) + max(0, counts["WR"] - s.slot_wr) + max(0, counts["TE"] - s.slot_te)
    flex_needed_est = max(0, s.slot_flex - overflow)
    superflex_needed_est = 0

    starter_slots = s.slot_qb + s.slot_rb + s.slot_wr + s.slot_te + s.slot_flex + s.slot_k + s.slot_def
    total_roster_size = starter_slots + s.slot_bench
    total_spots_remaining = max(0, total_roster_size - my_total)

    bench_filled = max(0, my_total - starter_slots)
    bench_spots_remaining = max(0, s.slot_bench - bench_filled)

    rb_wr_balance = float(counts["RB"] - counts["WR"])
    early_rounds_remaining = float(max(0, 6 - round_no))

    # Team weekly summaries
    T = team_weekly_vector(roster, universe_all, s)
    (t_mean, t_std, t_min, t_p10, t_holes, t_worst, t_worst_idx_norm) = weekly_summaries(T)

    cand = universe_all[candidate_id]

    cand_pos_dropoff = pos_dropoff_estimate(pre_pick_state, cand.pos, p_next)
    cand_pos_remaining = float(pos_remaining_topN(pre_pick_state, cand.pos, top_n=100))

    fills_required = 0.0
    if cand.pos == "QB" and qb_needed > 0: fills_required = 1.0
    if cand.pos == "RB" and rb_needed > 0: fills_required = 1.0
    if cand.pos == "WR" and wr_needed > 0: fills_required = 1.0
    if cand.pos == "TE" and te_needed > 0: fills_required = 1.0
    if cand.pos == "K" and k_needed > 0: fills_required = 1.0
    if cand.pos == "DEF" and def_needed > 0: fills_required = 1.0

    roster_plus = roster + [candidate_id]
    T2 = team_weekly_vector(roster_plus, universe_all, s)
    (t2_mean, t2_std, t2_min, t2_p10, t2_holes, t2_worst, t2_worst_idx_norm) = weekly_summaries(T2)

    feats: Dict[str, float] = {}

    # A
    feats.update({
        "num_teams": float(s.num_teams),
        "num_rounds": float(s.rounds),
        "ppr_points_per_reception": float(s.ppr_points_per_reception),
        "superflex": float(s.superflex),
        "slot_qb": float(s.slot_qb),
        "slot_rb": float(s.slot_rb),
        "slot_wr": float(s.slot_wr),
        "slot_te": float(s.slot_te),
        "slot_flex": float(s.slot_flex),
        "slot_superflex": float(s.slot_superflex),
        "slot_k": float(s.slot_k),
        "slot_def": float(s.slot_def),
        "slot_bench": float(s.slot_bench),
        "is_snake": float(s.is_snake),
        "season_weeks": float(s.season_weeks),
    })

    # B
    feats.update({
        "pick_no": float(pick_no),
        "round_no": float(round_no),
        "pick_in_round": float(pick_in_round),
        "picks_until_next_turn": float(p_next),
        "drafted_total": float(pick_no - 1),
        "drafted_qb_total": float(totals["QB"]),
        "drafted_rb_total": float(totals["RB"]),
        "drafted_wr_total": float(totals["WR"]),
        "drafted_te_total": float(totals["TE"]),
        "drafted_k_total": float(totals["K"]),
        "drafted_def_total": float(totals["DEF"]),
        "recent_qb_rate": float(rates["QB"]),
        "recent_rb_rate": float(rates["RB"]),
        "recent_wr_rate": float(rates["WR"]),
        "recent_te_rate": float(rates["TE"]),
        "recent_k_rate": float(rates["K"]),
        "recent_def_rate": float(rates["DEF"]),
        "best_remaining_adp": float(best_remaining_adp),
        "best_remaining_proj": float(best_remaining_proj),
        "remaining_players_count": float(remaining_players_count),
    })

    # C
    feats.update({
        "my_qb_count": float(counts["QB"]),
        "my_rb_count": float(counts["RB"]),
        "my_wr_count": float(counts["WR"]),
        "my_te_count": float(counts["TE"]),
        "my_k_count": float(counts["K"]),
        "my_def_count": float(counts["DEF"]),
        "my_total_count": float(my_total),
        "qb_needed": float(qb_needed),
        "rb_needed": float(rb_needed),
        "wr_needed": float(wr_needed),
        "te_needed": float(te_needed),
        "k_needed": float(k_needed),
        "def_needed": float(def_needed),
        "flex_needed_est": float(flex_needed_est),
        "superflex_needed_est": float(superflex_needed_est),
        "bench_spots_remaining": float(bench_spots_remaining),
        "total_spots_remaining": float(total_spots_remaining),
        "rb_wr_balance": float(rb_wr_balance),
        "early_rounds_remaining": float(early_rounds_remaining),
    })

    # D
    feats.update({
        "team_weekly_mean": float(t_mean),
        "team_weekly_std": float(t_std),
        "team_weekly_min": float(t_min),
        "team_weekly_p10": float(t_p10),
        "team_bye_hole_count": float(t_holes),
        "team_worst_week_value": float(t_worst),
        "team_worst_week_index_norm": float(t_worst_idx_norm),
    })

    # E
    feats.update({
        "pref_bye_safety_weight": float(prefs.bye_safety_weight),
        "pref_risk_tolerance": float(prefs.risk_tolerance),
        "pref_reach_penalty_weight": float(prefs.reach_penalty_weight),
        "pref_stack_weight": float(prefs.stack_weight),
        "pref_w_qb": float(prefs.w_qb),
        "pref_w_rb": float(prefs.w_rb),
        "pref_w_wr": float(prefs.w_wr),
        "pref_w_te": float(prefs.w_te),
        "pref_w_k": float(prefs.w_k),
        "pref_w_def": float(prefs.w_def),
    })

    # F
    feats.update({
        "cand_pos_is_qb": 1.0 if cand.pos == "QB" else 0.0,
        "cand_pos_is_rb": 1.0 if cand.pos == "RB" else 0.0,
        "cand_pos_is_wr": 1.0 if cand.pos == "WR" else 0.0,
        "cand_pos_is_te": 1.0 if cand.pos == "TE" else 0.0,
        "cand_pos_is_k": 1.0 if cand.pos == "K" else 0.0,
        "cand_pos_is_def": 1.0 if cand.pos == "DEF" else 0.0,
    })

    # G
    feats.update({
        "cand_adp": float(cand.adp),
        "cand_pos_adp_rank": float(cand.pos_adp_rank),
        "cand_proj_week1_pts_ppr": float(cand.week1_pts_ppr),
        "cand_proj_season_value": float(cand.season_value),
        "cand_bye_week": float(cand.bye_week),
        "cand_weekly_std_proxy": float(cand.weekly_std_proxy),
        "cand_boom_rate": float(cand.boom_rate),
        "cand_bust_rate": float(cand.bust_rate),
    })

    # H
    feats.update({
        "cand_adp_delta": float(cand.adp - pick_no),
        "cand_fills_required_starter": float(fills_required),
        "cand_pos_dropoff": float(cand_pos_dropoff),
        "cand_pos_remaining_topN": float(cand_pos_remaining),
        "cand_delta_min_week": float(t2_min - t_min),
        "cand_delta_bye_hole_count": float(t2_holes - t_holes),
        "cand_delta_team_weekly_std": float(t2_std - t_std),
    })

    missing = [k for k in FEATURE_INDEX if k not in feats]
    extra = [k for k in feats.keys() if k not in FEATURE_SET]
    if missing:
        raise RuntimeError(f"Missing features: {missing}")
    if extra:
        raise RuntimeError(f"Unexpected features: {extra}")

    vec = [float(feats[name]) for name in FEATURE_INDEX]
    return feats, vec


# -----------------------------
# Labeling
# -----------------------------
def label_state_with_rollouts(
    pre_pick_state: DraftState,
    user_team_idx: int,
    candidate_ids: List[str],
    universe_all: Dict[str, Player],
    settings: LeagueSettings,
    opponent_policy,
    user_policy_after,
    k_rollouts: int = 1,
) -> Dict[str, float]:
    y: Dict[str, float] = {}
    order = snake_order(settings)

    for cand in candidate_ids:
        rewards: List[float] = []
        for _ in range(k_rollouts):
            st = DraftState(
                settings=pre_pick_state.settings,
                rosters={k: v.copy() for k, v in pre_pick_state.rosters.items()},
                available=pre_pick_state.available.copy(),
                pick_no=pre_pick_state.pick_no,
                pick_history_pos=pre_pick_state.pick_history_pos.copy(),
            )

            apply_pick(st, user_team_idx, cand)

            start_idx = st.pick_no - 1
            for team_idx in order[start_idx:]:
                pid = user_policy_after(st, team_idx) if team_idx == user_team_idx else opponent_policy(st, team_idx)
                apply_pick(st, team_idx, pid)

            rewards.append(reward(st.rosters[user_team_idx], universe_all, settings))

        y[cand] = sum(rewards) / len(rewards)

    return y


# -----------------------------
# Dataset generation
# -----------------------------
def generate_dataset(
    universe: Dict[str, Player],
    settings: LeagueSettings,
    prefs: Prefs,
    user_team_idx: int,
    num_drafts: int,
    candidates_k: int,
    rollouts_k: int,
    sleep_s: float,
    out_jsonl: Path,
    seed: int,
    user_model_policy: Optional[Any] = None,
    user_model_prob: float = 0.0,
) -> None:
    random.seed(seed)

    opponent_policy = lambda st, tid: choose_pick_adp_need_noise(st, tid, universe)
    adp_policy = lambda st, tid: choose_pick_adp_need_noise(st, tid, universe)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w") as f:
        row_count = 0

        for d in range(num_drafts):
            use_model = (
                user_model_policy is not None
                and random.random() < user_model_prob
            )
            user_policy = user_model_policy if use_model else adp_policy
            user_policy_after = user_policy

            _, user_states = simulate_one_draft(
                universe=universe,
                settings=settings,
                user_team_idx=user_team_idx,
                opponent_policy=opponent_policy,
                user_policy=user_policy,
            )

            for s_state in user_states:
                cand_ids = get_candidates(s_state, k=candidates_k)

                y_map = label_state_with_rollouts(
                    pre_pick_state=s_state,
                    user_team_idx=user_team_idx,
                    candidate_ids=cand_ids,
                    universe_all=universe,
                    settings=settings,
                    opponent_policy=opponent_policy,
                    user_policy_after=user_policy_after,
                    k_rollouts=rollouts_k,
                )

                state_id = f"draft{d}_pick{s_state.pick_no}"
                group_size = len(cand_ids)

                # Per-state normalized targets
                y_vals = [float(y_map[cid]) for cid in cand_ids]
                if not y_vals:
                    continue

                mean_y = sum(y_vals) / len(y_vals)
                var_y = sum((v - mean_y) ** 2 for v in y_vals) / len(y_vals)
                std_y = math.sqrt(var_y)
                if std_y > 0.0:
                    y_z_vals = [(v - mean_y) / std_y for v in y_vals]
                else:
                    y_z_vals = [0.0 for _ in y_vals]

                softmax_temp = 50.0
                max_y = max(y_vals)
                exp_vals = [math.exp((v - max_y) / softmax_temp) for v in y_vals]
                denom = sum(exp_vals) or 1.0
                y_soft_vals = [ev / denom for ev in exp_vals]

                for idx, cid in enumerate(cand_ids):
                    feats, vec = featurize(s_state, user_team_idx, cid, universe, prefs)
                    f.write(json.dumps({
                        "state_id": state_id,
                        "group_size": group_size,
                        "candidate_id": cid,
                        "y": y_map[cid],
                        "y_z": y_z_vals[idx],
                        "y_softmax": y_soft_vals[idx],
                        "x": vec,
                        "features": feats,
                    }) + "\n")
                    row_count += 1

            if sleep_s > 0:
                time.sleep(sleep_s)

            print(f"Sim draft {d+1}/{num_drafts} done. Total rows so far: {row_count}")

    save_json(FEATURE_INDEX, OUT_DIR / "feature_index.json")
    save_json({
        "seed": seed,
        "num_drafts": num_drafts,
        "candidates_k": candidates_k,
        "rollouts_k": rollouts_k,
        "league_settings": settings.__dict__,
        "prefs": prefs.__dict__,
        "user_team_idx": user_team_idx,
        "universe_size": len(universe),
        "user_model_prob": user_model_prob if user_model_policy is not None else None,
        "notes": "Updated: bench_spots_remaining fixed; added total_spots_remaining; worst_week_index_norm neutral when std==0; added per-state y_z and y_softmax targets; optional user_model_policy for data gen",
    }, OUT_DIR / "meta.json")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--top-n", type=int, default=700)
    ap.add_argument("--num-drafts", type=int, default=40)
    ap.add_argument("--candidates-k", type=int, default=60)
    ap.add_argument("--rollouts-k", type=int, default=1)
    ap.add_argument("--user-team-idx", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sleep", type=float, default=0.0)

    ap.add_argument("--pref-bye", type=float, default=0.6)
    ap.add_argument("--pref-risk", type=float, default=0.5)
    ap.add_argument("--pref-reach", type=float, default=0.5)
    ap.add_argument("--pref-stack", type=float, default=0.0)
    ap.add_argument("--w-qb", type=float, default=1.0)
    ap.add_argument("--w-rb", type=float, default=1.0)
    ap.add_argument("--w-wr", type=float, default=1.0)
    ap.add_argument("--w-te", type=float, default=1.0)
    ap.add_argument("--w-k", type=float, default=0.6)
    ap.add_argument("--w-def", type=float, default=0.6)

    ap.add_argument("--model", type=str, default="", help="Path to model_mlp.pt; if set, user uses this model with probability --user-model-prob")
    ap.add_argument("--user-model-prob", type=float, default=0.5, help="Probability each draft uses model as user policy (default 0.5); only used if --model is set")

    args = ap.parse_args()

    settings = LeagueSettings()
    prefs = Prefs(
        bye_safety_weight=args.pref_bye,
        risk_tolerance=args.pref_risk,
        reach_penalty_weight=args.pref_reach,
        stack_weight=args.pref_stack,
        w_qb=args.w_qb, w_rb=args.w_rb, w_wr=args.w_wr, w_te=args.w_te, w_k=args.w_k, w_def=args.w_def,
    )

    universe = build_player_universe(args.season)
    universe = trim_universe_top_n_by_adp(universe, top_n=args.top_n)

    needed_picks = settings.num_teams * settings.rounds
    if len(universe) < needed_picks + 50:
        raise RuntimeError(f"Universe too small ({len(universe)}) for {needed_picks} picks. Increase --top-n or fix filters.")

    user_model_policy = None
    if args.model:
        import sys
        _root = Path(__file__).resolve().parent.parent
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        model_path = Path(args.model)
        if not model_path.is_absolute():
            model_path = _root / model_path
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        import torch
        from model import MLPScorer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(model_path, map_location=device, weights_only=True)
        _model = MLPScorer(input_dim=ckpt["input_dim"], hidden_dim=ckpt["hidden_dim"])
        _model.load_state_dict(ckpt["state_dict"])
        _model.to(device)
        _model.eval()
        _feature_index = ckpt["feature_index"]
        _candidates_k_model = 200

        def _model_user_policy(st: DraftState, tid: int) -> str:
            if tid != args.user_team_idx:
                return choose_pick_adp_need_noise(st, tid, universe)
            cand_ids = get_candidates(st, k=_candidates_k_model)
            if not cand_ids:
                return next(iter(st.available.keys()))
            vecs = []
            for cid in cand_ids:
                _, vec = featurize(st, args.user_team_idx, cid, universe, prefs)
                vecs.append(vec)
            X = torch.tensor(vecs, dtype=torch.float32, device=device)
            with torch.no_grad():
                scores = _model(X)
            return cand_ids[int(scores.argmax().item())]

        user_model_policy = _model_user_policy
        print(f"Using model from {model_path} for user with probability {args.user_model_prob}")

    out_jsonl = OUT_DIR / "train_rows.jsonl"
    generate_dataset(
        universe=universe,
        settings=settings,
        prefs=prefs,
        user_team_idx=args.user_team_idx,
        num_drafts=args.num_drafts,
        candidates_k=args.candidates_k,
        rollouts_k=args.rollouts_k,
        sleep_s=args.sleep,
        out_jsonl=out_jsonl,
        seed=args.seed,
        user_model_policy=user_model_policy,
        user_model_prob=args.user_model_prob if user_model_policy else 0.0,
    )

    print("\nDone.")
    print(f"Wrote: {out_jsonl}")
    print(f"Wrote: {OUT_DIR/'feature_index.json'}")
    print(f"Wrote: {OUT_DIR/'meta.json'}")


if __name__ == "__main__":
    main()