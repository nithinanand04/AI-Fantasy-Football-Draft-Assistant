from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

POS = ("QB", "RB", "WR", "TE", "K", "DEF")


@dataclass(frozen=True)
class LeagueSettings:
    num_teams: int = 12
    rounds: int = 16
    season_weeks: int = 14

    slot_qb: int = 1
    slot_rb: int = 2
    slot_wr: int = 3
    slot_te: int = 1
    slot_flex: int = 1
    slot_k: int = 1
    slot_def: int = 1


@dataclass(frozen=True)
class Prefs:
    risk_factor: float = 0.5
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
    team: str
    bye_week: int
    adp: float
    proj_week1_pts: float
    proj_season_pts: float
    risk_prob: float


@dataclass
class DraftState:
    settings: LeagueSettings
    rosters: Dict[int, List[str]]
    available: Dict[str, Player]
    pick_no: int
    pick_history_pos: List[str]
    order: List[int]


# Read JSON file from disk and return parsed object.
def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


# Compute/resolve weekly PPR points from Sleeper-style stat dictionaries.
def weekly_points_ppr(stats: Dict[str, Any]) -> float:
    if not isinstance(stats, dict):
        return 0.0
    for k in ("pts_ppr", "pts"):
        v = stats.get(k)
        if isinstance(v, (int, float)):
            return float(v)

    def f(key: str) -> float:
        v = stats.get(key)
        return float(v) if isinstance(v, (int, float)) else 0.0

    pts = 0.0
    pts += f("pass_yd") * 0.04 + f("pass_td") * 4.0 - f("pass_int") * 2.0
    pts += f("rush_yd") * 0.1 + f("rush_td") * 6.0
    pts += f("rec_yd") * 0.1 + f("rec_td") * 6.0 + f("rec") * 1.0
    pts -= f("fum_lost") * 2.0
    return pts


# Estimate volatility risk in [0,1] from weekly points dispersion.
def compute_risk_prob(pid: str, player: Dict[str, Any], stats_weekly: Dict[str, Any], season_weeks: int) -> float:
    bye_week = int((player.get("metadata") or {}).get("bye_week") or 0)
    vals: List[float] = []
    per_player = stats_weekly.get(pid, {})
    for w in range(1, season_weeks + 1):
        if bye_week and w == bye_week:
            continue
        wd = per_player.get(str(w))
        if not isinstance(wd, dict):
            continue
        vals.append(weekly_points_ppr(wd.get("stats") or {}))
    if not vals:
        return 0.0
    mu = sum(vals) / len(vals)
    var = sum((x - mu) ** 2 for x in vals) / len(vals)
    sd = math.sqrt(var)
    denom = sd + mu
    if denom <= 0:
        return 0.0
    return max(0.0, min(1.0, sd / denom))


# Count roster composition by fantasy position for a team.
def roster_counts(roster: List[str], universe: Dict[str, Player]) -> Dict[str, int]:
    counts = {p: 0 for p in POS}
    for pid in roster:
        p = universe.get(pid)
        if p:
            counts[p.pos] += 1
    return counts


# Return number of picks until the given team drafts again.
def picks_until_next_turn(order: List[int], pick_no: int, team_idx: int) -> int:
    start = pick_no
    for j in range(start, len(order)):
        if order[j] == team_idx:
            return j - (pick_no - 1)
    return 0


# Build the 34-feature vector for one (state, candidate, user prefs) tuple.
def featurize_with_counts(
    state: DraftState,
    user_idx: int,
    cand: Player,
    prefs: Prefs,
    my_counts: Dict[str, int],
) -> List[float]:
    s = state.settings
    pick_no = state.pick_no
    round_no = (pick_no - 1) // s.num_teams + 1
    pick_in_round = (pick_no - 1) % s.num_teams + 1
    next_turn = picks_until_next_turn(state.order, pick_no, user_idx)

    qb_needed = max(0, s.slot_qb - my_counts["QB"])
    rb_needed = max(0, s.slot_rb - my_counts["RB"])
    wr_needed = max(0, s.slot_wr - my_counts["WR"])
    te_needed = max(0, s.slot_te - my_counts["TE"])
    k_needed = max(0, s.slot_k - my_counts["K"])
    def_needed = max(0, s.slot_def - my_counts["DEF"])

    fills = 0.0
    if cand.pos == "QB" and qb_needed > 0:
        fills = 1.0
    if cand.pos == "RB" and rb_needed > 0:
        fills = 1.0
    if cand.pos == "WR" and wr_needed > 0:
        fills = 1.0
    if cand.pos == "TE" and te_needed > 0:
        fills = 1.0
    if cand.pos == "K" and k_needed > 0:
        fills = 1.0
    if cand.pos == "DEF" and def_needed > 0:
        fills = 1.0

    return [
        prefs.risk_factor,
        prefs.w_qb,
        prefs.w_rb,
        prefs.w_wr,
        prefs.w_te,
        prefs.w_k,
        prefs.w_def,
        float(pick_no),
        float(round_no),
        float(pick_in_round),
        float(next_turn),
        float(len(state.available)),
        float(my_counts["QB"]),
        float(my_counts["RB"]),
        float(my_counts["WR"]),
        float(my_counts["TE"]),
        float(my_counts["K"]),
        float(my_counts["DEF"]),
        float(len(state.rosters[user_idx])),
        float(qb_needed),
        float(rb_needed),
        float(wr_needed),
        float(te_needed),
        float(k_needed),
        float(def_needed),
        1.0 if cand.pos == "QB" else 0.0,
        1.0 if cand.pos == "RB" else 0.0,
        1.0 if cand.pos == "WR" else 0.0,
        1.0 if cand.pos == "TE" else 0.0,
        1.0 if cand.pos == "K" else 0.0,
        1.0 if cand.pos == "DEF" else 0.0,
        float(cand.proj_season_pts),
        float(cand.risk_prob),
        fills,
    ]
