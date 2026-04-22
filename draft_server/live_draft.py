"""Build DraftState from Sleeper draft + picks + league rosters (MVP: 12 teams)."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple

from data_scripts.label_data import DraftState, LeagueSettings, Player


# Extract ordered round-1 roster ids from Sleeper draft_order format.
def _round1_roster_ids(draft: Dict[str, Any]) -> List[str]:
    raw = draft.get("draft_order")
    if isinstance(raw, list):
        return [str(x) for x in raw if x is not None]
    if isinstance(raw, dict):
        pairs = []
        for k, v in raw.items():
            try:
                slot = int(k)
            except (TypeError, ValueError):
                continue
            pairs.append((slot, str(v)))
        pairs.sort(key=lambda x: x[0])
        return [rid for _, rid in pairs]
    return []


# Build full snake draft pick order from round-1 team index order.
def _snake_order(round1: List[int], rounds: int) -> List[int]:
    order: List[int] = []
    for r in range(rounds):
        order.extend(round1 if r % 2 == 0 else list(reversed(round1)))
    return order


def _rid_to_team_idx_from_draft(draft: Dict[str, Any], settings: LeagueSettings) -> Dict[str, int]:
    # Team index = round-1 draft slot (must match Sleeper draft_order, not sorted roster ids).
    r1 = _round1_roster_ids(draft)
    if len(r1) != settings.num_teams:
        raise ValueError(f"draft_order length {len(r1)} != num_teams {settings.num_teams}.")
    return {str(rid): i for i, rid in enumerate(r1)}


# Resolve the requesting user's team index from league roster ownership.
def resolve_user_team_idx(rosters: List[Dict[str, Any]], user_id: str, rid_to_idx: Dict[str, int]) -> int:
    uid = str(user_id)
    for r in rosters:
        owners = [str(r.get("owner_id") or "")]
        for co in r.get("co_owners") or []:
            owners.append(str(co))
        if uid in owners:
            rid = str(r.get("roster_id") or "")
            if rid in rid_to_idx:
                return rid_to_idx[rid]
    raise ValueError("user_id not found on any roster in this league.")


# Construct current DraftState from live Sleeper draft/pick payloads.
def build_draft_state(
    draft: Dict[str, Any],
    picks: List[Dict[str, Any]],
    rosters: List[Dict[str, Any]],
    user_id: str,
    universe: Dict[str, Player],
    players_map: Dict[str, Any],
    settings: LeagueSettings | None = None,
) -> Tuple[DraftState, int, Dict[str, Player]]:
    settings = settings or LeagueSettings()
    rid_to_idx = _rid_to_team_idx_from_draft(draft, settings)
    my_idx = resolve_user_team_idx(rosters, user_id, rid_to_idx)

    r1_ids = _round1_roster_ids(draft)
    round1_idx = [rid_to_idx[str(rid)] for rid in r1_ids]
    order = _snake_order(round1_idx, settings.rounds)

    sorted_picks = sorted(picks, key=lambda p: int(p.get("pick_no") or 0))
    rosters_by: Dict[int, List[str]] = {i: [] for i in range(settings.num_teams)}
    history_pos: List[str] = []

    avail = copy.deepcopy(universe)
    for p in sorted_picks:
        pid = str(p.get("player_id") or "")
        if not pid:
            continue
        rid = str(p.get("roster_id") or "")
        if rid not in rid_to_idx:
            continue
        tid = rid_to_idx[rid]
        rosters_by[tid].append(pid)
        meta = players_map.get(pid, {}) or {}
        history_pos.append(str(meta.get("position") or "?"))
        avail.pop(pid, None)

    pick_no = len([p for p in sorted_picks if p.get("player_id")]) + 1
    state = DraftState(
        settings=settings,
        rosters=rosters_by,
        available=avail,
        pick_no=pick_no,
        pick_history_pos=history_pos,
        order=order,
    )
    return state, my_idx, universe
