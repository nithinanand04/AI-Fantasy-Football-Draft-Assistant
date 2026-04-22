"""Sleeper HTTP reads (no auth required for public endpoints)."""

from __future__ import annotations

from typing import Any, Dict, List

import httpx

BASE = "https://api.sleeper.app/v1"


# Generic Sleeper GET helper returning parsed JSON.
def _get(path: str, timeout: float = 30.0) -> Any:
    with httpx.Client(timeout=timeout) as c:
        r = c.get(f"{BASE}{path}")
        r.raise_for_status()
        return r.json()


# Resolve Sleeper user profile by username.
def get_user_by_username(username: str) -> Dict[str, Any]:
    data = _get(f"/user/{username}")
    return data if isinstance(data, dict) else {}


# List drafts for a user/sport/season combination.
def get_user_drafts(user_id: str, sport: str, season: int) -> List[Dict[str, Any]]:
    data = _get(f"/user/{user_id}/drafts/{sport}/{season}", timeout=60.0)
    return data if isinstance(data, list) else []


# Fetch draft metadata by draft_id.
def get_draft(draft_id: str) -> Dict[str, Any]:
    data = _get(f"/draft/{draft_id}")
    return data if isinstance(data, dict) else {}


# Fetch all picks currently recorded for a draft.
def get_draft_picks(draft_id: str) -> List[Dict[str, Any]]:
    data = _get(f"/draft/{draft_id}/picks", timeout=60.0)
    return data if isinstance(data, list) else []


# Fetch roster ownership/mapping for a league.
def get_league_rosters(league_id: str) -> List[Dict[str, Any]]:
    data = _get(f"/league/{league_id}/rosters", timeout=60.0)
    return data if isinstance(data, list) else []


# Fetch league metadata (e.g., name) for display.
def get_league(league_id: str) -> Dict[str, Any]:
    data = _get(f"/league/{league_id}")
    return data if isinstance(data, dict) else {}
