"""
Local server: Sleeper API + static UI + model recommendations.

Run from repo root:
  uvicorn draft_server.app:app --reload --host 127.0.0.1 --port 8787
Then open http://127.0.0.1:8787/
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_scripts.label_data import LeagueSettings, Prefs

from draft_server import sleeper_client
from draft_server.live_draft import build_draft_state
from draft_server.paths import REPO_ROOT
from draft_server.ranking import ensure_model_loaded, load_players_map, load_universe, recommend_top

_players_map: Dict[str, Any] = {}
_universe_cache: Dict[int, Dict[str, Any]] = {}


def _universe(season: int):
    if season not in _universe_cache:
        _universe_cache[season] = load_universe(season, 500, LeagueSettings())
    return _universe_cache[season]


class PrefsBody(BaseModel):
    risk_factor: float = 0.5
    w_qb: float = 4.0
    w_rb: float = 6.0
    w_wr: float = 5.0
    w_te: float = 3.0
    w_k: float = 2.0
    w_def: float = 1.0


class RecommendBody(BaseModel):
    draft_id: str
    user_id: str
    season: int = 2024
    candidates_k: int = 60
    limit: int = 5
    prefs: PrefsBody = Field(default_factory=PrefsBody)


app = FastAPI(title="Draft Assistant (local)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    global _players_map
    ensure_model_loaded()
    _players_map = load_players_map()


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/user")
def api_user(username: str = Query(..., description="Sleeper username")):
    u = sleeper_client.get_user_by_username(username)
    if not u.get("user_id"):
        raise HTTPException(404, "User not found")
    return {"user_id": u.get("user_id"), "username": u.get("username"), "display_name": u.get("display_name")}


@app.get("/api/drafts")
def api_drafts(
    user_id: str = Query(...),
    sport: str = Query("nfl"),
    season: int = Query(2026),
):
    drafts = sleeper_client.get_user_drafts(user_id, sport, season)
    out = []
    for d in drafts:
        if not isinstance(d, dict):
            continue
        lid = str(d.get("league_id") or "")
        league = sleeper_client.get_league(lid) if lid else {}
        out.append(
            {
                "draft_id": d.get("draft_id"),
                "league_id": lid,
                "season": d.get("season"),
                "status": d.get("status"),
                "draft_type": d.get("type"),
                "league_name": league.get("name") or "",
            }
        )
    return {"drafts": out, "count": len(out)}


@app.get("/api/draft-status")
def api_draft_status(
    draft_id: str = Query(...),
    user_id: str = Query(...),
    season: int = Query(2024, description="Season for local stats/projections files"),
):
    settings = LeagueSettings()
    draft = sleeper_client.get_draft(draft_id)
    picks = sleeper_client.get_draft_picks(draft_id)
    league_id = str(draft.get("league_id") or "")
    if not league_id:
        raise HTTPException(400, "Draft has no league_id")
    rosters = sleeper_client.get_league_rosters(league_id)
    universe = _universe(season)
    try:
        state, my_idx, _u = build_draft_state(draft, picks, rosters, user_id, universe, _players_map, settings)
    except ValueError as e:
        raise HTTPException(400, str(e))

    order = state.order
    pn = state.pick_no
    idx = pn - 1
    in_progress = 0 <= idx < len(order)
    current_team = order[idx] if in_progress else None
    is_yours = in_progress and (current_team == my_idx)
    return {
        "draft_id": draft_id,
        "draft_status": draft.get("status"),
        "pick_no": pn,
        "picks_recorded": len([p for p in picks if p.get("player_id")]),
        "total_picks": len(order),
        "user_team_idx": my_idx,
        "current_team_idx": current_team,
        "is_your_pick": is_yours,
        "in_progress": in_progress,
    }


@app.post("/api/recommend")
def api_recommend(body: RecommendBody):
    settings = LeagueSettings()
    draft = sleeper_client.get_draft(body.draft_id)
    picks = sleeper_client.get_draft_picks(body.draft_id)
    league_id = str(draft.get("league_id") or "")
    if not league_id:
        raise HTTPException(400, "Draft has no league_id")
    rosters = sleeper_client.get_league_rosters(league_id)
    universe = _universe(body.season)
    try:
        state, my_idx, _u = build_draft_state(draft, picks, rosters, body.user_id, universe, _players_map, settings)
    except ValueError as e:
        raise HTTPException(400, str(e))

    prefs = Prefs(
        risk_factor=body.prefs.risk_factor,
        w_qb=body.prefs.w_qb,
        w_rb=body.prefs.w_rb,
        w_wr=body.prefs.w_wr,
        w_te=body.prefs.w_te,
        w_k=body.prefs.w_k,
        w_def=body.prefs.w_def,
    )
    recs = recommend_top(state, my_idx, universe, prefs, body.candidates_k, body.limit)
    for r in recs:
        pid = str(r.get("player_id") or "")
        meta = _players_map.get(pid, {}) or {}
        r["full_name"] = str(meta.get("full_name") or "")
    return {"recommendations": recs, "pick_no": state.pick_no, "remaining": len(state.available)}


app.mount("/", StaticFiles(directory=str(REPO_ROOT / "web"), html=True), name="web")
