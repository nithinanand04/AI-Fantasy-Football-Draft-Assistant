"""Load local exports + model; score candidates (same as draft_sim)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from data_scripts.label_data import LeagueSettings, Player, Prefs, compute_risk_prob, featurize_with_counts, load_json, roster_counts
from model import MLPScorer

from draft_server.paths import EXPORTS, MODEL_PATH

_model: MLPScorer | None = None
_device: torch.device | None = None


def load_players_map() -> Dict[str, Any]:
    return load_json(EXPORTS / "players_nfl.json")


def _load_stats_weekly_with_fallback(season: int) -> Dict[str, Any]:
    """
    Load weekly stats for risk computation.

    For preseason / early-season years (e.g. 2026), stats may be missing or empty.
    In that case, fall back to previous season stats for risk only.
    """
    current_path = EXPORTS / f"stats_weekly_{season}_players_500.json"
    if current_path.exists():
        data = load_json(current_path)
        if isinstance(data, dict) and data:
            return data

    fallback_path = EXPORTS / f"stats_weekly_{season - 1}_players_500.json"
    if fallback_path.exists():
        data = load_json(fallback_path)
        if isinstance(data, dict) and data:
            return data

    return {}


def load_universe(season: int, top_n: int, settings: LeagueSettings) -> Dict[str, Player]:
    players = load_json(EXPORTS / "players_nfl.json")
    proj_rows = load_json(EXPORTS / f"projections_{season}_week1.json")
    stats_weekly = _load_stats_weekly_with_fallback(season)
    selected_rows = load_json(EXPORTS / f"selected_top_500_{season}_week1_ppr.json")
    selected_ids = [str(r["player_id"]) for r in selected_rows[:top_n]]
    proj_by_pid = {str(r.get("player_id")): r for r in proj_rows if r.get("player_id") is not None}

    universe: Dict[str, Player] = {}
    for pid in selected_ids:
        meta = players.get(pid, {}) or {}
        pos = meta.get("position")
        if pos not in ("QB", "RB", "WR", "TE", "K", "DEF"):
            continue
        pr = proj_by_pid.get(pid, {})
        st = pr.get("stats") or {}
        adp = st.get("adp_dd_ppr") or st.get("adp_dd_std")
        if not isinstance(adp, (int, float)):
            adp = 9999.0
        proj_w1 = st.get("pts_ppr")
        if not isinstance(proj_w1, (int, float)):
            proj_w1 = 0.0
        bye_week = int((meta.get("metadata") or {}).get("bye_week") or 0)
        risk_prob = compute_risk_prob(pid, meta, stats_weekly, settings.season_weeks)
        universe[pid] = Player(
            player_id=pid,
            pos=pos,
            team=str(meta.get("team") or ""),
            bye_week=bye_week,
            adp=float(adp),
            proj_week1_pts=float(proj_w1),
            proj_season_pts=float(proj_w1) * settings.season_weeks,
            risk_prob=float(risk_prob),
        )
    return universe


def ensure_model_loaded(model_path: Path | None = None) -> Tuple[MLPScorer, torch.device]:
    global _model, _device
    if _model is not None and _device is not None:
        return _model, _device
    path = model_path or MODEL_PATH
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    m = MLPScorer(int(ckpt["input_dim"]), hidden_dim=int(ckpt.get("hidden_dim", 128))).to(device)
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    _model, _device = m, device
    return m, device


def recommend_top(
    state,
    user_team_idx: int,
    universe: Dict[str, Player],
    prefs: Prefs,
    candidates_k: int,
    limit: int,
) -> List[Dict[str, Any]]:
    model, device = ensure_model_loaded()
    avail_sorted = sorted(state.available.values(), key=lambda p: p.adp)
    cand_objs = avail_sorted[: min(candidates_k, len(avail_sorted))]
    if not cand_objs:
        return []
    my_counts = roster_counts(state.rosters[user_team_idx], universe)
    vecs = [featurize_with_counts(state, user_team_idx, c, prefs, my_counts) for c in cand_objs]
    X = torch.tensor(vecs, dtype=torch.float32, device=device)
    with torch.no_grad():
        scores = model(X).tolist()
    rows = []
    for c, s in zip(cand_objs, scores):
        rows.append(
            {
                "player_id": c.player_id,
                "score": float(s),
                "pos": c.pos,
                "team": c.team,
                "adp": c.adp,
                "proj_season_pts": c.proj_season_pts,
                "risk_prob": c.risk_prob,
            }
        )
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows[:limit]
