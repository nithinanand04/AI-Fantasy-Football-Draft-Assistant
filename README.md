# AI Fantasy Football Draft Assistant

Local draft assistant that:
- pulls **live draft state** from Sleeper,
- builds the same feature vector used in training,
- runs an MLP ranker,
- returns top player recommendations during your draft turn.

---

## 1) Code Structure

### Core model/training
- `model.py`
  - Inference model class (`MLPScorer`) used by the server.
- `train_model.py`
  - Full training/eval script (listwise grouping/loss, metrics, checkpoint save).

### Data pipeline
- `data_scripts/get_data.py`
  - Downloads Sleeper players/projections.
  - For historical years: optional weekly stats + injury filtering.
  - For 2026+: projections-only/top-N mode by default.
- `data_scripts/label_data.py`
  - Shared feature/data helper module used by server inference (settings, `Prefs`, `Player`, `DraftState`, featurization).

### Live server
- `draft_server/app.py`
  - FastAPI app and REST endpoints.
- `draft_server/sleeper_client.py`
  - Sleeper HTTP API wrappers.
- `draft_server/live_draft.py`
  - Converts draft+picks+rosters into current `DraftState`.
- `draft_server/ranking.py`
  - Loads local universe + model checkpoint and scores candidates.
- `draft_server/paths.py`
  - Shared repo/export/model paths.

### UI
- `web/index.html`
- `web/styles.css`
- `web/app.js`

### Generated assets (not committed)
- `sleeper_exports/*` (downloaded/generated data)
- `model_mlp.pt` (trained checkpoint)

---

## 2) Dependencies

Python 3.10+ recommended.

Install:

```bash
python3 -m pip install -r requirements-ui.txt
python3 -m pip install requests
```

`requirements-ui.txt` includes:
- `fastapi`
- `uvicorn[standard]`
- `httpx`
- `pydantic`
- `torch`

---

## 3) How To Run

### A) Download data exports

#### Historical/full mode (example: 2025)
```bash
python3 data_scripts/get_data.py --season 2025 --top-n 500 --season-weeks 14 --max-injury-weeks 1 --scoring ppr
```

#### 2026 mode (projections + selected top-N)
```bash
python3 data_scripts/get_data.py --season 2026 --top-n 500 --scoring ppr
```

### B) Train model (if you do not already have a checkpoint)
```bash
python3 train_model.py --epochs 25 --hidden-dim 128 --lr 1e-3 --weight-decay 1e-5
```

This writes `model_mlp.pt` at repo root.

### C) Run API + UI
```bash
uvicorn draft_server.app:app --reload --host 127.0.0.1 --port 8787
```

Open:
- `http://127.0.0.1:8787/`

---

## 4) API Endpoints (Local)

- `GET /api/health`
- `GET /api/user?username=<sleeper_username>`
- `GET /api/drafts?user_id=<id>&sport=nfl&season=2026`
- `GET /api/draft-status?draft_id=<id>&user_id=<id>&season=<year>`
- `POST /api/recommend`

`/api/recommend` expects prefs and returns ranked recommendations.

---

## 5) Dataset / Model Download Policy

### Dataset download
- This project uses Sleeper public APIs.
- `data_scripts/get_data.py` **automatically downloads** players/projections (and weekly stats when applicable).
- No manual dataset zip download is required.

### Model download
- There is no external hosted model download in this repo.
- If `model_mlp.pt` is missing, run `train_model.py` to generate it locally.

### Exception note
- Future-season weekly stats (e.g., 2026 preseason) may not exist yet in Sleeper.
- In server ranking, risk stats fall back to prior season when needed.

---

## 6) Authorship / Provenance

### External repository usage
- **No code copied from external repositories.**

### Authorship statement
- All active project code in this submission was written by me.
- The `old/` folder contains archived prior iterations by me and is not the runtime source for the current app.

### Adapted-from-prior (my own prior code) with edited line ranges
The following active files were refactored/rebuilt from my own earlier project iterations; line ranges below indicate the sections edited in this submission:

- `data_scripts/get_data.py`: lines `1-338`
- `data_scripts/label_data.py`: lines `1-199`
- `model.py`: lines `1-28`
- `train_model.py`: lines `1-199`
- `draft_server/app.py`: lines `1-186`
- `draft_server/sleeper_client.py`: lines `1-54`
- `draft_server/live_draft.py`: lines `1-101`
- `draft_server/ranking.py`: lines `1-132`
- `draft_server/paths.py`: lines `1-8`
- `web/index.html`: lines `1-69`
- `web/styles.css`: lines `1-129`
- `web/app.js`: lines `1-263`

---

## 7) Notes

- Current live draft logic assumes the standard league settings used in training (`12 teams`, `16 rounds`).
- UI positional preferences are rank-based (`1..6`) and converted internally to model weights.

