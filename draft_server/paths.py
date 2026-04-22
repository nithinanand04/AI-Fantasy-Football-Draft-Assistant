"""Shared filesystem paths used by the local draft server modules."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPORTS = REPO_ROOT / "sleeper_exports"
MODEL_PATH = REPO_ROOT / "model_mlp.pt"
