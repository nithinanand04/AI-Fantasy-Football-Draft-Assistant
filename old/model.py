#!/usr/bin/env python3
"""
Simple listwise neural ranker MVP for the draft assistant.

Trains on:
- sleeper_exports/train_rows.jsonl
- sleeper_exports/feature_index.json

Each JSONL row is one (state, candidate) with:
- state_id: str
- x: List[float]        (features in feature_index order)
- y_softmax: float      (per-state softmax target, sums to 1 over candidates)

We train a small MLP that outputs a scalar score per candidate. For each state
we apply a softmax over scores and minimize cross-entropy vs. y_softmax.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = ROOT / "sleeper_exports" / "train_rows.jsonl"
DEFAULT_FEATURE_INDEX_PATH = ROOT / "sleeper_exports" / "feature_index.json"
DEFAULT_MODEL_OUT = ROOT / "model_mlp.pt"


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class StateBatch:
    """
    One draft decision state (one pick for the user team).

    X: [group_size, input_dim]
    target: [group_size] softmax targets (sum to 1 for this state).
    """

    state_id: str
    X: torch.Tensor
    target: torch.Tensor


class DraftDataset:
    """
    In-memory dataset grouped by state_id.

    Each __getitem__ returns a StateBatch.
    """

    def __init__(
        self,
        data_path: Path,
        feature_index_path: Path,
        device: torch.device,
    ) -> None:
        self.data_path = Path(data_path)
        self.feature_index_path = Path(feature_index_path)
        self.device = device

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        if not self.feature_index_path.exists():
            raise FileNotFoundError(f"Feature index not found: {self.feature_index_path}")

        self.feature_names: List[str] = json.loads(self.feature_index_path.read_text())
        self.input_dim: int = len(self.feature_names)

        # Load and group rows by state_id
        grouped: Dict[str, List[Tuple[List[float], float]]] = {}
        with self.data_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)

                state_id = row["state_id"]
                x = row["x"]
                y_soft = float(row.get("y_softmax", 0.0))

                # Basic sanity: enforce correct length
                if len(x) != self.input_dim:
                    raise RuntimeError(
                        f"Feature length mismatch for state {state_id}, "
                        f"expected {self.input_dim}, got {len(x)}"
                    )

                grouped.setdefault(state_id, []).append((x, y_soft))

        # Convert to StateBatch objects and normalize any numeric drift in y_softmax
        self._states: List[StateBatch] = []
        for state_id, rows in grouped.items():
            xs, ys = zip(*rows)

            X = torch.tensor(xs, dtype=torch.float32, device=device)  # [G, D]
            y = torch.tensor(ys, dtype=torch.float32, device=device)  # [G]

            total = float(y.sum().item())
            if total <= 0.0 or not math.isfinite(total):
                # Fall back to uniform if something went wrong
                y = torch.full_like(y, 1.0 / len(y))
            else:
                y = y / total

            self._states.append(StateBatch(state_id=state_id, X=X, target=y))

        # Stable order but we can shuffle indices per epoch
        self._indices: List[int] = list(range(len(self._states)))

    def __len__(self) -> int:
        return len(self._states)

    def __getitem__(self, idx: int) -> StateBatch:
        return self._states[idx]

    def shuffled_indices(self) -> List[int]:
        idxs = self._indices[:]
        random.shuffle(idxs)
        return idxs


# -----------------------------
# Model
# -----------------------------

class MLPScorer(nn.Module):
    """
    Simple MLP that maps feature vectors to a scalar score.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: [group_size, input_dim]
        returns scores: [group_size]
        """
        scores = self.net(X)  # [G, 1]
        return scores.squeeze(-1)


# -----------------------------
# Training utilities
# -----------------------------

def listwise_loss(scores: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    """
    Listwise softmax cross-entropy for a single state.

    scores: [G]
    target_probs: [G], sum to 1
    """
    log_probs = torch.log_softmax(scores, dim=0)  # [G]
    loss = -(target_probs * log_probs).sum()
    return loss


def train(
    dataset: DraftDataset,
    model: MLPScorer,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    val_fraction: float = 0.5,
) -> None:
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    n_states = len(dataset)
    n_val = max(1, int(val_fraction * n_states))
    indices = list(range(n_states))
    random.shuffle(indices)
    val_idx = set(indices[:n_val])

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: List[float] = []

        for idx in dataset.shuffled_indices():
            if idx in val_idx:
                continue

            batch = dataset[idx]
            scores = model(batch.X)  # [G]
            loss = listwise_loss(scores, batch.target)

            optim.zero_grad()
            loss.backward()
            optim.step()

            train_losses.append(float(loss.item()))

        # Validation / test split (same 50% set every epoch)
        model.eval()
        val_losses: List[float] = []
        correct_top1 = 0
        correct_top5 = 0
        n_val_states = 0
        with torch.no_grad():
            for idx in val_idx:
                batch = dataset[idx]
                scores = model(batch.X)
                loss = listwise_loss(scores, batch.target)
                val_losses.append(float(loss.item()))

                # Accuracy-style metrics: top-1 and top-5 hit rate
                true_idx = int(batch.target.argmax().item())
                pred_order = scores.argsort(descending=True)
                if int(pred_order[0].item()) == true_idx:
                    correct_top1 += 1
                if (pred_order[:5] == true_idx).any().item():
                    correct_top5 += 1
                n_val_states += 1

        mean_train = sum(train_losses) / max(1, len(train_losses))
        mean_val = sum(val_losses) / max(1, len(val_losses))
        top1 = correct_top1 / max(1, n_val_states)
        top5 = correct_top5 / max(1, n_val_states)
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={mean_train:.4f} | val_loss={mean_val:.4f} | "
            f"val_top1={top1:.3f} | val_top5={top5:.3f}"
        )


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train a simple listwise MLP ranker on Sleeper draft data.")
    ap.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH), help="Path to train_rows.jsonl")
    ap.add_argument("--features", type=str, default=str(DEFAULT_FEATURE_INDEX_PATH), help="Path to feature_index.json")
    ap.add_argument("--model-out", type=str, default=str(DEFAULT_MODEL_OUT), help="Where to save the trained model .pt")

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--val-fraction", type=float, default=0.5, help="Fraction of states to use as validation/test")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    data_path = Path(args.data)
    feat_path = Path(args.features)
    model_out = Path(args.model_out)

    print(f"Loading dataset from {data_path}")
    dataset = DraftDataset(data_path=data_path, feature_index_path=feat_path, device=device)
    print(f"Loaded {len(dataset)} states")
    print(f"Input dim: {dataset.input_dim}")

    model = MLPScorer(input_dim=dataset.input_dim, hidden_dim=args.hidden_dim)

    train(
        dataset=dataset,
        model=model,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        val_fraction=args.val_fraction,
    )

    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": dataset.input_dim,
            "hidden_dim": args.hidden_dim,
            "feature_index": dataset.feature_names,
        },
        model_out,
    )
    print(f"Saved model to {model_out}")


if __name__ == "__main__":
    main()

