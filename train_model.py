#!/usr/bin/env python3
"""
Train/evaluate the listwise ranker and save a model checkpoint.

Inputs:
- sleeper_exports/train_rows.jsonl
- sleeper_exports/test_rows.jsonl
- sleeper_exports/feature_index.json

Output:
- model_mlp.pt
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

from model import MLPScorer

ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN_PATH = ROOT / "sleeper_exports" / "train_rows.jsonl"
DEFAULT_TEST_PATH = ROOT / "sleeper_exports" / "test_rows.jsonl"
DEFAULT_FEATURE_INDEX_PATH = ROOT / "sleeper_exports" / "feature_index.json"
DEFAULT_MODEL_OUT = ROOT / "model_mlp.pt"


@dataclass
class StateBatch:
    state_id: str
    X: torch.Tensor
    target: torch.Tensor


class StateDataset:
    # Load JSONL rows and group candidates by state_id for listwise training.
    def __init__(self, path: Path, feature_index_path: Path, device: torch.device) -> None:
        self.path = Path(path)
        self.feature_index_path = Path(feature_index_path)
        self.device = device

        if not self.path.exists():
            raise FileNotFoundError(f"Data file not found: {self.path}")
        if not self.feature_index_path.exists():
            raise FileNotFoundError(f"Feature index not found: {self.feature_index_path}")

        self.feature_names: List[str] = json.loads(self.feature_index_path.read_text())
        self.input_dim = len(self.feature_names)

        grouped: Dict[str, List[Tuple[List[float], float]]] = {}
        with self.path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                sid = row["state_id"]
                x = row["x"]
                y_soft = float(row.get("y_softmax", 0.0))
                if len(x) != self.input_dim:
                    raise RuntimeError(
                        f"Feature length mismatch for {sid}: got {len(x)}, expected {self.input_dim}"
                    )
                grouped.setdefault(sid, []).append((x, y_soft))

        self.states: List[StateBatch] = []
        for sid, rows in grouped.items():
            xs, ys = zip(*rows)
            X = torch.tensor(xs, dtype=torch.float32, device=device)
            y = torch.tensor(ys, dtype=torch.float32, device=device)
            denom = float(y.sum().item())
            if denom <= 0.0 or not math.isfinite(denom):
                y = torch.full_like(y, 1.0 / len(y))
            else:
                y = y / denom
            self.states.append(StateBatch(state_id=sid, X=X, target=y))

    # Return number of grouped listwise states.
    def __len__(self) -> int:
        return len(self.states)

    # Return shuffled state indices each epoch.
    def shuffled_indices(self) -> List[int]:
        idx = list(range(len(self.states)))
        random.shuffle(idx)
        return idx


# Compute listwise cross-entropy between model scores and target probability vector.
def listwise_loss(scores: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(scores, dim=0)
    return -(target_probs * log_probs).sum()


# Evaluate loss, top1, and top5 on a grouped dataset.
def evaluate(model: MLPScorer, dataset: StateDataset) -> Tuple[float, float, float]:
    model.eval()
    losses: List[float] = []
    top1 = 0
    top5 = 0
    n = 0
    with torch.no_grad():
        for batch in dataset.states:
            scores = model(batch.X)
            loss = listwise_loss(scores, batch.target)
            losses.append(float(loss.item()))

            true_idx = int(batch.target.argmax().item())
            pred_order = scores.argsort(descending=True)
            if int(pred_order[0].item()) == true_idx:
                top1 += 1
            if (pred_order[:5] == true_idx).any().item():
                top5 += 1
            n += 1
    mean_loss = sum(losses) / max(1, len(losses))
    return mean_loss, (top1 / max(1, n)), (top5 / max(1, n))


# Parse CLI arguments for train/eval inputs and optimization settings.
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train listwise MLP ranker on train/test JSONL files.")
    ap.add_argument("--train", type=str, default=str(DEFAULT_TRAIN_PATH))
    ap.add_argument("--test", type=str, default=str(DEFAULT_TEST_PATH))
    ap.add_argument("--features", type=str, default=str(DEFAULT_FEATURE_INDEX_PATH))
    ap.add_argument("--model-out", type=str, default=str(DEFAULT_MODEL_OUT))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


# Run training loop, report metrics each epoch, and save checkpoint.
def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    train_ds = StateDataset(Path(args.train), Path(args.features), device)
    test_ds = StateDataset(Path(args.test), Path(args.features), device)

    print(f"Train states: {len(train_ds)}")
    print(f"Test states:  {len(test_ds)}")
    print(f"Input dim:    {train_ds.input_dim}")

    model = MLPScorer(train_ds.input_dim, hidden_dim=args.hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_losses: List[float] = []
        for idx in train_ds.shuffled_indices():
            batch = train_ds.states[idx]
            scores = model(batch.X)
            loss = listwise_loss(scores, batch.target)

            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_losses.append(float(loss.item()))

        train_loss = sum(tr_losses) / max(1, len(tr_losses))
        test_loss, test_top1, test_top5 = evaluate(model, test_ds)
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"test_loss={test_loss:.4f} | "
            f"test_top1={test_top1:.3f} | "
            f"test_top5={test_top5:.3f}"
        )

    out = Path(args.model_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": train_ds.input_dim,
            "hidden_dim": args.hidden_dim,
            "feature_index": train_ds.feature_names,
        },
        out,
    )
    print(f"\nSaved model to {out}")


if __name__ == "__main__":
    main()

