"""Generic training loop for frozen modality embeddings.

Consumes a dict of (modality_name → numpy array) from .npz files and
trains any classifier from classification/classifiers.py. The classifier
is the ONLY component that learns per-target — modality encoders are frozen.

Usage:
    from s5_classification.train_loop import Trainer
    trainer = Trainer(model, config)
    results = trainer.fit(train_data, val_data)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score


@dataclass
class TrainingConfig:
    """Lightweight training configuration.

    For experiment-level config (target, modalities, architecture), see
    ``classification/train_config.py``.
    """

    n_classes: int
    hidden_dim: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 50
    early_stopping_patience: int = 10
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    batch_size: int = 32
    seed: int = 42


class Trainer:
    """Generic trainer consuming frozen modality embeddings.

    Args:
        model: Any classifier from ``classification/classifiers.py``.
        config: Training hyperparameters.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
    ):
        self.model = model
        self.config = config
        self.device = torch.device("cpu")

        torch.manual_seed(config.seed)

        self.model = self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            patience=config.scheduler_patience,
            factor=config.scheduler_factor,
        )

        # Training history
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.val_f1s: list[float] = []

    def _prepare_batch(
        self,
        data: dict[str, np.ndarray],
        indices: np.ndarray,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Extract a batch of embeddings and labels.

        Args:
            data: Dict of modality_name → numpy array (N, dim).
            indices: Indices into the N dimension.

        Returns:
            (embeddings_dict, labels_tensor).
        """
        modality_keys = [k for k in data if k.endswith("_emb")]
        embeddings = {
            key.replace("_emb", ""): torch.tensor(
                data[key][indices], dtype=torch.float32
            ).to(self.device)
            for key in modality_keys
        }
        labels = torch.tensor(
            data["labels"][indices], dtype=torch.long
        ).to(self.device)
        return embeddings, labels

    def _shuffle_indices(self, n: int) -> np.ndarray:
        """Return shuffled indices for one epoch."""
        indices = np.arange(n)
        np.random.default_rng(self.config.seed).shuffle(indices)
        return indices

    def fit(
        self,
        train_data: dict[str, np.ndarray],
        val_data: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        """Run the full training loop.

        Args:
            train_data: Training split from ``load_dataset()``.
            val_data: Validation split from ``load_dataset()``.

        Returns:
            Dictionary with training history and best metrics.
        """
        n_train = len(train_data["labels"])
        n_val = len(val_data["labels"])
        batch_size = self.config.batch_size

        best_val_f1 = -1.0
        best_epoch = 0
        best_state: dict[str, torch.Tensor] | None = None
        epochs_no_improve = 0

        for epoch in range(1, self.config.max_epochs + 1):
            # ── Train ──────────────────────────────────────────────────────
            self.model.train()
            train_loss = 0.0
            n_batches = 0
            indices = self._shuffle_indices(n_train)

            for start in range(0, n_train, batch_size):
                batch_idx = indices[start: start + batch_size]
                embeddings, labels = self._prepare_batch(train_data, batch_idx)

                self.optimizer.zero_grad()
                logits = self.model(embeddings)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                n_batches += 1

            avg_train_loss = train_loss / max(n_batches, 1)
            self.train_losses.append(avg_train_loss)

            # ── Validate ───────────────────────────────────────────────────
            val_loss, val_f1 = self._evaluate(val_data)
            self.val_losses.append(val_loss)
            self.val_f1s.append(val_f1)

            self.scheduler.step(val_f1)

            improved = val_f1 > best_val_f1
            if improved:
                best_val_f1 = val_f1
                best_epoch = epoch
                best_state = {
                    k: v.cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
                epochs_no_improve = 0
                marker = "* best"
            else:
                epochs_no_improve += 1
                marker = f"(no improve {epochs_no_improve}/{self.config.early_stopping_patience})"

            print(
                f"  Epoch {epoch:3d}: train_loss={avg_train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_f1={val_f1:.4f} {marker}"
            )

            if epochs_no_improve >= self.config.early_stopping_patience:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {self.config.early_stopping_patience} epochs)"
                )
                break

        # ── Restore best weights ───────────────────────────────────────────
        if best_state is not None:
            self.model.load_state_dict(best_state)

        print(f"\nTraining complete.")
        print(f"  Best val macro-F1: {best_val_f1:.4f} at epoch {best_epoch}")
        print(f"  Epochs run: {len(self.train_losses)}")

        return {
            "best_val_f1": best_val_f1,
            "best_epoch": best_epoch,
            "epochs_run": len(self.train_losses),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_f1s": self.val_f1s,
        }

    @torch.no_grad()
    def _evaluate(
        self,
        data: dict[str, np.ndarray],
    ) -> tuple[float, float]:
        """Evaluate on a full dataset.

        Returns:
            (average_loss, macro_f1).
        """
        self.model.eval()
        n = len(data["labels"])
        batch_size = self.config.batch_size

        total_loss = 0.0
        all_preds: list[int] = []
        all_labels: list[int] = []
        n_batches = 0

        for start in range(0, n, batch_size):
            batch_idx = np.arange(start, min(start + batch_size, n))
            embeddings, labels = self._prepare_batch(data, batch_idx)

            logits = self.model(embeddings)
            loss = self.criterion(logits, labels)
            preds = logits.argmax(dim=1)

            total_loss += loss.item()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        macro_f1 = float(
            f1_score(all_labels, all_preds, average="macro", zero_division=0)
        )
        return avg_loss, macro_f1

    @torch.no_grad()
    def evaluate(
        self,
        test_data: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        """Final evaluation on the test set.

        Args:
            test_data: Test split from ``load_dataset()``.

        Returns:
            Dictionary with macro_f1, per_class_f1, confusion_matrix,
            predictions, and labels.
        """
        self.model.eval()
        n = len(test_data["labels"])
        batch_size = self.config.batch_size

        all_preds: list[int] = []
        all_labels: list[int] = []

        for start in range(0, n, batch_size):
            batch_idx = np.arange(start, min(start + batch_size, n))
            embeddings, labels = self._prepare_batch(test_data, batch_idx)

            logits = self.model(embeddings)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        macro_f1 = float(
            f1_score(all_labels, all_preds, average="macro", zero_division=0)
        )
        per_class_f1 = {
            str(cls): float(f1)
            for cls, f1 in enumerate(
                f1_score(all_labels, all_preds, average=None, zero_division=0)
            )
        }
        cm = confusion_matrix(all_labels, all_preds).tolist()

        return {
            "macro_f1": macro_f1,
            "per_class_f1": per_class_f1,
            "confusion_matrix": cm,
            "predictions": np.array(all_preds, dtype=np.int64),
            "labels": np.array(all_labels, dtype=np.int64),
        }


def plot_curves(
    train_losses: list[float],
    val_losses: list[float],
    val_f1s: list[float],
    save_path: Path,
) -> None:
    """Plot and save training curves (2-panel: loss + F1)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(train_losses) + 1)

    axes[0].plot(epochs, train_losses, label="Train loss", marker="o", markersize=3)
    axes[0].plot(epochs, val_losses, label="Val loss", marker="o", markersize=3)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].set_title("Training and Validation Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, val_f1s, label="Val macro-F1", color="green", marker="o", markersize=3)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro-F1")
    axes[1].legend()
    axes[1].set_title("Validation Macro-F1")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved to {save_path}")
