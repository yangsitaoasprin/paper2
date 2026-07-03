"""
CMDM-Lab/CYP450 baseline run based on Experiment 6 logic.

This variant:
- uses 02_Data/cmdm_lab_cyp450_baseline_ready.csv
- saves results to a dedicated output file
- reports MCC in addition to AUC / PR-AUC / Accuracy / F1
"""

from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "0"

from bundle_paths import DATA_DIR, MODEL_DIR, RESULTS_DIR

RESULT_DIR = RESULTS_DIR
RESULT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATA_DIR / "cmdm_lab_cyp450_baseline_ready.csv"
RESULT_PATH = Path(os.getenv("CMDM_RESULT_PATH", str(RESULT_DIR / "cmdm_experiment6_drugonly_result.json")))
MODEL_PATH = Path(os.getenv("CMDM_MODEL_PATH", str(MODEL_DIR / "cmdm_experiment6_drugonly_best.pt")))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = int(os.getenv("CMDM_SEED", "42"))
EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-3
DROPOUT = 0.3
MORGAN_BITS = 2048
MORGAN_RADIUS = 2

RDLogger.DisableLog("rdApp.error")
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS,
    fpSize=MORGAN_BITS,
)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def mol_to_morgan_fp(smiles: str, radius: int = MORGAN_RADIUS, n_bits: int = MORGAN_BITS) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    if radius == MORGAN_RADIUS and n_bits == MORGAN_BITS:
        return np.array(MORGAN_GENERATOR.GetFingerprint(mol), dtype=np.float32)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return np.array(generator.GetFingerprint(mol), dtype=np.float32)


def load_data(csv_path: Path) -> dict[str, np.ndarray]:
    df = pd.read_csv(csv_path)
    labels = df["label"].values.astype(np.float32)
    drug_fps = np.array([mol_to_morgan_fp(s) for s in df["drug_smiles"]], dtype=np.float32)

    train_mask = df["split"] == "train"
    test_mask = df["split"] == "test"
    train_idx = np.where(train_mask)[0]
    np.random.shuffle(train_idx)
    n_val = int(0.2 * len(train_idx))
    val_idx = train_idx[:n_val]
    train_idx = train_idx[n_val:]
    test_idx = np.where(test_mask)[0]

    return {
        "drug_fps": drug_fps,
        "labels": labels,
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }


class DrugOnlyModel(nn.Module):
    def __init__(self, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(MORGAN_BITS, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, fp: torch.Tensor) -> torch.Tensor:
        return self.net(fp)


def train_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    drug_fps: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    np.random.shuffle(indices)

    for i in range(0, len(indices), batch_size):
        batch_idx = indices[i : i + batch_size]
        batch_fp = torch.FloatTensor(drug_fps[batch_idx]).to(DEVICE)
        batch_labels = torch.FloatTensor(labels[batch_idx]).unsqueeze(1).to(DEVICE)

        optimizer.zero_grad()
        logits = model(batch_fp)
        loss = criterion(logits, batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(
    model: nn.Module,
    drug_fps: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    batch_size: int = 256,
) -> dict[str, float]:
    model.eval()
    all_probs: list[float] = []

    with torch.no_grad():
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i : i + batch_size]
            batch_fp = torch.FloatTensor(drug_fps[batch_idx]).to(DEVICE)
            logits = model(batch_fp)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs.tolist())

    all_probs_arr = np.array(all_probs)
    true_labels = labels[indices]
    preds = (all_probs_arr >= 0.5).astype(int)

    return {
        "auc": float(roc_auc_score(true_labels, all_probs_arr)),
        "pr_auc": float(average_precision_score(true_labels, all_probs_arr)),
        "mcc": float(matthews_corrcoef(true_labels, preds)),
        "accuracy": float(accuracy_score(true_labels, preds)),
        "f1": float(f1_score(true_labels, preds)),
    }


def evaluate_per_isoform(model: nn.Module, df: pd.DataFrame, drug_fps: np.ndarray) -> list[dict[str, float | str | int]]:
    test_df = df[df["split"] == "test"].copy()
    summaries: list[dict[str, float | str | int]] = []
    for isoform_name, group in test_df.groupby("enzyme_name"):
        indices = group.index.to_numpy()
        labels = df["label"].values.astype(np.float32)
        metrics = evaluate(model, drug_fps, labels, indices)
        summaries.append(
            {
                "isoform_name": isoform_name,
                "rows": int(len(group)),
                "positives": int((group["label"] == 1).sum()),
                "negatives": int((group["label"] == 0).sum()),
                **metrics,
            }
        )
    return summaries


def main() -> None:
    set_seed(SEED)
    print("=" * 60)
    print("CMDM EXPERIMENT 6: Drug-Only Baseline")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Data: {DATA_PATH}")

    print("\n[1/4] Loading data...")
    df = pd.read_csv(DATA_PATH)
    data = load_data(DATA_PATH)
    print(f"  Train: {len(data['train'])} | Val: {len(data['val'])} | Test: {len(data['test'])}")
    print(f"  Positive rate - Train: {data['labels'][data['train']].mean():.3f} | Test: {data['labels'][data['test']].mean():.3f}")

    print("\n[2/4] Initializing model...")
    model = DrugOnlyModel(dropout=DROPOUT).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    pos_weight = (data["labels"][data["train"]] == 0).sum() / (data["labels"][data["train"]] == 1).sum()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(pos_weight)]).to(DEVICE))
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"\n[3/4] Training for {EPOCHS} epochs...")
    history = []
    best_val_auc = 0.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(
            model,
            optimizer,
            criterion,
            data["drug_fps"],
            data["labels"],
            data["train"].copy(),
            BATCH_SIZE,
        )
        val_metrics = evaluate(model, data["drug_fps"], data["labels"], data["val"])
        history.append({"epoch": epoch, "train_loss": train_loss, "val_auc": val_metrics["auc"]})

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:2d}: loss={train_loss:.4f}, val_auc={val_metrics['auc']:.4f}, val_mcc={val_metrics['mcc']:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    print("\n[4/4] Test evaluation...")
    test_metrics = evaluate(model, data["drug_fps"], data["labels"], data["test"])
    per_isoform = evaluate_per_isoform(model, df, data["drug_fps"])

    print(f"  Test AUC:      {test_metrics['auc']:.4f}")
    print(f"  Test PR-AUC:   {test_metrics['pr_auc']:.4f}")
    print(f"  Test MCC:      {test_metrics['mcc']:.4f}")
    print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Test F1:       {test_metrics['f1']:.4f}")

    result = {
        "experiment": "CMDM Experiment 6: Drug-Only Baseline",
        "data_path": str(DATA_PATH),
        "config": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "dropout": DROPOUT,
            "seed": SEED,
        },
        "parameters": {"total": total_params},
        "best_val_auc": float(best_val_auc),
        "test_metrics": test_metrics,
        "per_isoform_test_metrics": per_isoform,
        "history": history,
    }

    with RESULT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    torch.save(model.state_dict(), MODEL_PATH)

    print(f"\nResults saved to: {RESULT_PATH}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()

