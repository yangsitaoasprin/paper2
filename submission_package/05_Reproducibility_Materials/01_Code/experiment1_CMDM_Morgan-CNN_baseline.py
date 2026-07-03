"""
CMDM-Lab/CYP450 baseline run based on Experiment 1 logic.

This variant:
- uses 02_Data/cmdm_lab_cyp450_baseline_ready.csv
- saves results to dedicated CMDM output files
- reports MCC in addition to AUC / PR-AUC / Accuracy / F1
- summarizes per-isoform test metrics
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
import torch.nn.functional as F
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


from bundle_paths import DATA_DIR, MODEL_DIR, RESULTS_DIR

RESULT_DIR = RESULTS_DIR
RESULT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATA_DIR / "cmdm_lab_cyp450_baseline_ready.csv"
RESULT_PATH = Path(os.getenv("CMDM_RESULT_PATH", str(RESULT_DIR / "cmdm_experiment1_morgancnn_result.json")))
MODEL_PATH = Path(os.getenv("CMDM_MODEL_PATH", str(MODEL_DIR / "cmdm_experiment1_morgancnn_best.pt")))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = int(os.getenv("CMDM_SEED", "42"))
EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-3
DROPOUT = 0.3
MORGAN_BITS = 2048
MORGAN_RADIUS = 2
MAX_SEQ_LEN = 512

RDLogger.DisableLog("rdApp.error")
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS,
    fpSize=MORGAN_BITS,
)
AMINO_VOCAB = {a: i + 1 for i, a in enumerate("ACDEFGHIKLMNPQRSTVWY")}


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


def seq_to_indices(seq: str, max_len: int = MAX_SEQ_LEN) -> np.ndarray:
    indices = [AMINO_VOCAB.get(a, 0) for a in seq[:max_len]]
    if len(indices) < max_len:
        indices += [0] * (max_len - len(indices))
    return np.array(indices, dtype=np.int32)


def load_data(csv_path: Path) -> dict[str, object]:
    df = pd.read_csv(csv_path)
    drug_fps = np.array([mol_to_morgan_fp(s) for s in df["drug_smiles"]], dtype=np.float32)
    enzyme_seqs = np.array([seq_to_indices(seq) for seq in df["enzyme_seq"]], dtype=np.int32)
    labels = df["label"].values.astype(np.float32)

    train_mask = df["split"] == "train"
    test_mask = df["split"] == "test"

    train_idx = np.where(train_mask)[0]
    np.random.shuffle(train_idx)
    n_val = int(0.2 * len(train_idx))
    val_idx = train_idx[:n_val]
    train_idx = train_idx[n_val:]
    test_idx = np.where(test_mask)[0]

    return {
        "df": df,
        "drug_fps": drug_fps,
        "enzyme_seqs": enzyme_seqs,
        "labels": labels,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }


class CYP450Dataset(torch.utils.data.Dataset):
    def __init__(self, fps: np.ndarray, seqs: np.ndarray, labels: np.ndarray) -> None:
        self.fps = fps
        self.seqs = seqs
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.fps[i], dtype=torch.float32),
            torch.tensor(self.seqs[i], dtype=torch.long),
            torch.tensor(self.labels[i], dtype=torch.float32),
        )


class SimpleModel(nn.Module):
    def __init__(self, dropout: float = DROPOUT) -> None:
        super().__init__()
        vocab_size = 21

        self.drug_encoder = nn.Sequential(
            nn.Linear(MORGAN_BITS, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.enzyme_embed = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.enzyme_conv1 = nn.Conv1d(128, 128, 7, padding=3)
        self.enzyme_pool1 = nn.MaxPool1d(3)
        self.enzyme_conv2 = nn.Conv1d(128, 128, 5, padding=2)
        self.enzyme_pool2 = nn.MaxPool1d(3)
        self.enzyme_fc = nn.Linear(128 * 56, 128)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, fp: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        drug_emb = self.drug_encoder(fp)

        x = self.enzyme_embed(seq).transpose(1, 2)
        x = self.enzyme_pool1(F.relu(self.enzyme_conv1(x)))
        x = self.enzyme_pool2(F.relu(self.enzyme_conv2(x)))
        x = x.view(x.size(0), -1)
        enzyme_emb = self.enzyme_fc(x)

        combined = torch.cat([drug_emb, enzyme_emb], dim=1)
        return self.classifier(combined).squeeze(1)


def train_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch_fps, batch_seqs, batch_labels in train_loader:
        batch_fps = batch_fps.to(device)
        batch_seqs = batch_seqs.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        outputs = model(batch_fps, batch_seqs)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch_labels)

    return total_loss / len(train_loader.dataset)


def evaluate(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    all_probs: list[float] = []
    all_labels: list[float] = []

    with torch.no_grad():
        for batch_fps, batch_seqs, batch_labels in data_loader:
            batch_fps = batch_fps.to(device)
            batch_seqs = batch_seqs.to(device)
            outputs = model(batch_fps, batch_seqs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(batch_labels.numpy().tolist())

    all_probs_arr = np.array(all_probs)
    all_labels_arr = np.array(all_labels)
    preds = (all_probs_arr >= 0.5).astype(int)

    return {
        "auc": float(roc_auc_score(all_labels_arr, all_probs_arr)),
        "pr_auc": float(average_precision_score(all_labels_arr, all_probs_arr)),
        "mcc": float(matthews_corrcoef(all_labels_arr, preds)),
        "accuracy": float(accuracy_score(all_labels_arr, preds)),
        "f1": float(f1_score(all_labels_arr, preds)),
    }


def build_loader(
    drug_fps: np.ndarray,
    enzyme_seqs: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool = False,
) -> torch.utils.data.DataLoader:
    dataset = CYP450Dataset(drug_fps[indices], enzyme_seqs[indices], labels[indices])
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_per_isoform(
    model: nn.Module,
    df: pd.DataFrame,
    drug_fps: np.ndarray,
    enzyme_seqs: np.ndarray,
    labels: np.ndarray,
) -> list[dict[str, float | int | str]]:
    test_df = df[df["split"] == "test"].copy()
    summaries: list[dict[str, float | int | str]] = []

    for isoform_name, group in test_df.groupby("enzyme_name"):
        indices = group.index.to_numpy()
        iso_loader = build_loader(drug_fps, enzyme_seqs, labels, indices, batch_size=256, shuffle=False)
        metrics = evaluate(model, iso_loader, DEVICE)
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
    print("CMDM EXPERIMENT 1: Morgan-CNN Baseline")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Data: {DATA_PATH}")

    print("\n[1/4] Loading and preprocessing data...")
    data = load_data(DATA_PATH)
    df = data["df"]
    drug_fps = data["drug_fps"]
    enzyme_seqs = data["enzyme_seqs"]
    labels = data["labels"]
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]

    print(f"  Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
    print(f"  Positive rate - Train: {labels[train_idx].mean():.3f} | Test: {labels[test_idx].mean():.3f}")

    train_loader = build_loader(drug_fps, enzyme_seqs, labels, train_idx, BATCH_SIZE, shuffle=True)
    val_loader = build_loader(drug_fps, enzyme_seqs, labels, val_idx, BATCH_SIZE, shuffle=False)
    test_loader = build_loader(drug_fps, enzyme_seqs, labels, test_idx, BATCH_SIZE, shuffle=False)

    print("\n[2/4] Initializing model...")
    model = SimpleModel().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    pos_weight = (labels[train_idx] == 0).sum() / (labels[train_idx] == 1).sum()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(pos_weight)]).to(DEVICE))
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"\n[3/4] Training for {EPOCHS} epochs...")
    history = []
    best_val_auc = 0.0
    best_epoch = 0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, optimizer, criterion, train_loader, DEVICE)
        val_metrics = evaluate(model, val_loader, DEVICE)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_auc": val_metrics["auc"],
                "val_pr_auc": val_metrics["pr_auc"],
                "val_mcc": val_metrics["mcc"],
            }
        )

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:2d}: loss={train_loss:.4f}, "
                f"val_auc={val_metrics['auc']:.4f}, val_mcc={val_metrics['mcc']:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    print("\n[4/4] Evaluating on test set...")
    test_metrics = evaluate(model, test_loader, DEVICE)
    per_isoform = evaluate_per_isoform(model, df, drug_fps, enzyme_seqs, labels)

    print(f"  Best validation AUC: {best_val_auc:.4f} (epoch {best_epoch})")
    print(f"  Test AUC:      {test_metrics['auc']:.4f}")
    print(f"  Test PR-AUC:   {test_metrics['pr_auc']:.4f}")
    print(f"  Test MCC:      {test_metrics['mcc']:.4f}")
    print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Test F1:       {test_metrics['f1']:.4f}")

    result = {
        "experiment": "CMDM Experiment 1: Morgan-CNN Baseline",
        "data_path": str(DATA_PATH),
        "config": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "dropout": DROPOUT,
            "seed": SEED,
        },
        "parameters": {"total": total_params},
        "best_epoch": best_epoch,
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

