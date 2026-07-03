"""
CMDM-Lab/CYP450 baseline run based on Experiment 10 logic.

This variant:
- uses 02_Data/cmdm_lab_cyp450_baseline_ready.csv
- replaces Morgan-style drug features with MACCS keys (167 bits)
- applies 3x oversampling to CYP2E1-positive samples in the training split
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
from rdkit.Chem import MACCSkeys
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
RESULT_PATH = RESULT_DIR / "cmdm_experiment10_twotowerv4_result.json"
MODEL_PATH = MODEL_DIR / "cmdm_experiment10_twotowerv4_best.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-3
DROPOUT = 0.3
MAX_SEQ_LEN = 512
AMINO_VOCAB = {a: i + 1 for i, a in enumerate("ACDEFGHIKLMNPQRSTVWY")}
ENZYME_MAP = {
    "CYP1A2": 0,
    "CYP2C9": 1,
    "CYP2C19": 2,
    "CYP2D6": 3,
    "CYP2E1": 4,
    "CYP3A4": 5,
}

RDLogger.DisableLog("rdApp.error")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def mol_to_maccs_fp(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(167, dtype=np.float32)
    fp = MACCSkeys.GenMACCSKeys(mol)
    return np.array(fp, dtype=np.float32)


def seq_to_indices(seq: str, max_len: int = MAX_SEQ_LEN) -> np.ndarray:
    indices = [AMINO_VOCAB.get(a, 0) for a in seq[:max_len]]
    if len(indices) < max_len:
        indices += [0] * (max_len - len(indices))
    return np.array(indices, dtype=np.int64)


def load_data_with_oversampling(
    csv_path: Path,
    oversample_enzyme: str = "CYP2E1",
    oversample_factor: int = 3,
) -> dict[str, object]:
    df = pd.read_csv(csv_path)
    df["enzyme_id"] = df["enzyme_name"].map(ENZYME_MAP)

    drug_features = np.array([mol_to_maccs_fp(s) for s in df["drug_smiles"]], dtype=np.float32)
    enzyme_seqs = np.array([seq_to_indices(seq) for seq in df["enzyme_seq"]], dtype=np.int64)
    enzyme_ids = df["enzyme_id"].values.astype(np.int64)
    labels = df["label"].values.astype(np.float32)

    train_mask = df["split"] == "train"
    test_mask = df["split"] == "test"
    train_idx = np.where(train_mask)[0]
    original_train_count = len(train_idx)
    np.random.shuffle(train_idx)
    n_val = int(0.2 * len(train_idx))
    val_idx = train_idx[:n_val]
    train_idx = train_idx[n_val:]
    base_train_count = len(train_idx)
    test_idx = np.where(test_mask)[0]

    train_df = df.loc[train_idx].reset_index(drop=True)
    oversample_mask = (train_df["enzyme_name"] == oversample_enzyme) & (train_df["label"] == 1)
    oversample_pos_idx = train_df[oversample_mask].index.to_numpy()

    duplicated_indices: list[int] = []
    for _ in range(oversample_factor):
        duplicated_indices.extend(oversample_pos_idx.tolist())
    if duplicated_indices:
        train_idx = np.concatenate([train_idx, train_idx[np.array(duplicated_indices, dtype=np.int64)]])
        np.random.shuffle(train_idx)

    added_count = len(oversample_pos_idx) * oversample_factor
    return {
        "df": df,
        "drug_features": drug_features,
        "enzyme_seqs": enzyme_seqs,
        "enzyme_ids": enzyme_ids,
        "labels": labels,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "oversampling": {
            "enzyme": oversample_enzyme,
            "factor": oversample_factor,
            "original_train_rows": original_train_count,
            "base_train_rows": base_train_count,
            "oversampled_positive_rows": int(len(oversample_pos_idx)),
            "added_rows": int(added_count),
            "final_train_rows": int(len(train_idx)),
        },
    }


class DrugTowerV4(nn.Module):
    def __init__(self, input_dim: int = 167, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class EnzymeTowerV4(nn.Module):
    def __init__(self, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.enzyme_embed = nn.Embedding(21, 128, padding_idx=0)
        self.conv1 = nn.Conv1d(128, 128, 7, padding=3)
        self.pool1 = nn.MaxPool1d(3)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.ec_embed = nn.Embedding(6, 32)
        self.fusion = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
        )

    def forward(self, x_seq: torch.Tensor, enzyme_ids: torch.Tensor) -> torch.Tensor:
        h = self.enzyme_embed(x_seq).transpose(1, 2)
        h = self.pool1(F.relu(self.conv1(h)))
        h = self.gap(h).squeeze(-1)
        ec = self.ec_embed(enzyme_ids.long())
        h = torch.cat([h, ec], dim=1)
        return self.fusion(h)


class TwoTowerV4(nn.Module):
    def __init__(self, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.drug_tower = DrugTowerV4(dropout=dropout)
        self.enzyme_tower = EnzymeTowerV4(dropout=dropout)
        self.predictor = nn.Sequential(
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, drug_x: torch.Tensor, enzyme_x: torch.Tensor, enzyme_ids: torch.Tensor) -> torch.Tensor:
        drug_feat = self.drug_tower(drug_x)
        enzyme_feat = self.enzyme_tower(enzyme_x, enzyme_ids)
        return self.predictor(torch.cat([drug_feat, enzyme_feat], dim=1))


def create_loader(
    drug_features: np.ndarray,
    enzyme_seqs: np.ndarray,
    enzyme_ids: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = True,
) -> torch.utils.data.DataLoader:
    tensors = [
        torch.FloatTensor(drug_features[indices]),
        torch.LongTensor(enzyme_seqs[indices]),
        torch.LongTensor(enzyme_ids[indices]),
        torch.FloatTensor(labels[indices]),
    ]
    dataset = torch.utils.data.TensorDataset(*tensors)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    loader: torch.utils.data.DataLoader,
) -> float:
    model.train()
    total_loss = 0.0
    for drug, enz_seq, enz_id, labels in loader:
        drug = drug.to(DEVICE)
        enz_seq = enz_seq.to(DEVICE)
        enz_id = enz_id.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(drug, enz_seq, enz_id).squeeze(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
) -> dict[str, float]:
    model.eval()
    all_probs: list[float] = []
    all_labels: list[float] = []
    with torch.no_grad():
        for drug, enz_seq, enz_id, labels in loader:
            drug = drug.to(DEVICE)
            enz_seq = enz_seq.to(DEVICE)
            enz_id = enz_id.to(DEVICE)
            outputs = model(drug, enz_seq, enz_id).squeeze(-1)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs.flatten().tolist())
            all_labels.extend(labels.numpy().tolist())
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


def evaluate_per_isoform(
    model: nn.Module,
    df: pd.DataFrame,
    drug_features: np.ndarray,
    enzyme_seqs: np.ndarray,
    enzyme_ids: np.ndarray,
    labels: np.ndarray,
) -> list[dict[str, float | int | str]]:
    test_df = df[df["split"] == "test"].copy()
    summaries: list[dict[str, float | int | str]] = []
    for isoform_name, group in test_df.groupby("enzyme_name"):
        indices = group.index.to_numpy()
        loader = create_loader(drug_features, enzyme_seqs, enzyme_ids, labels, indices, batch_size=256, shuffle=False)
        metrics = evaluate(model, loader)
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
    print("=" * 72)
    print("CMDM EXPERIMENT 10: Two-Tower V4 (MACCS + CYP2E1 3x Oversampling)")
    print("=" * 72)
    print(f"Device: {DEVICE}")
    print(f"Data:   {DATA_PATH}")

    print("\n[1/4] Loading data with oversampling...")
    data = load_data_with_oversampling(DATA_PATH, oversample_enzyme="CYP2E1", oversample_factor=3)
    df = data["df"]
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]
    drug_features = data["drug_features"]
    enzyme_seqs = data["enzyme_seqs"]
    enzyme_ids = data["enzyme_ids"]
    labels = data["labels"]
    oversampling = data["oversampling"]

    print(
        "  Train rows: "
        f"{oversampling['original_train_rows']} original -> "
        f"{oversampling['base_train_rows']} after val split -> "
        f"{oversampling['final_train_rows']} after oversampling"
    )
    print(
        f"  Oversampled {oversampling['enzyme']} positives: "
        f"{oversampling['oversampled_positive_rows']} x {oversampling['factor']} "
        f"(added {oversampling['added_rows']})"
    )
    print(f"  Val: {len(val_idx)} | Test: {len(test_idx)}")

    print("\n[2/4] Initializing model...")
    model = TwoTowerV4(dropout=DROPOUT).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,}")

    train_labels = labels[train_idx]
    pos_weight = torch.tensor(
        [(train_labels == 0).sum() / (train_labels == 1).sum()],
        dtype=torch.float32,
        device=DEVICE,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    train_loader = create_loader(drug_features, enzyme_seqs, enzyme_ids, labels, train_idx, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = create_loader(drug_features, enzyme_seqs, enzyme_ids, labels, val_idx, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = create_loader(drug_features, enzyme_seqs, enzyme_ids, labels, test_idx, batch_size=BATCH_SIZE, shuffle=False)

    print(f"\n[3/4] Training for {EPOCHS} epochs...")
    history = []
    best_val_auc = 0.0
    best_state = None
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, optimizer, criterion, train_loader)
        val_metrics = evaluate(model, val_loader)
        scheduler.step(val_metrics["auc"])
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
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:2d}: loss={train_loss:.4f}, "
                f"val_auc={val_metrics['auc']:.4f}, val_mcc={val_metrics['mcc']:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    print("\n[4/4] Evaluating on test set...")
    test_metrics = evaluate(model, test_loader)
    per_isoform = evaluate_per_isoform(model, df, drug_features, enzyme_seqs, enzyme_ids, labels)

    print(f"  Best val AUC:   {best_val_auc:.4f} (epoch {best_epoch})")
    print(f"  Test AUC:       {test_metrics['auc']:.4f}")
    print(f"  Test PR-AUC:    {test_metrics['pr_auc']:.4f}")
    print(f"  Test MCC:       {test_metrics['mcc']:.4f}")
    print(f"  Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Test F1:        {test_metrics['f1']:.4f}")

    result = {
        "experiment": "CMDM Experiment 10: Two-Tower V4 with MACCS and CYP2E1 3x Oversampling",
        "data_path": str(DATA_PATH),
        "config": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "dropout": DROPOUT,
            "seed": SEED,
            "drug_feature": "MACCS_167",
        },
        "oversampling": oversampling,
        "parameters": {"total": total_params},
        "best_epoch": best_epoch,
        "best_val_auc": float(best_val_auc),
        "test_metrics": test_metrics,
        "per_isoform_test_metrics": per_isoform,
        "history": history,
    }
    with RESULT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"\nResults saved to: {RESULT_PATH}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()

