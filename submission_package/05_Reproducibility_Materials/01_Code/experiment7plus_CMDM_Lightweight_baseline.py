"""
CMDM-Lab/CYP450 baseline run based on Experiment 7-plus logic.

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
import torch.optim as optim
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
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
RESULT_PATH = Path(os.getenv("CMDM_RESULT_PATH", str(RESULT_DIR / "cmdm_experiment7plus_lightweight_result.json")))
MODEL_PATH = Path(os.getenv("CMDM_MODEL_PATH", str(MODEL_DIR / "cmdm_experiment7plus_lightweight_best.pt")))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = int(os.getenv("CMDM_SEED", "42"))

EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-3
DROPOUT = 0.3
MORGAN_RADIUS = 2
MORGAN_BITS = 2048
ENZYME_MAP = {
    "CYP1A2": 0,
    "CYP2C9": 1,
    "CYP2C19": 2,
    "CYP2D6": 3,
    "CYP2E1": 4,
    "CYP3A4": 5,
}

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


def mol_to_morgan_fp_with_desc(smiles: str, radius: int = MORGAN_RADIUS, n_bits: int = MORGAN_BITS) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        fp = np.zeros(n_bits, dtype=np.float32)
        desc = np.zeros(8, dtype=np.float32)
    else:
        if radius == MORGAN_RADIUS and n_bits == MORGAN_BITS:
            fp = np.array(MORGAN_GENERATOR.GetFingerprint(mol), dtype=np.float32)
        else:
            generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
            fp = np.array(generator.GetFingerprint(mol), dtype=np.float32)
        desc = np.array(
            [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumRotatableBonds(mol),
                len(Chem.GetSymmSSSR(mol)),
                Descriptors.FractionCSP3(mol),
            ],
            dtype=np.float32,
        )
    return np.concatenate([fp, desc])


def load_data(csv_path: Path) -> dict[str, object]:
    df = pd.read_csv(csv_path)
    df["enzyme_id"] = df["enzyme_name"].map(ENZYME_MAP)

    drug_features = np.array([mol_to_morgan_fp_with_desc(s) for s in df["drug_smiles"]], dtype=np.float32)
    labels = df["label"].values.astype(np.float32)
    enzyme_ids = df["enzyme_id"].values.astype(np.int64)

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
        "drug_features": drug_features,
        "enzyme_ids": enzyme_ids,
        "labels": labels,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }


class LightweightModel(nn.Module):
    def __init__(self, drug_dim: int = 2056, embed_dim: int = 128, n_enzymes: int = 6, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.drug_proj = nn.Linear(drug_dim, embed_dim)
        self.enzyme_embed = nn.Embedding(n_enzymes, 32)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim + 32, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, drug_x: torch.Tensor, enzyme_ids: torch.Tensor) -> torch.Tensor:
        drug_feat = self.drug_proj(drug_x)
        enz_feat = self.enzyme_embed(enzyme_ids.long())
        combined = torch.cat([drug_feat, enz_feat], dim=1)
        return self.classifier(combined)


def train_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    drug_x: np.ndarray,
    enzyme_ids: np.ndarray,
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
        batch_drug = torch.FloatTensor(drug_x[batch_idx]).to(DEVICE)
        batch_enz = torch.LongTensor(enzyme_ids[batch_idx]).to(DEVICE)
        batch_labels = torch.FloatTensor(labels[batch_idx]).unsqueeze(1).to(DEVICE)

        optimizer.zero_grad()
        outputs = model(batch_drug, batch_enz)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(
    model: nn.Module,
    drug_x: np.ndarray,
    enzyme_ids: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    batch_size: int = 256,
) -> dict[str, float]:
    model.eval()
    all_probs: list[float] = []

    with torch.no_grad():
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i : i + batch_size]
            batch_drug = torch.FloatTensor(drug_x[batch_idx]).to(DEVICE)
            batch_enz = torch.LongTensor(enzyme_ids[batch_idx]).to(DEVICE)

            logits = model(batch_drug, batch_enz)
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


def evaluate_per_isoform(
    model: nn.Module,
    df: pd.DataFrame,
    drug_features: np.ndarray,
    enzyme_ids: np.ndarray,
    labels: np.ndarray,
) -> list[dict[str, float | int | str]]:
    test_df = df[df["split"] == "test"].copy()
    summaries: list[dict[str, float | int | str]] = []

    for isoform_name, group in test_df.groupby("enzyme_name"):
        indices = group.index.to_numpy()
        metrics = evaluate(model, drug_features, enzyme_ids, labels, indices)
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
    print("CMDM EXPERIMENT 7-PLUS: Lightweight Design")
    print("=" * 60)
    print("Drug: Morgan+Descriptors | Enzyme: ID embedding only")
    print(f"Device: {DEVICE}")
    print(f"Data: {DATA_PATH}")

    print("\n[1/4] Loading data...")
    data = load_data(DATA_PATH)
    df = data["df"]
    print(f"  Train: {len(data['train_idx'])} | Val: {len(data['val_idx'])} | Test: {len(data['test_idx'])}")
    print(
        f"  Positive rate - Train: {data['labels'][data['train_idx']].mean():.3f} | "
        f"Test: {data['labels'][data['test_idx']].mean():.3f}"
    )

    print("\n[2/4] Initializing model...")
    model = LightweightModel(dropout=DROPOUT).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    print("  Architecture: Drug(2056->128) + EnzymeID(6x32) -> concat(160->128->1)")

    pos_weight = (data["labels"][data["train_idx"]] == 0).sum() / (data["labels"][data["train_idx"]] == 1).sum()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(pos_weight)]).to(DEVICE))
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"\n[3/4] Training for {EPOCHS} epochs...")
    history = []
    best_val_auc = 0.0
    best_epoch = 0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(
            model,
            optimizer,
            criterion,
            data["drug_features"],
            data["enzyme_ids"],
            data["labels"],
            data["train_idx"].copy(),
            BATCH_SIZE,
        )
        val_metrics = evaluate(model, data["drug_features"], data["enzyme_ids"], data["labels"], data["val_idx"])

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

    print(f"\n  Best validation AUC: {best_val_auc:.4f} (epoch {best_epoch})")

    print("\n[4/4] Test evaluation...")
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, data["drug_features"], data["enzyme_ids"], data["labels"], data["test_idx"])
    per_isoform = evaluate_per_isoform(model, df, data["drug_features"], data["enzyme_ids"], data["labels"])

    print(f"  Test AUC:      {test_metrics['auc']:.4f}")
    print(f"  Test PR-AUC:   {test_metrics['pr_auc']:.4f}")
    print(f"  Test MCC:      {test_metrics['mcc']:.4f}")
    print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Test F1:       {test_metrics['f1']:.4f}")

    result = {
        "experiment": "CMDM Experiment 7-plus: Lightweight Design (Morgan+Desc + EnzymeID)",
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
    print(f"\nResults saved to: {RESULT_PATH}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()

