"""
CMDM-Lab/CYP450 baseline run based on Experiment 11 logic.

This variant:
- uses 02_Data/cmdm_lab_cyp450_baseline_ready.csv
- extends drug features with Balaban J, Bertz complexity, and SMARTS counts
- learns enzyme-specific weights and biases on top of the shared two-tower backbone
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
from rdkit.Chem import Descriptors, GraphDescriptors, rdFingerprintGenerator
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
RESULT_PATH = RESULT_DIR / "cmdm_experiment11_twotowerv5_result.json"
MODEL_PATH = MODEL_DIR / "cmdm_experiment11_twotowerv5_best.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-3
DROPOUT = 0.3
MAX_SEQ_LEN = 512
MORGAN_RADIUS = 2
MORGAN_BITS = 2048
AMINO_VOCAB = {a: i + 1 for i, a in enumerate("ACDEFGHIKLMNPQRSTVWY")}
ENZYME_MAP = {
    "CYP1A2": 0,
    "CYP2C9": 1,
    "CYP2C19": 2,
    "CYP2D6": 3,
    "CYP2E1": 4,
    "CYP3A4": 5,
}
ENZYME_INIT_WEIGHTS = [1.11, 2.34, 1.89, 3.12, 5.88, 2.56]
SMARTS_PATTERNS = [
    ("aromatic_ring", "c1ccccc1"),
    ("heteroatom_aromatic", "c1cc[n,o,s]cc1"),
    ("hbd_oh", "[OH]"),
    ("hba_n", "[NH0]"),
    ("halogen", "[F,Cl,Br,I]"),
    ("carbonyl", "C=O"),
    ("nitro", "N(=O)=O"),
    ("sulfur", "[S]"),
]

RDLogger.DisableLog("rdApp.error")
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS,
    fpSize=MORGAN_BITS,
)
SMARTS_COMPILED = [(name, Chem.MolFromSmarts(smarts)) for name, smarts in SMARTS_PATTERNS]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def mol_to_smarts_counts(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(len(SMARTS_COMPILED), dtype=np.float32)
    counts = []
    for _, patt in SMARTS_COMPILED:
        if patt is None:
            counts.append(0)
        else:
            counts.append(len(mol.GetSubstructMatches(patt)))
    return np.array(counts, dtype=np.float32)


def mol_to_features_v5(smiles: str, radius: int = MORGAN_RADIUS, n_bits: int = MORGAN_BITS) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        fp = np.zeros(n_bits, dtype=np.float32)
        desc = np.zeros(8, dtype=np.float32)
        balaban = np.zeros(1, dtype=np.float32)
        bertz = np.zeros(1, dtype=np.float32)
        smarts = np.zeros(len(SMARTS_COMPILED), dtype=np.float32)
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
        try:
            balaban = np.array([GraphDescriptors.BalabanJ(mol)], dtype=np.float32)
        except Exception:
            balaban = np.zeros(1, dtype=np.float32)
        try:
            bertz = np.array([GraphDescriptors.BertzCT(mol)], dtype=np.float32)
        except Exception:
            bertz = np.zeros(1, dtype=np.float32)
        smarts = mol_to_smarts_counts(smiles)
    return np.concatenate([fp, desc, balaban, bertz, smarts])


def seq_to_indices(seq: str, max_len: int = MAX_SEQ_LEN) -> np.ndarray:
    indices = [AMINO_VOCAB.get(a, 0) for a in seq[:max_len]]
    if len(indices) < max_len:
        indices += [0] * (max_len - len(indices))
    return np.array(indices, dtype=np.int64)


def load_data(csv_path: Path) -> dict[str, object]:
    df = pd.read_csv(csv_path)
    df["enzyme_id"] = df["enzyme_name"].map(ENZYME_MAP)
    labels = df["label"].values.astype(np.float32)
    drug_features = np.array([mol_to_features_v5(s) for s in df["drug_smiles"]], dtype=np.float32)
    enzyme_seqs = np.array([seq_to_indices(seq) for seq in df["enzyme_seq"]], dtype=np.int64)
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
        "enzyme_seqs": enzyme_seqs,
        "enzyme_ids": enzyme_ids,
        "labels": labels,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }


class DrugTowerV5(nn.Module):
    def __init__(self, input_dim: int = 2066, dropout: float = DROPOUT) -> None:
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


class EnzymeTowerV5(nn.Module):
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


class TwoTowerV5(nn.Module):
    def __init__(self, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.drug_tower = DrugTowerV5(dropout=dropout)
        self.enzyme_tower = EnzymeTowerV5(dropout=dropout)
        self.enzyme_weights = nn.Parameter(torch.tensor(ENZYME_INIT_WEIGHTS, dtype=torch.float32))
        self.enzyme_bias = nn.Parameter(torch.zeros(6, dtype=torch.float32))
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
        combined = torch.cat([drug_feat, enzyme_feat], dim=1)
        logits = self.predictor(combined).squeeze(-1)
        weights = self.enzyme_weights[enzyme_ids]
        bias = self.enzyme_bias[enzyme_ids]
        return logits * weights + bias


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
        outputs = model(drug, enz_seq, enz_id)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader) -> dict[str, float]:
    model.eval()
    all_probs: list[float] = []
    all_labels: list[float] = []
    with torch.no_grad():
        for drug, enz_seq, enz_id, labels in loader:
            drug = drug.to(DEVICE)
            enz_seq = enz_seq.to(DEVICE)
            enz_id = enz_id.to(DEVICE)
            outputs = model(drug, enz_seq, enz_id)
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
    print("CMDM EXPERIMENT 11: Two-Tower V5 (Enzyme Weights + SMARTS + Topology)")
    print("=" * 72)
    print(f"Device: {DEVICE}")
    print(f"Data:   {DATA_PATH}")

    print("\n[1/4] Loading data...")
    data = load_data(DATA_PATH)
    df = data["df"]
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]
    drug_features = data["drug_features"]
    enzyme_seqs = data["enzyme_seqs"]
    enzyme_ids = data["enzyme_ids"]
    labels = data["labels"]

    print(f"  Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
    print(f"  Drug feature dim: {drug_features.shape[1]}")

    print("\n[2/4] Initializing model...")
    model = TwoTowerV5(dropout=DROPOUT).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,}")
    print(f"  Enzyme weights init: {ENZYME_INIT_WEIGHTS}")

    train_labels = labels[train_idx]
    pos_weight = torch.tensor(
        [(train_labels == 0).sum() / (train_labels == 1).sum()],
        dtype=torch.float32,
        device=DEVICE,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    train_loader = create_loader(drug_features, enzyme_seqs, enzyme_ids, labels, train_idx, shuffle=True)
    val_loader = create_loader(drug_features, enzyme_seqs, enzyme_ids, labels, val_idx, shuffle=False)
    test_loader = create_loader(drug_features, enzyme_seqs, enzyme_ids, labels, test_idx, shuffle=False)

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
    learned_weights = model.enzyme_weights.detach().cpu().numpy().tolist()
    learned_bias = model.enzyme_bias.detach().cpu().numpy().tolist()

    print(f"  Best val AUC:   {best_val_auc:.4f} (epoch {best_epoch})")
    print(f"  Test AUC:       {test_metrics['auc']:.4f}")
    print(f"  Test PR-AUC:    {test_metrics['pr_auc']:.4f}")
    print(f"  Test MCC:       {test_metrics['mcc']:.4f}")
    print(f"  Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Test F1:        {test_metrics['f1']:.4f}")
    print(f"  Learned weights: {[round(v, 3) for v in learned_weights]}")
    print(f"  Learned bias:    {[round(v, 4) for v in learned_bias]}")

    result = {
        "experiment": "CMDM Experiment 11: Two-Tower V5 with Enzyme Weights, SMARTS, and Topology",
        "data_path": str(DATA_PATH),
        "config": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "dropout": DROPOUT,
            "seed": SEED,
            "drug_feature": "Morgan2048+8desc+BalabanJ+Bertz+8SMARTS",
        },
        "parameters": {"total": total_params},
        "best_epoch": best_epoch,
        "best_val_auc": float(best_val_auc),
        "test_metrics": test_metrics,
        "learned_weights_init": ENZYME_INIT_WEIGHTS,
        "learned_weights": learned_weights,
        "learned_bias": learned_bias,
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

