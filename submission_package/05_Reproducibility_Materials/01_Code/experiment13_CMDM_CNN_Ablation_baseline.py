"""
CMDM-Lab/CYP450 baseline run based on Experiment 13 logic.

This variant:
- uses 02_Data/cmdm_lab_cyp450_baseline_ready.csv
- runs CNN-1 / CNN-2 / CNN-3 enzyme-side ablation
- saves results to dedicated CMDM output files
- reports MCC in addition to AUC / PR-AUC / Accuracy / F1
- summarizes per-isoform test metrics for each variant
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
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "0"
RDLogger.DisableLog("rdApp.error")

from bundle_paths import DATA_DIR, MODEL_DIR, RESULTS_DIR

RESULT_DIR = RESULTS_DIR
RESULT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATA_DIR / "cmdm_lab_cyp450_baseline_ready.csv"
SUMMARY_PATH = Path(os.getenv("CMDM_SUMMARY_PATH", str(RESULT_DIR / "cmdm_experiment13_cnn_ablation_summary.json")))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = int(os.getenv("CMDM_SEED", "42"))
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
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS,
    fpSize=MORGAN_BITS,
)
RESULT_TEMPLATE = os.getenv("CMDM_RESULT_TEMPLATE", "cmdm_experiment13_cnn{layer}_result.json")
MODEL_TEMPLATE = os.getenv("CMDM_MODEL_TEMPLATE", "cmdm_experiment13_cnn{layer}_best.pt")
SELECTED_LAYERS = [int(item.strip()) for item in os.getenv("CMDM_CNN_LAYERS", "1,2,3").split(",") if item.strip()]


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


def seq_to_indices(seq: str, max_len: int = MAX_SEQ_LEN) -> np.ndarray:
    indices = [AMINO_VOCAB.get(a, 0) for a in seq[:max_len]]
    if len(indices) < max_len:
        indices += [0] * (max_len - len(indices))
    return np.array(indices, dtype=np.int64)


def load_data(csv_path: Path) -> dict[str, object]:
    df = pd.read_csv(csv_path)
    df["enzyme_id"] = df["enzyme_name"].map(ENZYME_MAP)

    labels = df["label"].values.astype(np.float32)
    drug_fps = np.array([mol_to_morgan_fp(s) for s in df["drug_smiles"]], dtype=np.float32)
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
        "drug_fps": drug_fps,
        "enzyme_seqs": enzyme_seqs,
        "enzyme_ids": enzyme_ids,
        "labels": labels,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }


class DrugTower(nn.Module):
    def __init__(self, input_dim: int = 2056, dropout: float = DROPOUT) -> None:
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


class EnzymeTowerCNN(nn.Module):
    def __init__(self, num_cnn_layers: int = 2, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.enzyme_embed = nn.Embedding(21, 128, padding_idx=0)

        channels = [128, 256, 512]
        layers: list[nn.Module] = []
        in_ch = 128
        for i in range(num_cnn_layers):
            out_ch = channels[i]
            layers.extend(
                [
                    nn.Conv1d(in_ch, out_ch, kernel_size=7, padding=3),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.MaxPool1d(3),
                ]
            )
            in_ch = out_ch

        self.cnn = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.ec_embed = nn.Embedding(6, 32)

        cnn_out_ch = channels[num_cnn_layers - 1]
        self.fusion = nn.Sequential(
            nn.Linear(cnn_out_ch + 32, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
        )

    def forward(self, x_seq: torch.Tensor, enzyme_ids: torch.Tensor) -> torch.Tensor:
        h = self.enzyme_embed(x_seq).transpose(1, 2)
        h = self.cnn(h)
        h = self.global_pool(h).squeeze(-1)
        ec = self.ec_embed(enzyme_ids.long())
        h = torch.cat([h, ec], dim=1)
        return self.fusion(h)


class Predictor(nn.Module):
    def __init__(self, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, drug_feat: torch.Tensor, enzyme_feat: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([drug_feat, enzyme_feat], dim=1)
        return self.mlp(combined)


class TwoTowerCNN(nn.Module):
    def __init__(self, num_cnn_layers: int = 2) -> None:
        super().__init__()
        self.drug_tower = DrugTower()
        self.enzyme_tower = EnzymeTowerCNN(num_cnn_layers=num_cnn_layers)
        self.predictor = Predictor()

    def forward(self, drug_x: torch.Tensor, enzyme_seq: torch.Tensor, enzyme_id: torch.Tensor) -> torch.Tensor:
        drug_feat = self.drug_tower(drug_x)
        enzyme_feat = self.enzyme_tower(enzyme_seq, enzyme_id)
        return self.predictor(drug_feat, enzyme_feat)


def create_loader(
    drug_fps: np.ndarray,
    enzyme_seqs: np.ndarray,
    enzyme_ids: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = True,
) -> torch.utils.data.DataLoader:
    tensors = [
        torch.FloatTensor(drug_fps[indices]),
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
    data: dict[str, object],
) -> list[dict[str, float | int | str]]:
    df = data["df"]
    drug_fps = data["drug_fps"]
    enzyme_seqs = data["enzyme_seqs"]
    enzyme_ids = data["enzyme_ids"]
    labels = data["labels"]
    test_df = df[df["split"] == "test"].copy()
    summaries: list[dict[str, float | int | str]] = []
    for isoform_name, group in test_df.groupby("enzyme_name"):
        indices = group.index.to_numpy()
        loader = create_loader(drug_fps, enzyme_seqs, enzyme_ids, labels, indices, batch_size=256, shuffle=False)
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


def run_experiment(num_cnn_layers: int, data: dict[str, object], epochs: int = EPOCHS, batch_size: int = BATCH_SIZE, lr: float = LR) -> dict[str, object]:
    print(f"\n{'=' * 60}")
    print(f"Training CNN-{num_cnn_layers} on CMDM...")
    print(f"{'=' * 60}")

    set_seed(SEED)
    model = TwoTowerCNN(num_cnn_layers=num_cnn_layers).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    train_labels = data["labels"][data["train_idx"]]
    pos_weight = (train_labels == 0).sum() / (train_labels == 1).sum()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(pos_weight)]).to(DEVICE))
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader = create_loader(data["drug_fps"], data["enzyme_seqs"], data["enzyme_ids"], data["labels"], data["train_idx"], batch_size=batch_size, shuffle=True)
    val_loader = create_loader(data["drug_fps"], data["enzyme_seqs"], data["enzyme_ids"], data["labels"], data["val_idx"], batch_size=256, shuffle=False)
    test_loader = create_loader(data["drug_fps"], data["enzyme_seqs"], data["enzyme_ids"], data["labels"], data["test_idx"], batch_size=256, shuffle=False)

    history = []
    best_val_auc = 0.0
    best_epoch = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, optimizer, criterion, train_loader)
        val_metrics = evaluate(model, val_loader)
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

    print(f"\n  Best val AUC: {best_val_auc:.4f} (epoch {best_epoch})")

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader)
    per_isoform = evaluate_per_isoform(model, data)

    print(f"  Test AUC:      {test_metrics['auc']:.4f}")
    print(f"  Test PR-AUC:   {test_metrics['pr_auc']:.4f}")
    print(f"  Test MCC:      {test_metrics['mcc']:.4f}")
    print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Test F1:       {test_metrics['f1']:.4f}")

    result = {
        "model": f"CNN-{num_cnn_layers}",
        "data_path": str(DATA_PATH),
        "parameters": total_params,
        "best_epoch": best_epoch,
        "best_val_auc": float(best_val_auc),
        "test_metrics": test_metrics,
        "per_isoform_test_metrics": per_isoform,
        "history": history,
    }

    result_path = RESULT_DIR / RESULT_TEMPLATE.format(layer=num_cnn_layers)
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    model_path = MODEL_DIR / MODEL_TEMPLATE.format(layer=num_cnn_layers)
    torch.save(model.state_dict(), model_path)

    return result


def load_existing_result(num_cnn_layers: int) -> dict[str, object] | None:
    result_path = RESULT_DIR / RESULT_TEMPLATE.format(layer=num_cnn_layers)
    if not result_path.exists():
        return None
    with result_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    print("=" * 60)
    print("CMDM EXPERIMENT 13: CNN Layer Ablation")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Data: {DATA_PATH}")

    print("\n[1/2] Loading data...")
    data = load_data(DATA_PATH)
    print(f"  Train: {len(data['train_idx'])} | Val: {len(data['val_idx'])} | Test: {len(data['test_idx'])}")
    print(f"  Enzyme sequences: {data['enzyme_seqs'].shape} (real amino acid indices)")

    print("\n[2/2] Training CNN variants...")
    results: dict[int, dict[str, object]] = {}
    for num_layers in SELECTED_LAYERS:
        existing = load_existing_result(num_layers)
        if existing is not None:
            print(f"\nReusing existing result for CNN-{num_layers}: {RESULT_DIR / RESULT_TEMPLATE.format(layer=num_layers)}")
            results[num_layers] = existing
            continue
        results[num_layers] = run_experiment(num_layers, data, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR)

    summary = {
        "experiment": "CMDM Experiment 13: CNN Layer Ablation",
        "data_path": str(DATA_PATH),
        "variants": results,
    }
    with SUMMARY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\n" + "=" * 72)
    print("SUMMARY: CMDM CNN Layer Ablation")
    print("=" * 72)
    print(f"\n{'Config':<8} {'Params':<12} {'Best Val':<10} {'Test AUC':<10} {'Test PR':<10} {'Test MCC':<10}")
    print("-" * 72)
    for num_layers in SELECTED_LAYERS:
        r = results[num_layers]
        print(
            f"{r['model']:<8} {r['parameters']:<12,} {r['best_val_auc']:<10.4f} "
            f"{r['test_metrics']['auc']:<10.4f} {r['test_metrics']['pr_auc']:<10.4f} "
            f"{r['test_metrics']['mcc']:<10.4f}"
        )


if __name__ == "__main__":
    main()

