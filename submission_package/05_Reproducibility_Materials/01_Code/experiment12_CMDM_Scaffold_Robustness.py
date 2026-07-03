"""
CMDM-Lab/CYP450 scaffold robustness benchmark.

This variant:
- uses 02_Data/cmdm_lab_cyp450_baseline_ready.csv
- evaluates scaffold split ratio sensitivity on the new-paper CMDM model set
- excludes old-paper ESM routes to keep the evidence chain aligned
- saves dedicated CMDM result and figure artifacts
"""

from __future__ import annotations

import json
import os
import random
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "0"
RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")

from bundle_paths import DATA_DIR, DIAGNOSTIC_FIGURE_DIR, RESULTS_DIR

RESULT_DIR = RESULTS_DIR
FIGURE_DIR = DIAGNOSTIC_FIGURE_DIR
RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATA_DIR / "cmdm_lab_cyp450_baseline_ready.csv"
RESULT_PATH = RESULT_DIR / "cmdm_experiment12_scaffold_robustness.json"
PARTIAL_RESULT_PATH = RESULT_DIR / "cmdm_experiment12_scaffold_robustness.partial.json"
DEBUG_PROGRESS_PATH = RESULT_DIR / "cmdm_experiment12_scaffold_robustness.ndjson"
FIGURE_PATH = FIGURE_DIR / "cmdm_scaffold_robustness.png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
EPOCHS = 30
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


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


@lru_cache(maxsize=20000)
def smiles_to_mol(smiles: str) -> Chem.Mol | None:
    return Chem.MolFromSmiles(smiles)


@lru_cache(maxsize=20000)
def get_cached_scaffold(smiles: str) -> str:
    mol = smiles_to_mol(smiles)
    if mol is None:
        return ""
    try:
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold_mol is None:
            return smiles
        scaffold = Chem.MolToSmiles(scaffold_mol)
        return scaffold if scaffold else smiles
    except Exception:
        return smiles


def append_debug_progress(payload: dict[str, object]) -> None:
    with open(DEBUG_PROGRESS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def mol_to_morgan_fp(smiles: str, radius: int = MORGAN_RADIUS, n_bits: int = MORGAN_BITS) -> np.ndarray:
    mol = smiles_to_mol(smiles)
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


def scaffold_split_ratio(df: pd.DataFrame, train_ratio: float, val_ratio: float, seed: int = SEED) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    tmp = df.copy()
    tmp["scaffold"] = tmp["drug_smiles"].apply(get_cached_scaffold)
    scaffolds = list(tmp.groupby("scaffold").groups.keys())
    rng.shuffle(scaffolds)
    n = len(scaffolds)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    train_s = set(scaffolds[:n_train])
    val_s = set(scaffolds[n_train : n_train + n_val])
    test_s = set(scaffolds[n_train + n_val :])
    return (
        tmp[tmp["scaffold"].isin(train_s)].index.to_numpy(),
        tmp[tmp["scaffold"].isin(val_s)].index.to_numpy(),
        tmp[tmp["scaffold"].isin(test_s)].index.to_numpy(),
    )


class DrugOnlyModel(nn.Module):
    def __init__(self, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, fp: torch.Tensor) -> torch.Tensor:
        return self.net(fp).squeeze(-1)


class MorganCNN(nn.Module):
    def __init__(self, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.drug_encoder = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.enzyme_embed = nn.Embedding(21, 128, padding_idx=0)
        self.conv1 = nn.Conv1d(128, 128, 7, padding=3)
        self.pool1 = nn.MaxPool1d(3)
        self.conv2 = nn.Conv1d(128, 128, 5, padding=2)
        self.pool2 = nn.MaxPool1d(3)
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
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.reshape(x.size(0), -1)
        enzyme_emb = self.enzyme_fc(x)
        return self.classifier(torch.cat([drug_emb, enzyme_emb], dim=1)).squeeze(-1)


class TwoTowerV1(nn.Module):
    def __init__(self, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.drug_tower = nn.Sequential(
            nn.Linear(2056, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 32),
        )
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
        self.predictor = nn.Sequential(
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, drug_x: torch.Tensor, enzyme_x: torch.Tensor, enzyme_ids: torch.Tensor) -> torch.Tensor:
        drug_feat = self.drug_tower(drug_x)
        h = self.enzyme_embed(enzyme_x).transpose(1, 2)
        h = self.pool1(F.relu(self.conv1(h)))
        h = self.gap(h).squeeze(-1)
        ec = self.ec_embed(enzyme_ids.long())
        enzyme_feat = self.fusion(torch.cat([h, ec], dim=1))
        return self.predictor(torch.cat([drug_feat, enzyme_feat], dim=1)).squeeze(-1)


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
        return self.classifier(torch.cat([drug_feat, enz_feat], dim=1)).squeeze(-1)


class DrugTowerV3(nn.Module):
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
        self.attention = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.mlp(x)
        attn = self.attention(h)
        return h * attn


class EnzymeTowerV3(nn.Module):
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
        return self.fusion(torch.cat([h, ec], dim=1))


class TwoTowerV3(nn.Module):
    def __init__(self, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.drug_tower = DrugTowerV3(dropout=dropout)
        self.enzyme_tower = EnzymeTowerV3(dropout=dropout)
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
        return self.predictor(torch.cat([drug_feat, enzyme_feat], dim=1)).squeeze(-1)


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
        return self.fusion(torch.cat([h, ec], dim=1))


class TwoTowerCNN(nn.Module):
    def __init__(self, num_cnn_layers: int = 2) -> None:
        super().__init__()
        self.drug_tower = DrugTower()
        self.enzyme_tower = EnzymeTowerCNN(num_cnn_layers=num_cnn_layers)
        self.predictor = nn.Sequential(
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 1),
        )

    def forward(self, drug_x: torch.Tensor, enzyme_seq: torch.Tensor, enzyme_id: torch.Tensor) -> torch.Tensor:
        drug_feat = self.drug_tower(drug_x)
        enzyme_feat = self.enzyme_tower(enzyme_seq, enzyme_id)
        return self.predictor(torch.cat([drug_feat, enzyme_feat], dim=1)).squeeze(-1)


MODEL_SPECS: list[dict[str, object]] = [
    {"name": "Drug-Only", "class": DrugOnlyModel, "feature_key": "morgan2048", "mode": "drug_only"},
    {"name": "Morgan-CNN", "class": MorganCNN, "feature_key": "morgan2048", "mode": "drug_seq"},
    {"name": "Exp7-plus", "class": LightweightModel, "feature_key": "morgan2056", "mode": "drug_enzid"},
    {"name": "Two-Tower V1", "class": TwoTowerV1, "feature_key": "morgan2056", "mode": "drug_seq_enzid"},
    {"name": "Exp9", "class": TwoTowerV3, "feature_key": "morgan2056", "mode": "drug_seq_enzid"},
    {"name": "CNN-2", "class": TwoTowerCNN, "feature_key": "morgan2056", "mode": "drug_seq_enzid", "kwargs": {"num_cnn_layers": 2}},
]


def evaluate_predictions(true_labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    preds = (probs >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(true_labels, probs)),
        "pr_auc": float(average_precision_score(true_labels, probs)),
        "mcc": float(matthews_corrcoef(true_labels, preds)),
        "accuracy": float(accuracy_score(true_labels, preds)),
        "f1": float(f1_score(true_labels, preds)),
    }


def forward_by_mode(
    model: nn.Module,
    mode: str,
    drug_tensor: torch.Tensor,
    seq_tensor: torch.Tensor,
    enz_id_tensor: torch.Tensor,
) -> torch.Tensor:
    if mode == "drug_only":
        return model(drug_tensor)
    if mode == "drug_seq":
        return model(drug_tensor, seq_tensor)
    if mode == "drug_enzid":
        return model(drug_tensor, enz_id_tensor)
    return model(drug_tensor, seq_tensor, enz_id_tensor)


def train_and_evaluate(
    model_spec: dict[str, object],
    feature_bank: dict[str, np.ndarray],
    enzyme_seqs: np.ndarray,
    enzyme_ids: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    epochs: int = EPOCHS,
) -> dict[str, float]:
    set_seed(SEED)
    model_mode = str(model_spec["mode"])
    drug_features = feature_bank[str(model_spec["feature_key"])]

    train_drug = torch.FloatTensor(drug_features[train_idx]).to(DEVICE)
    val_drug = torch.FloatTensor(drug_features[val_idx]).to(DEVICE)
    test_drug = torch.FloatTensor(drug_features[test_idx]).to(DEVICE)
    train_seq = torch.LongTensor(enzyme_seqs[train_idx]).to(DEVICE)
    val_seq = torch.LongTensor(enzyme_seqs[val_idx]).to(DEVICE)
    test_seq = torch.LongTensor(enzyme_seqs[test_idx]).to(DEVICE)
    train_enz_id = torch.LongTensor(enzyme_ids[train_idx]).to(DEVICE)
    val_enz_id = torch.LongTensor(enzyme_ids[val_idx]).to(DEVICE)
    test_enz_id = torch.LongTensor(enzyme_ids[test_idx]).to(DEVICE)
    train_labels_tensor = torch.FloatTensor(labels[train_idx]).to(DEVICE)

    kwargs = dict(model_spec.get("kwargs", {}))
    model = model_spec["class"](**kwargs).to(DEVICE)

    train_loader = []
    for i in range(0, len(train_idx), BATCH_SIZE):
        sl = slice(i, i + BATCH_SIZE)
        train_loader.append(
            (
                train_drug[sl],
                train_seq[sl],
                train_enz_id[sl],
                train_labels_tensor[sl],
            )
        )

    train_labels = labels[train_idx]
    pos_weight = (train_labels == 0).sum() / max((train_labels == 1).sum(), 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(pos_weight)], device=DEVICE))
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_auc = 0.0
    best_state: dict[str, torch.Tensor] | None = None

    for _ in range(epochs):
        model.train()
        random.shuffle(train_loader)
        for drug_batch, seq_batch, enz_id_batch, label_batch in train_loader:
            optimizer.zero_grad()
            outputs = forward_by_mode(model, model_mode, drug_batch, seq_batch, enz_id_batch)
            loss = criterion(outputs, label_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = forward_by_mode(model, model_mode, val_drug, val_seq, val_enz_id)
            val_probs = torch.sigmoid(val_outputs).cpu().numpy()
        val_auc = roc_auc_score(labels[val_idx], val_probs)
        if val_auc > best_val_auc:
            best_val_auc = float(val_auc)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_outputs = forward_by_mode(model, model_mode, test_drug, test_seq, test_enz_id)
        test_probs = torch.sigmoid(test_outputs).cpu().numpy()

    metrics = evaluate_predictions(labels[test_idx], test_probs)
    metrics["val_auc"] = float(best_val_auc)
    return metrics


def load_feature_bank(df: pd.DataFrame) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    df = df.copy()
    df["enzyme_id"] = df["enzyme_name"].map(ENZYME_MAP)
    labels = df["label"].values.astype(np.float32)
    morgan_desc = np.array([mol_to_morgan_fp(s) for s in df["drug_smiles"]], dtype=np.float32)
    enzyme_seqs = np.array([seq_to_indices(seq) for seq in df["enzyme_seq"]], dtype=np.int64)
    enzyme_ids = df["enzyme_id"].values.astype(np.int64)
    feature_bank = {
        "morgan2048": morgan_desc[:, :2048],
        "morgan2056": morgan_desc,
    }
    return feature_bank, enzyme_seqs, enzyme_ids, labels


def plot_results(results: dict[str, dict[str, dict[str, float]]], ratios: list[tuple[float, float, str]]) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = {
        "Drug-Only": "#7f8c8d",
        "Morgan-CNN": "#e74c3c",
        "Exp7-plus": "#2ecc71",
        "Two-Tower V1": "#3498db",
        "Exp9": "#9b59b6",
        "CNN-2": "#f39c12",
    }
    markers = {
        "Drug-Only": "^",
        "Morgan-CNN": "o",
        "Exp7-plus": "D",
        "Two-Tower V1": "s",
        "Exp9": "P",
        "CNN-2": "X",
    }
    ratio_labels = [r[2] for r in ratios]
    x_pos = np.arange(len(ratio_labels))
    for model_name, model_results in results.items():
        aucs = [model_results[r]["auc"] for r in ratio_labels]
        ax.plot(
            x_pos,
            aucs,
            marker=markers[model_name],
            label=model_name,
            color=colors[model_name],
            linewidth=2,
            markersize=8,
        )
    ax.set_xlabel("Scaffold Split Ratio (train/val/test)", fontsize=12)
    ax.set_ylabel("Test AUROC", fontsize=12)
    ax.set_title("CMDM Scaffold Robustness Across Representative Models", fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ratio_labels)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    all_aucs = [results[m][r]["auc"] for m in results for r in ratio_labels]
    ax.set_ylim([max(0.20, min(all_aucs) - 0.05), min(0.95, max(all_aucs) + 0.03)])
    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=300)


def main() -> None:
    set_seed(SEED)
    print("=" * 70)
    print("CMDM EXPERIMENT 12: Scaffold Robustness")
    print("=" * 70)
    print("Representative models: Drug-Only | Morgan-CNN | Exp7-plus | Two-Tower V1 | Exp9 | CNN-2")
    print("Ratios: 80/10/10 | 70/15/15 | 60/20/20 | 50/25/25")
    print(f"Device: {DEVICE}")

    if DEBUG_PROGRESS_PATH.exists():
        DEBUG_PROGRESS_PATH.unlink()
    if PARTIAL_RESULT_PATH.exists():
        PARTIAL_RESULT_PATH.unlink()
    append_debug_progress({"event": "run_started", "data_path": str(DATA_PATH)})

    df = pd.read_csv(DATA_PATH)
    feature_bank, enzyme_seqs, enzyme_ids, labels = load_feature_bank(df)
    ratios = [
        (0.80, 0.10, "80/10/10"),
        (0.70, 0.15, "70/15/15"),
        (0.60, 0.20, "60/20/20"),
        (0.50, 0.25, "50/25/25"),
    ]

    results: dict[str, dict[str, dict[str, float]]] = {}
    for model_spec in MODEL_SPECS:
        model_name = str(model_spec["name"])
        print(f"\n{'=' * 60}")
        print(f"Model: {model_name}")
        print(f"{'=' * 60}")
        append_debug_progress({"event": "model_started", "model": model_name})
        results[model_name] = {}

        for train_r, val_r, ratio_str in ratios:
            train_idx, val_idx, test_idx = scaffold_split_ratio(df, train_r, val_r, SEED)
            print(f"  Ratio {ratio_str}: Train {len(train_idx)} | Val {len(val_idx)} | Test {len(test_idx)}")
            append_debug_progress(
                {
                    "event": "ratio_started",
                    "model": model_name,
                    "ratio": ratio_str,
                    "train": int(len(train_idx)),
                    "val": int(len(val_idx)),
                    "test": int(len(test_idx)),
                }
            )
            res = train_and_evaluate(
                model_spec,
                feature_bank,
                enzyme_seqs,
                enzyme_ids,
                labels,
                train_idx,
                val_idx,
                test_idx,
                epochs=EPOCHS,
            )
            results[model_name][ratio_str] = res
            print(
                f"    AUROC {res['auc']:.4f} | AUPRC {res['pr_auc']:.4f} | "
                f"MCC {res['mcc']:.4f} | F1 {res['f1']:.4f}"
            )
            with open(PARTIAL_RESULT_PATH, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            append_debug_progress(
                {
                    "event": "ratio_finished",
                    "model": model_name,
                    "ratio": ratio_str,
                    "auc": float(res["auc"]),
                    "pr_auc": float(res["pr_auc"]),
                    "mcc": float(res["mcc"]),
                    "f1": float(res["f1"]),
                }
            )
        mean_auc = float(np.mean([results[model_name][r[2]]["auc"] for r in ratios]))
        results[model_name]["mean_auc"] = {"value": mean_auc}
        append_debug_progress({"event": "model_finished", "model": model_name, "mean_auc": mean_auc})

    payload = {
        "experiment": "CMDM Experiment 12: Scaffold Robustness",
        "data_path": str(DATA_PATH),
        "representative_models": [str(spec["name"]) for spec in MODEL_SPECS],
        "ratios": [r[2] for r in ratios],
        "results": results,
    }
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    append_debug_progress({"event": "final_results_saved", "result_path": str(RESULT_PATH)})

    plot_results(results, ratios)
    append_debug_progress({"event": "figure_saved", "figure_path": str(FIGURE_PATH)})
    append_debug_progress({"event": "run_finished"})

    print("\n" + "=" * 70)
    print("SUMMARY: CMDM Scaffold Robustness")
    print("=" * 70)
    print(f"\n{'Model':<16} {'80/10/10':<10} {'70/15/15':<10} {'60/20/20':<10} {'50/25/25':<10} {'Mean':<10}")
    print("-" * 82)
    for model_name, model_results in results.items():
        row = f"{model_name:<16}"
        for _, _, ratio_str in ratios:
            row += f" {model_results[ratio_str]['auc']:<10.4f}"
        row += f" {model_results['mean_auc']['value']:<10.4f}"
        print(row)
    print("=" * 82)
    print(f"Result saved to: {RESULT_PATH}")
    print(f"Figure saved to: {FIGURE_PATH}")


if __name__ == "__main__":
    main()

