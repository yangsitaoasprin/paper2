"""
Run a compact multi-seed released-split benchmark for the core CMDM models.

The goal is to strengthen the manuscript's main claims with repeated runs on
the most decision-relevant models rather than re-running the full benchmark
branch indiscriminately.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
from bundle_paths import BUNDLE_ROOT, MODEL_DIR, RESULTS_DIR

PROJECT_ROOT = BUNDLE_ROOT
RESULT_DIR = RESULTS_DIR
RESULT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SEEDS = [42, 43, 44]
METRIC_KEYS = ["auc", "pr_auc", "mcc", "accuracy", "f1"]

MODEL_SPECS = [
    {
        "name": "Drug-Only",
        "script": SCRIPT_DIR / "experiment6_CMDM_Drug-Only_baseline.py",
        "result_name": "cmdm_experiment6_drugonly",
        "model_name": "cmdm_experiment6_drugonly",
    },
    {
        "name": "Morgan-CNN",
        "script": SCRIPT_DIR / "experiment1_CMDM_Morgan-CNN_baseline.py",
        "result_name": "cmdm_experiment1_morgancnn",
        "model_name": "cmdm_experiment1_morgancnn",
    },
    {
        "name": "Two-Tower V1",
        "script": SCRIPT_DIR / "experiment2_CMDM_Two-Tower-V1_baseline.py",
        "result_name": "cmdm_experiment2_twotowerv1",
        "model_name": "cmdm_experiment2_twotowerv1",
    },
    {
        "name": "Exp7-plus",
        "script": SCRIPT_DIR / "experiment7plus_CMDM_Lightweight_baseline.py",
        "result_name": "cmdm_experiment7plus_lightweight",
        "model_name": "cmdm_experiment7plus_lightweight",
    },
    {
        "name": "CNN-2",
        "script": SCRIPT_DIR / "experiment13_CMDM_CNN_Ablation_baseline.py",
        "result_name": "cmdm_experiment13_cnn2",
        "model_name": "cmdm_experiment13_cnn2",
        "extra_env": {
            "CMDM_CNN_LAYERS": "2",
            "CMDM_SUMMARY_PATH": str(RESULT_DIR / "cmdm_experiment13_cnn2_multiseed_summary_placeholder.json"),
        },
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run compact multi-seed CMDM released-split baselines.")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS, help="Seeds to run, e.g. --seeds 42 43 44")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[spec["name"] for spec in MODEL_SPECS],
        help="Subset of model names to run",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULT_DIR / "cmdm_multiseed_released_split_summary.json",
        help="Output JSON summary path",
    )
    return parser.parse_args()


def build_run_paths(result_name: str, model_name: str, seed: int) -> tuple[Path, Path]:
    result_path = RESULT_DIR / f"{result_name}_seed{seed}.json"
    model_path = MODEL_DIR / f"{model_name}_seed{seed}.pt"
    return result_path, model_path


def run_single_seed(spec: dict[str, object], seed: int) -> dict[str, object]:
    result_path, model_path = build_run_paths(str(spec["result_name"]), str(spec["model_name"]), seed)

    if result_path.exists():
        with result_path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        return {
            "seed": seed,
            "result_path": str(result_path),
            "model_path": str(model_path),
            "best_epoch": result.get("best_epoch"),
            "best_val_auc": result.get("best_val_auc"),
            "test_metrics": result["test_metrics"],
        }

    env = os.environ.copy()
    env["CMDM_SEED"] = str(seed)
    env["CMDM_RESULT_PATH"] = str(result_path)
    env["CMDM_MODEL_PATH"] = str(model_path)
    for key, value in dict(spec.get("extra_env", {})).items():
        env[key] = str(value).replace("seed_placeholder", str(seed))

    if spec["name"] == "CNN-2":
        env["CMDM_SUMMARY_PATH"] = str(RESULT_DIR / f"cmdm_experiment13_cnn2_seed{seed}_summary.json")
        env["CMDM_RESULT_TEMPLATE"] = f"{spec['result_name']}_seed{seed}.json"
        env["CMDM_MODEL_TEMPLATE"] = f"{spec['model_name']}_seed{seed}.pt"

    print(f"\nRunning {spec['name']} with seed {seed}")
    subprocess.run([sys.executable, str(spec["script"])], cwd=PROJECT_ROOT, env=env, check=True)

    with result_path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    return {
        "seed": seed,
        "result_path": str(result_path),
        "model_path": str(model_path),
        "best_epoch": result.get("best_epoch"),
        "best_val_auc": result.get("best_val_auc"),
        "test_metrics": result["test_metrics"],
    }


def summarize_runs(runs: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {
        "n_runs": len(runs),
        "seeds": [int(run["seed"]) for run in runs],
        "metrics": {},
    }
    for key in METRIC_KEYS:
        values = [float(run["test_metrics"][key]) for run in runs]
        summary["metrics"][key] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "values": values,
        }
    return summary


def main() -> None:
    args = parse_args()
    selected_specs = [spec for spec in MODEL_SPECS if spec["name"] in set(args.models)]
    if not selected_specs:
        raise SystemExit("No valid models selected.")

    payload: dict[str, object] = {
        "experiment": "CMDM released-split compact multi-seed benchmark",
        "python_executable": sys.executable,
        "seeds": args.seeds,
        "models": {},
    }

    for spec in selected_specs:
        runs = [run_single_seed(spec, seed) for seed in args.seeds]
        payload["models"][spec["name"]] = {
            "runs": runs,
            "summary": summarize_runs(runs),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"\nSaved summary to: {args.output}")


if __name__ == "__main__":
    main()

