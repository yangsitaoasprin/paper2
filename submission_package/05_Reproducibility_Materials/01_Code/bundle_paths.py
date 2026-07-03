from __future__ import annotations

from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
BUNDLE_ROOT = CODE_DIR.parent
PACKAGE_ROOT = BUNDLE_ROOT.parent

DATA_DIR = BUNDLE_ROOT / "02_Data"
RESULTS_DIR = BUNDLE_ROOT / "03_Results"
FIGURE_DATA_DIR = BUNDLE_ROOT / "04_Figure_Data"
MODEL_DIR = RESULTS_DIR / "checkpoints"
DIAGNOSTIC_FIGURE_DIR = RESULTS_DIR / "diagnostic_figures"
MANUSCRIPT_FIGURE_DIR = PACKAGE_ROOT / "03_Figures"

