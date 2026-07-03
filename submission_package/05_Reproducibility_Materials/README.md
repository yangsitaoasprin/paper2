# Reproducibility Materials

This folder collects the active code, data, figure inputs, and result artifacts that support the rebuilt English CMDM manuscript.

For the quickest reviewer-facing startup path, see `RUN_ME_FIRST.md`.

## Purpose

- Provide a reviewer-facing reproducibility bundle rather than only the manuscript PDFs and figure exports.
- Keep the submission line limited to the active CMDM benchmark workflow.
- Exclude legacy scripts and out-of-scope experiment branches that are not used in the current paper narrative.

## Folder Structure

- `01_Code/`
  - Active data-preparation scripts.
  - Active main-line experiment scripts for `Exp1`, `Exp2`, `Exp6`, `Exp7-plus`, `Exp8-11`, `Exp12`, and `Exp13`.
  - Active manuscript figure-generation script: `plot_manuscript_figures.py`.
- `02_Data/`
  - `cmdm_lab_cyp450_baseline_ready.csv`: analysis-ready benchmark table used by the manuscript experiments.
  - `cmdm_lab_cyp450_root_canonical.csv`: canonical intermediate table used to build the analysis-ready benchmark.
  - `cyp450_real.csv`: enzyme-sequence source used in the active preprocessing chain.
  - `external_cmdm_root_csv/`: released per-isoform source CSV files retained for provenance tracing.
- `03_Results/`
  - Active `cmdm_*` result files used in the manuscript evidence chain.
  - Includes repeated-run released-split outputs and the scaffold-robustness result artifact.
- `04_Figure_Data/`
  - CSV tables used to generate manuscript Figures `2` and `3`.

## Runtime Note

- The bundle is arranged so reviewers can run scripts directly from `01_Code/` without path errors caused by missing local data folders.
- The numbered folders are now the only active runtime layout used by the scripts:
  - `02_Data/` for benchmark inputs and preprocessing outputs
  - `03_Results/` for result artifacts and script-generated checkpoints
  - `04_Figure_Data/` for figure CSV inputs
- `plot_manuscript_figures.py` writes regenerated manuscript figures to `submission_package/03_Figures/`.
- Full model-script reruns also require the Python dependencies listed in `requirements.txt`, especially `torch`, `rdkit`, `numpy`, `pandas`, `matplotlib`, and `scikit-learn`.
- For conda-based recreation, an `environment.yml` file is also provided for the locally verified `cyp450` environment.
- For figure generation on Windows, use the pinned bundle environment as provided. The locally verified stable render stack is `numpy 2.3.5`, `matplotlib 3.10.6`, `pillow 12.0.0`, `kiwisolver 1.4.9`, and `fonttools 4.60.1`.
- In the current local verification, the working conda environment is `cyp450`. If `conda run -n <env>` behaves unexpectedly on a given machine, activate the environment first or call that environment's `python.exe` directly.
- A typical local check is:
  - `cd 01_Code`
  - `conda activate cyp450`
  - `python plot_manuscript_figures.py --figures 1 2 3 S1`
  - `python build_cmdm_lab_cyp450_root_canonical.py`
  - `python prepare_cmdm_lab_baseline_ready.py`
- If activation is inconvenient, the same commands can be run with the interpreter path directly, for example:
  - `C:\Users\Administrator\.conda\envs\cyp450\python.exe plot_manuscript_figures.py --figures S1`
  - `C:\Users\Administrator\.conda\envs\cyp450\python.exe build_cmdm_lab_cyp450_root_canonical.py`
  - `C:\Users\Administrator\.conda\envs\cyp450\python.exe prepare_cmdm_lab_baseline_ready.py`

## Verified Environment

- The submission bundle has been locally verified in the conda environment `cyp450`.
- Verified interpreter:
  - `Python 3.13.14`
  - `Windows-11-10.0.26200-SP0`
- Verified core package versions:
  - `numpy 2.3.5`
  - `pandas 3.0.3`
  - `matplotlib 3.10.6`
  - `pillow 12.0.0`
  - `kiwisolver 1.4.9`
  - `fonttools 4.60.1`
  - `scikit-learn 1.9.0`
  - `rdkit 2026.03.3`
  - `torch 2.11.0+cu128`
- Verified CUDA-related runtime state:
  - `torch.version.cuda = 12.8`
  - `cuDNN = 91900`
  - `torch.cuda.is_available() = True`
  - `torch.cuda.device_count() = 1`
  - `GPU = NVIDIA GeForce RTX 5090 Laptop GPU`
  - `compute capability = (12, 0)`
- Practical interpretation:
  - preprocessing and figure-generation scripts can run on CPU
  - full model reruns are best attempted in a GPU-enabled environment close to the verified stack above
- Render stability note:
  - an older local `cyp450` environment snapshot using newer graph-stack variants on `Windows 11 + Python 3.13.14` crashed during `savefig()` and `FigureCanvasAgg.draw()`
  - the submission bundle therefore pins the full locally verified render stack above, and the plotting script still fails fast if it detects the known-crashing `matplotlib 3.11.0` Windows renderer stack
- Version scope note:
  - the current bundle is explicitly verified on `Python 3.13.14`
  - if an exact match is important, use the `cyp450` environment or recreate the package versions pinned in `requirements.txt`
- for conda users, prefer `conda env create -f environment.yml`

## Recommended Reading Order

1. Inspect `02_Data/` to confirm the benchmark provenance chain.
2. Inspect `01_Code/prepare_cmdm_lab_baseline_ready.py` and `01_Code/build_cmdm_lab_cyp450_root_canonical.py` for preprocessing.
3. Inspect the active experiment scripts in `01_Code/`.
4. Inspect `03_Results/` and `04_Figure_Data/`.
5. Regenerate figures with `01_Code/plot_manuscript_figures.py` if needed.

## Scope Note

This folder is intentionally restricted to the current manuscript line. Older exploratory scripts, historical benchmark branches, and archived legacy figures remain in the repository but are not part of this reviewer-facing bundle.
