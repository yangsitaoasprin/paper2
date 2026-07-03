# RUN ME FIRST

This note is a one-page quick-start guide for reviewers who want to execute the reproducibility bundle directly from `05_Reproducibility_Materials/`.

## 1. Recommended Environment

- Recommended conda environment name: `cyp450`
- Locally verified Python version: `3.13.14`
- Locally verified operating system: `Windows 11`
- Locally verified deep-learning stack:
  - `torch 2.11.0+cu128`
  - `torch.version.cuda = 12.8`
  - `cuDNN = 91900`
- Recommended figure-rendering pin:
  - `numpy 2.3.5`
  - `matplotlib 3.10.6`
  - `pillow 12.0.0`
  - `kiwisolver 1.4.9`
  - `fonttools 4.60.1`
- Locally verified GPU:
  - `NVIDIA GeForce RTX 5090 Laptop GPU`
  - `compute capability = (12, 0)`

## 2. Recommended Setup

### Option A. Conda recreation

```powershell
cd 05_Reproducibility_Materials
conda env create -f environment.yml
conda activate cyp450
```

### Option B. Pip-style installation

```powershell
cd 05_Reproducibility_Materials
pip install -r requirements.txt
```

Note:
- `requirements.txt` pins the core Python package versions.
- The locally verified PyTorch build was `2.11.0+cu128`.
- For Windows figure generation, keep the pinned render stack above. In local verification, newer graph-stack variants in the older `cyp450` snapshot caused native crashes during `savefig()`.
- If a machine requires a CUDA-specific PyTorch wheel, install the matching wheel for that machine and CUDA stack first, then install the remaining packages.

## 3. Where To Run

Run commands from:

```powershell
cd 05_Reproducibility_Materials\01_Code
```

The bundle uses a single numbered runtime layout:

- `02_Data/` for benchmark inputs and preprocessing outputs
- `03_Results/` for result artifacts and script-generated checkpoints
- `04_Figure_Data/` for figure CSV inputs
- `submission_package/03_Figures/` for regenerated manuscript figure files

## 4. Minimum Sanity Checks

These are the fastest recommended checks.

### Rebuild canonical table

```powershell
python build_cmdm_lab_cyp450_root_canonical.py
```

### Rebuild analysis-ready benchmark table

```powershell
python prepare_cmdm_lab_baseline_ready.py
```

### Regenerate manuscript figures

```powershell
python plot_manuscript_figures.py --figures 1 2 3 S1
```

## 5. Direct Interpreter Alternative

If `conda run` or shell activation behaves unexpectedly on a given machine, use the environment interpreter path directly. In the locally verified setup, that was:

```powershell
C:\Users\Administrator\.conda\envs\cyp450\python.exe plot_manuscript_figures.py --figures S1
C:\Users\Administrator\.conda\envs\cyp450\python.exe build_cmdm_lab_cyp450_root_canonical.py
C:\Users\Administrator\.conda\envs\cyp450\python.exe prepare_cmdm_lab_baseline_ready.py
```

## 6. Practical Interpretation

- Preprocessing and figure-generation scripts can run without GPU.
- Full model reruns are best attempted in a GPU-enabled environment close to the verified stack above.
- The active reviewer-facing package is organized in numbered folders:
  - `01_Code`
  - `02_Data`
  - `03_Results`
  - `04_Figure_Data`

For fuller environment notes and version details, see `README.md`, `requirements.txt`, and `environment.yml`.
