# Submission Package

This directory is the slim reviewer-facing submission bundle. It keeps only the manuscript, supplementary file, figure assets, cover-letter draft, and active reproducibility materials needed to inspect or rerun the current CMDM manuscript line.

## Submit These Files

- `01_Manuscript/Task-Aligned-Complexity-in-Small-Sample-Multi-Enzyme-CYP450-Prediction  A- Benchmark-Guided-Analysis.docx`
- `02_Supplementary/paper_REBUILT_MAINLINE_SI.md`
- `03_Figures/figure1_mainline_experiment_map.png`
- `03_Figures/figure1_mainline_experiment_map.pdf`
- `03_Figures/figure2_enzyme_cnn_ablation.png`
- `03_Figures/figure2_enzyme_cnn_ablation.pdf`
- `03_Figures/figure3_scaffold_robustness_profile.png`
- `03_Figures/figure3_scaffold_robustness_profile.pdf`
- `03_Figures/figureS1_representative_model_routes.png`
- `03_Figures/figureS1_representative_model_routes.pdf`
- `04_Cover_Letter/paper_EN_COVER_LETTER_DRAFT.md`
- `05_Reproducibility_Materials/` as a required reviewer-facing reproducibility bundle; at minimum, the active script files in `01_Code/` must be uploaded even if the journal portal requires the bundle to be zipped or deposited externally

## Figure Cross-Check

- `Figure 1` in the manuscript is the main-line experiment map and corresponds to `03_Figures/figure1_mainline_experiment_map.png` and `03_Figures/figure1_mainline_experiment_map.pdf`.
- `Figure 2` in the manuscript is the controlled `Exp13` enzyme-side CNN depth ablation and corresponds to `03_Figures/figure2_enzyme_cnn_ablation.png` and `03_Figures/figure2_enzyme_cnn_ablation.pdf`.
- `Figure 3` in the manuscript is the `Exp12` scaffold robustness profile across four scaffold ratios and corresponds to `03_Figures/figure3_scaffold_robustness_profile.png` and `03_Figures/figure3_scaffold_robustness_profile.pdf`.
- `Figure S1` in the Supplementary Information is the reviewer-oriented representative model-route schematic and corresponds to `03_Figures/figureS1_representative_model_routes.png` and `03_Figures/figureS1_representative_model_routes.pdf`.
- The active main-text manuscript line uses exactly Figures `1-3`, while `Figure S1` belongs to the Supplementary Information; older out-of-scope figure assets and legacy scaffold result files should not be submitted.

## Reproducibility Materials

- `05_Reproducibility_Materials/01_Code/`: active preprocessing, experiment, and figure-generation scripts for the current CMDM manuscript line
- `05_Reproducibility_Materials/02_Data/`: analysis-ready benchmark table, upstream canonical table, enzyme-sequence source, and released root CSV files
- `05_Reproducibility_Materials/03_Results/`: active `cmdm_*` result artifacts supporting the manuscript claims
- `05_Reproducibility_Materials/04_Figure_Data/`: CSV inputs for manuscript Figures `2` and `3`
- `05_Reproducibility_Materials/README.md`: reviewer-facing guide to the reproducibility bundle
- The script files in `05_Reproducibility_Materials/01_Code/` are required submission assets for this package, not optional internal notes.
- Reviewers should treat `05_Reproducibility_Materials/01_Code/` as the primary code entry point referenced by the Supplementary Information.

## Author Finalize Before Submission

- Replace the provisional `Data and Code Availability` statement.
- Confirm the final journal reference style.
- Confirm author-contribution wording and corresponding-author details.
- Confirm whether the target journal requires a TOC graphic, highlights, or other additional files.
- If the journal portal does not accept raw code/data folders directly, zip `05_Reproducibility_Materials` or deposit the same contents in the final repository/archive referenced by the manuscript, but do not omit the active script files.
