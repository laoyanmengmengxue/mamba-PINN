# Evidence Map

This file maps the main revision evidence to the folders in this repository.

## Main Benchmark

Location: `results/main_benchmark/`

Includes compact metric summaries, training histories, and evaluation scripts for the reported single-cylinder and three-cylinder benchmark.

## Temporal Baselines

Location: `results/revision_studies/baselines/`

- `pinn_lstm/`: public PINN-LSTM baseline inference artifacts, summary metrics, logs, and figures.
- `gru_tcn/`: GRU and TCN temporal baseline artifacts, compact prediction files, model checkpoints, figures, and summary tables.

The PINN-LSTM comparison is kept as a public-method baseline check. It is not presented as a same-protocol end-to-end retraining of the proposed MI-PINN.

## Probe-Density Sensitivity

Location: `results/revision_studies/probe_density/`

Includes the 500/1000/2000/4000 virtual-probe sensitivity summary and selected run outputs for single-cylinder and three-cylinder cases.

The 2000-probe setting is treated as a practical accuracy-cost choice under the tested benchmark, not a universal optimum.

## Pseudo-Label Consistency Ablation

Location: `results/revision_studies/pseudo_label_ablation/`

Includes strict no-pseudo-label ablation outputs for single-cylinder and three-cylinder cases.

The ablation is used to evaluate the contribution of the Mamba-based temporal consistency term. It should be interpreted together with the main benchmark results rather than as a replacement for the full method.

## Residual, Temporal, and Spectral Diagnostics

Location: `results/revision_studies/physical_diagnostics/`

Includes continuity and momentum residual diagnostics, probe time-series diagnostics, and spectral summaries. These files support residual-level physical-consistency claims.

For the three-cylinder case, the spectral metrics should be read cautiously because the multi-body wake is less strictly periodic than the single-cylinder wake.

## Computational Cost

Location: `results/revision_studies/computational_cost/`

Includes the end-to-end training-cost figure and plotting script used to compare PINN-only, MI-PINN, and PINN+LSTM cost levels.

## Independent COMSOL Long-Window Follow-Up

Locations:

- `supplementary/comsol_long_window/`
- `results/revision_studies/comsol_long_window/`
- `src/comsol_followup/`

These files document the independent COMSOL follow-up experiment used to address the editor's longer-window concern. This experiment is separate from the cited public benchmark and should not be described as a strict extension of that public dataset.

## Artifact Manifest

Location: `manifests/`

The manifest records which server-side artifacts were included and which large artifacts were intentionally excluded from the lightweight GitHub package.

