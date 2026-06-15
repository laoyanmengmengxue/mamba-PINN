# Evidence Map

This file separates evidence stored directly in this lightweight GitHub repository from larger machine-readable artifacts stored in the linked revision data package.

## Directly Included in This Repository

### Source Code

- `src/mipinn/`: MI-PINN/PINN implementation and configuration files used for the revised manuscript workflow.
- `src/baselines/`: public PINN-LSTM and GRU/TCN baseline scripts retained for method-level traceability.
- `supplementary/comsol_long_window/src/`: scripts used for the independent COMSOL long-window follow-up, including field export, formal MI-PINN/PINN training, and plotting.

### Public Data Links

- `data/single_cylinder_google_drive.txt`: link record for the public single-cylinder benchmark data.
- `data/three_cylinder_google_drive.txt`: link record for the public three-cylinder benchmark data.
- `data/revision_added_experiment_data_google_drive.txt`: link and SHA256 record for the additional revision-experiment data package.

### Compact Revision Evidence

- `results/revision_studies/comsol_long_window/`: compact diagnostics, probe time-series data, temporal metrics, and formal MI-PINN/PINN long-window summary.
- `results/revision_studies/pseudo_label_ablation/pseudo_label_ablation_summary.csv`: compact strict no-pseudo-label ablation summary.
- `supplementary/comsol_long_window/`: supplementary PDF, LaTeX source, scripts, and figure for the independent long-window evidence.

### Manifest

- `manifests/artifact_manifest.csv`: file-level byte size and SHA256 manifest for the lightweight GitHub package.

## Stored in the Linked Revision Data Package

The larger machine-readable revision evidence is stored in the Google Drive package recorded in `data/revision_added_experiment_data_google_drive.txt`.

The package is named `revision_added_experiment_data_package_20260615.zip` and has SHA256:

```text
11D5E6E1BE41E8A8FA422A3C8E3BA77F5C05EB55561808FD4E8DD5F8382C9F20
```

It contains the larger records that are not duplicated directly in the lightweight GitHub repository, including:

- Main benchmark summaries and selected training/evaluation records.
- Temporal baseline outputs for public PINN-LSTM, GRU, and TCN checks.
- 500/1000/2000/4000 virtual-probe sensitivity records.
- Residual, temporal, and spectral diagnostic data.
- Computational-cost records.
- Selected checkpoints, logs, scripts, manifests, and compact arrays needed to trace the revision response.

## Interpretation Notes

The PINN-LSTM comparison is retained as a public-method baseline check. It is not presented as a same-protocol end-to-end retraining of the proposed MI-PINN.

The three-cylinder spectral metrics should be read cautiously because the multi-body wake is less strictly periodic than the single-cylinder wake.

The independent COMSOL long-window experiment is separate from the cited public benchmark and should not be numerically merged with the public benchmark metrics.
