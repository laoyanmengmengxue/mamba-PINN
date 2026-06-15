# MI-PINN for Sparse PIV Reconstruction and Short-Horizon Extrapolation

This repository contains the code, compact result artifacts, and supplementary evidence for the revised manuscript:

**Robust Extrapolation of Sparse Particle Image Velocimetry Measurements via a Mamba-Integrated Physics-Informed Neural Network Framework**

## Scope

The reported public benchmark evaluates sparse-flow reconstruction and short-horizon observation-free extrapolation. The main benchmark uses a 0--5 s observation window and a 5--7 s extrapolation window.

The Mamba module is used as a temporal prior for probe-wise velocity evolution. The final velocity and pressure fields are constrained through the PINN residual terms and the Navier-Stokes equations.

The independent COMSOL long-window experiment is included as supplementary revision evidence. It should not be merged numerically with the cited public benchmark because it is a separate follow-up dataset.

## Repository Layout

```text
.
|-- src/
|   |-- mipinn/                     # MI-PINN/PINN implementation and evaluation utilities
|   `-- baselines/                  # Public PINN-LSTM and GRU/TCN baseline scripts
|-- data/
|   |-- single_cylinder_google_drive.txt
|   |-- three_cylinder_google_drive.txt
|   `-- revision_added_experiment_data_google_drive.txt
|-- results/
|   `-- revision_studies/
|       |-- comsol_long_window/      # Compact independent long-window evidence
|       `-- pseudo_label_ablation/   # Compact strict ablation summary
|-- supplementary/
|   `-- comsol_long_window/         # Supplementary PDF, LaTeX source, figure, and scripts
|-- docs/                           # Evidence map and data-boundary notes
|-- manifests/                      # File manifest for this lightweight package
`-- requirements.txt
```

## Raw Data

The raw public benchmark data are not included directly in this lightweight repository. Download links are provided in:

- `data/single_cylinder_google_drive.txt`
- `data/three_cylinder_google_drive.txt`

The machine-readable data package for the additional revision experiments is tracked separately in:

- `data/revision_added_experiment_data_google_drive.txt`

## Direct Repository Evidence

- `src/mipinn/`: MI-PINN/PINN implementation and configuration files.
- `src/baselines/`: public PINN-LSTM and GRU/TCN baseline scripts.
- `results/revision_studies/comsol_long_window/`: independent COMSOL long-window diagnostics and formal MI-PINN/PINN summary outputs.
- `results/revision_studies/pseudo_label_ablation/`: compact no-pseudo-label ablation summary.
- `supplementary/comsol_long_window/`: supplementary PDF, LaTeX source, scripts, and figure for the independent long-window evidence.
- `manifests/artifact_manifest.csv`: file-level SHA256 manifest for the lightweight GitHub package.

## Revision Data Package

Additional machine-readable revision artifacts are stored in the Google Drive package linked in `data/revision_added_experiment_data_google_drive.txt`. This package contains the larger benchmark and revision-evidence records, including baseline outputs, probe-density sensitivity, residual/temporal/spectral diagnostics, computational-cost records, selected checkpoints, logs, and summary tables.

## Installation

```bash
pip install -r requirements.txt
```

The training scripts require a PyTorch environment. Some COMSOL follow-up scripts require COMSOL 6.3 with the Python/MPh workflow used during revision.

## Data Boundary

This is a lightweight GitHub package. Large raw artifacts are intentionally excluded, including COMSOL `.mph` files, full rendered figure archives, very large baseline prediction arrays, and bulky time-evolution frame sequences.

The file manifest is recorded in `manifests/artifact_manifest.csv`. See `docs/data_boundary.md` for details.

## Revision Evidence

The evidence used to support the revision response is indexed in `docs/evidence_map.md`. The evidence is organized by direct GitHub artifacts and linked revision-data artifacts, so the repository remains readable as a public project without becoming a raw-data dump.
