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
|   |-- baselines/                  # Public PINN-LSTM and GRU/TCN baseline scripts
|   |-- revision_studies/           # Probe-density and pseudo-label ablation scripts
|   `-- comsol_followup/            # COMSOL long-window export/training scripts
|-- data/
|   |-- single_cylinder_google_drive.txt
|   |-- three_cylinder_google_drive.txt
|   `-- revision_added_experiment_data_google_drive.txt
|-- results/
|   |-- main_benchmark/             # Main benchmark metrics and histories
|   `-- revision_studies/           # Baselines, ablations, diagnostics, COMSOL follow-up
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

## Key Result Folders

- `results/main_benchmark/`: summary metrics for the reported single-cylinder and three-cylinder benchmark.
- `results/revision_studies/baselines/`: PINN-LSTM, GRU, and TCN comparison artifacts.
- `results/revision_studies/probe_density/`: 500/1000/2000/4000 probe-density sensitivity results.
- `results/revision_studies/pseudo_label_ablation/`: no-pseudo-label ablation results.
- `results/revision_studies/physical_diagnostics/`: residual, temporal, and spectral diagnostic figures and CSV files.
- `results/revision_studies/comsol_long_window/`: independent COMSOL long-window diagnostics and formal MI-PINN/PINN runs.

## Installation

```bash
pip install -r requirements.txt
```

The training scripts require a PyTorch environment. Some COMSOL follow-up scripts require COMSOL 6.3 with the Python/MPh workflow used during revision.

## Data Boundary

This is a lightweight GitHub package. Large raw artifacts are intentionally excluded, including COMSOL `.mph` files, full rendered figure archives, very large baseline prediction arrays, and bulky time-evolution frame sequences.

The file manifest is recorded in `manifests/artifact_manifest.csv`. See `docs/data_boundary.md` for details.

## Revision Evidence

The evidence used to support the revision response is indexed in `docs/evidence_map.md`. The evidence is organized by result type rather than by internal working package, so the repository remains readable as a public project.

