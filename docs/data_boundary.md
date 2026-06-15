# Data and Artifact Boundary

This repository is organized as a lightweight public package for manuscript revision traceability.

## Included

- Source scripts for the MI-PINN/PINN workflow, baselines, revision studies, and COMSOL follow-up.
- Google Drive link files for the single-cylinder and three-cylinder raw benchmark data.
- A Google Drive link placeholder and checksum record for the additional revision-experiment data package.
- Compact benchmark outputs and summary metrics.
- Selected model checkpoints and compact prediction files where practical.
- Figures, CSV summaries, logs, and scripts used in the revision evidence.
- Supplementary COMSOL long-window PDF and LaTeX source.
- Manifest files documenting included and excluded server-side artifacts.

## Excluded

The following artifacts are intentionally excluded to avoid making the repository a raw-data dump:

- COMSOL `.mph` model files.
- Full raw COMSOL model files and bulky rendered frame sequences.
- Very large raw prediction arrays and redundant rendered figure archives.
- Bulky time-evolution frame sequences.
- Cache files, backup files, and internal working notes.

The raw public benchmark data should be downloaded through the two benchmark text files under `data/`. The additional revision-experiment data package is documented in `data/revision_added_experiment_data_google_drive.txt`.

## Interpretation Boundary

The main public benchmark remains the reported 0--5 s observation and 5--7 s extrapolation task.

The COMSOL long-window follow-up is independent evidence for the revision response. It supports the limited conclusion that the Mamba temporal-prior consistency term can reduce error growth relative to a PINN baseline under the independent COMSOL setting. It does not prove arbitrary long-term forecasting and should not be merged with the public benchmark metrics.

## Reproducibility Note

Some scripts depend on local data paths or server-side raw arrays that are not included in the lightweight package. The compact outputs, manifests, and external data-package record are provided so the reported revision evidence can be traced without turning the repository itself into a raw-data dump.

