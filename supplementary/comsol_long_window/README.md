# Supplementary COMSOL Long-Window Follow-Up

This folder contains the lightweight materials for the independent COMSOL long-window follow-up experiment added in response to the editor's concern about the extrapolation horizon.

## Scope

The experiment is an independent COMSOL follow-up, not a strict extension of the public Dou et al. benchmark. The public-benchmark results in the manuscript remain defined by the original 0--5 s observation window and 5--7 s extrapolation window. The COMSOL follow-up is reported separately in the response letter and supplementary material.

## Included Files

- `supplementary_comsol_long_window.pdf`: supplementary note describing the protocol, results, interpretation, and traceability.
- `supplementary_comsol_long_window.tex`: LaTeX source for the supplementary note.
- `editor2_formal_training_summary.csv`: summary of the formal PINN and MI-PINN results over 65--67 s, 65--70 s, and 65--75 s.
- `editor2_formal_long_window_summary.png`: response/supplementary figure summarizing speed-field R2 and RMSE.
- `src/editor2_export_comsol_training_fields.py`: COMSOL-field export and regridding script used to build the training arrays.
- `src/editor2_train_mipinn_pinn.py`: formal PINN/MI-PINN training script for the independent COMSOL follow-up.
- `src/editor2_run_formal_all.cmd`: batch entry for the four formal runs.
- `src/plot_editor2_formal_summary.py`: plotting script for the summary figure.

## Training Protocol

- Cases: single-cylinder wake and three-cylinder wake at Re=100.
- Observation window: 60--65 s.
- Evaluation windows: 65--67 s, 65--70 s, and 65--75 s.
- Time step: dt = 0.05 s.
- Spatial regions: single cylinder [1,9] x [-2,2], three cylinders [6,14] x [-2,2].
- Grid: dx = dy = 0.05, giving 161 x 81 = 13041 points.
- Sparse probes: 2000 fixed Eulerian virtual probes.
- Observation mini-batch: 4000 samples per iteration.
- Physics collocation mini-batch: 20000 points per iteration.
- Mamba consistency mini-batch: 1140 extrapolation-window samples.
- PINN optimization: 20000 Adam iterations.
- Mamba optimization in MI-PINN: 3000 Adam iterations.

## Results Summary

MI-PINN reduces RMSE relative to PINN in all three evaluation windows:

- Single-cylinder case: 67.14%, 84.42%, and 87.78% reduction over 65--67 s, 65--70 s, and 65--75 s.
- Three-cylinder case: 34.05%, 70.25%, and 73.49% reduction over 65--67 s, 65--70 s, and 65--75 s.

These results support a limited conclusion: under this independent COMSOL setting, the Mamba temporal-prior consistency term slows error growth relative to a standard PINN baseline. They do not prove arbitrary long-term forecasting and should not be merged with the public-benchmark metrics.

## Excluded Large Artifacts

The COMSOL `.mph` files and exported raw `.npz` arrays are not included here to keep the GitHub package lightweight. Selected compact formal-training summaries are available under `results/revision_studies/comsol_long_window/`. The run suffixes are listed in the supplementary PDF.
