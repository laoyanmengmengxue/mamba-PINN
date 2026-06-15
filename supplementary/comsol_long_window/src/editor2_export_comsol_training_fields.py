from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.tri as mtri
import mph
import numpy as np


OUT = Path(os.environ.get("EAAI_COMSOL_TRAINING_ROOT", "outputs/comsol_long_window")).resolve()
DATA_DIR = OUT / "data"

CASES = {
    "single": {
        "model": Path(os.environ.get("EAAI_SINGLE_COMSOL_MPH", "models/single_cylinder_re100_shedding_probe_hr.mph")).resolve(),
        "roi": (1.0, 9.0, -2.0, 2.0),
    },
    "three": {
        "model": Path(os.environ.get("EAAI_THREE_COMSOL_MPH", "models/three_cylinder_re100_shedding_probe_hr.mph")).resolve(),
        "roi": (6.0, 14.0, -2.0, 2.0),
    },
}

T_ABS_START = 60.0
T_ABS_END = 75.0
NU = 0.01
GRID_STEP = 0.05


def nearest_window_indices(times: np.ndarray, start: float, end: float) -> np.ndarray:
    mask = (times >= start - 1e-9) & (times <= end + 1e-9)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        raise RuntimeError(f"No COMSOL times found in requested window {start}--{end}.")
    return idx


def make_roi_grid(roi: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    x0, x1, y0, y1 = roi
    xs = np.round(np.arange(x0, x1 + 0.5 * GRID_STEP, GRID_STEP), 8)
    ys = np.round(np.arange(y0, y1 + 0.5 * GRID_STEP, GRID_STEP), 8)
    xx, yy = np.meshgrid(xs, ys)
    return xx.ravel().astype(np.float32), yy.ravel().astype(np.float32), (len(ys), len(xs))


def interp_time_stack(tri: mtri.Triangulation, fields: np.ndarray, xg: np.ndarray, yg: np.ndarray) -> np.ndarray:
    out = np.empty((fields.shape[0], xg.size), dtype=np.float32)
    last = None
    for i in range(fields.shape[0]):
        interp = mtri.LinearTriInterpolator(tri, fields[i])
        vals = np.asarray(interp(xg, yg), dtype=np.float32)
        if np.ma.isMaskedArray(vals):
            vals = vals.filled(np.nan).astype(np.float32)
        if np.any(~np.isfinite(vals)):
            if last is not None:
                vals[~np.isfinite(vals)] = last[~np.isfinite(vals)]
            vals[~np.isfinite(vals)] = np.nanmean(vals)
        out[i] = vals
        last = vals
    return out


def extract_case(client, case: str, spec: dict) -> dict:
    model_path: Path = spec["model"]
    roi = tuple(float(v) for v in spec["roi"])
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    model = client.load(model_path)
    try:
        java = model.java
        all_times = np.asarray(java.sol("sol1").getPVals(), dtype=np.float64)
        idx = nearest_window_indices(all_times, T_ABS_START, T_ABS_END)
        selected_abs = all_times[idx]

        tag = f"eval_editor2_train_{case}"
        try:
            java.result().numerical().remove(tag)
        except Exception:
            pass
        num = java.result().numerical().create(tag, "Eval")
        num.set("data", "dset1")
        num.set("expr", ["u", "v", "p"])
        coords = np.asarray(num.getCoordinates(), dtype=np.float64)
        values = np.asarray(num.getData(), dtype=np.float32)

        if coords.shape[0] != 2:
            raise ValueError(f"{case}: expected 2D coordinates, got {coords.shape}")
        if values.ndim != 3 or values.shape[0] < 2 or values.shape[1] != all_times.size:
            raise ValueError(f"{case}: unexpected value tensor shape {values.shape}; time_count={all_times.size}")

        tri = mtri.Triangulation(coords[0], coords[1])
        x, y, grid_shape = make_roi_grid(roi)
        u = interp_time_stack(tri, values[0, idx, :], x, y)
        v = interp_time_stack(tri, values[1, idx, :], x, y)
        p = interp_time_stack(tri, values[2, idx, :], x, y) if values.shape[0] >= 3 else np.full_like(u, np.nan)
        t_shift = (selected_abs - T_ABS_START).astype(np.float32)

        out_file = DATA_DIR / f"editor2_{case}_roi_grid_60_75s.npz"
        np.savez_compressed(
            out_file,
            case=np.array(case),
            model_path=np.array(str(model_path)),
            roi=np.array(roi, dtype=np.float32),
            grid_step=np.array(GRID_STEP, dtype=np.float32),
            grid_shape=np.array(grid_shape, dtype=np.int32),
            x=x,
            y=y,
            t_abs=selected_abs.astype(np.float32),
            t=t_shift,
            u=u,
            v=v,
            p=p,
            nu=np.array(NU, dtype=np.float32),
            t_abs_start=np.array(T_ABS_START, dtype=np.float32),
            t_abs_end=np.array(T_ABS_END, dtype=np.float32),
        )

        dt = float(np.median(np.diff(selected_abs)))
        summary = {
            "case": case,
            "model_path": str(model_path),
            "output_file": str(out_file),
            "mesh_points": int(x.size),
            "time_count": int(selected_abs.size),
            "time_abs_start": float(selected_abs[0]),
            "time_abs_end": float(selected_abs[-1]),
            "time_shift_start": float(t_shift[0]),
            "time_shift_end": float(t_shift[-1]),
            "dt_median": dt,
            "nu": NU,
            "field_variables": ["u", "v", "p"],
            "roi": roi,
            "grid_step": GRID_STEP,
            "grid_shape": grid_shape,
            "grid_points": int(x.size),
            "x_min": float(np.nanmin(x)),
            "x_max": float(np.nanmax(x)),
            "y_min": float(np.nanmin(y)),
            "y_max": float(np.nanmax(y)),
        }
        print("CASE_EXPORTED", json.dumps(summary, ensure_ascii=False))
        return summary
    finally:
        client.remove(model)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    mph.option("session", "stand-alone")
    client = mph.start(version="6.3", cores=64)
    summaries = {}
    try:
        for case, spec in CASES.items():
            summaries[case] = extract_case(client, case, spec)
    finally:
        try:
            client.clear()
        except Exception:
            pass

    summary_path = OUT / "editor2_training_field_export_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")

    zip_path = OUT / "editor2_training_field_export.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(summary_path, summary_path.name)
        for item in summaries.values():
            path = Path(item["output_file"])
            zf.write(path, f"data/{path.name}")
    print("SUMMARY_PATH", summary_path)
    print("ZIP_PATH", zip_path)


if __name__ == "__main__":
    main()
