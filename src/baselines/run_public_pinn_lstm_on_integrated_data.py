from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn


RAW_DATA_ROOT = Path(os.environ.get("EAAI_RAW_DATA_ROOT", "data")).resolve()
SINGLE_DATA_DIR = Path(os.environ.get("EAAI_SINGLE_TEXT_DATA", RAW_DATA_ROOT / "single_cylinder")).resolve()
THREE_DATA_DIR = Path(os.environ.get("EAAI_THREE_TEXT_DATA", RAW_DATA_ROOT / "three_cylinder")).resolve()
PUBLIC_ROOT = Path(os.environ.get("PINN_LSTM_REPO_ROOT", "external/PINN-combined-with-LSTM-main")).resolve()
OUT_ROOT = Path(os.environ.get("EAAI_PINN_LSTM_OUT", "results/revision_studies/baselines/pinn_lstm")).resolve()


@dataclass(frozen=True)
class CaseSpec:
    key: str
    title: str
    data_dir: Path
    filename_regex: str
    filename_template: str
    public_model_path: Path


CASES: Dict[str, CaseSpec] = {
    "single": CaseSpec(
        key="single",
        title="Single-cylinder",
        data_dir=SINGLE_DATA_DIR,
        filename_regex=r"sigle_t=(\d+\.\d+)s\.txt",
        filename_template="sigle_t={time:.2f}s.txt",
        public_model_path=PUBLIC_ROOT / "Single-cylinder case" / "Data" / "LSTM-PINN" / "model_LP.pt",
    ),
    "triple": CaseSpec(
        key="triple",
        title="Three-cylinder",
        data_dir=THREE_DATA_DIR,
        filename_regex=r"t=(\d+\.\d+)s\.txt",
        filename_template="t={time:.2f}s.txt",
        public_model_path=PUBLIC_ROOT / "Three-cylinder case" / "Data" / "LSTM-PINN" / "model_LP.pt",
    ),
}


class PublicLPNet(nn.Module):
    """Architecture copied from the public PINN-LSTM Data_output.py."""

    def __init__(self) -> None:
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(3, 20), nn.Tanh()]
        for _ in range(8):
            layers.extend([nn.Linear(20, 20), nn.Tanh()])
        layers.append(nn.Linear(20, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )


def available_times(data_dir: Path, spec: CaseSpec, eval_end: float) -> List[float]:
    rx = re.compile(spec.filename_regex)
    times = []
    for p in data_dir.glob("*.txt"):
        m = rx.match(p.name)
        if m:
            t = float(m.group(1))
            if t <= eval_end + 1e-9:
                times.append(t)
    if not times:
        raise RuntimeError(f"No data files found in {data_dir}")
    return sorted(times)


def read_frame(path: Path, sort_xy: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+")
    required = ["x(mm)", "y(mm)", "u(m/s)", "v(m/s)"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    if sort_xy:
        df = df.sort_values(["x(mm)", "y(mm)"], kind="mergesort").reset_index(drop=True)
    return df


def load_case_truth(data_dir: Path, spec: CaseSpec, times: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xy_ref = None
    uv_frames = []
    for t in times:
        frame = read_frame(data_dir / spec.filename_template.format(time=t))
        xy = frame[["x(mm)", "y(mm)"]].to_numpy(dtype=np.float32)
        if xy_ref is None:
            xy_ref = xy
        elif xy.shape != xy_ref.shape or not np.allclose(xy, xy_ref, atol=1e-6):
            raise ValueError(f"Coordinate mismatch at t={t:.2f}s in {data_dir}")
        uv_frames.append(frame[["u(m/s)", "v(m/s)"]].to_numpy(dtype=np.float32))
    if xy_ref is None:
        raise RuntimeError("No frames loaded")
    return np.asarray(times, dtype=np.float32), xy_ref, np.stack(uv_frames, axis=0)


def localize_xy_for_public_model(xy: np.ndarray) -> np.ndarray:
    x = xy[:, 0]
    y = xy[:, 1]
    x_rng = max(float(x.max() - x.min()), 1e-12)
    y_rng = max(float(y.max() - y.min()), 1e-12)
    x_local = 8.0 * (x - float(x.min())) / x_rng
    y_local = 4.0 * (y - float(y.min())) / y_rng
    return np.stack([x_local, y_local], axis=1).astype(np.float32)


def load_public_model(path: Path, device: torch.device) -> nn.Module:
    if not path.exists():
        raise FileNotFoundError(f"Missing public PINN-LSTM model: {path}")
    model = PublicLPNet().net.to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def grad(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u),
        create_graph=False,
        retain_graph=True,
        only_inputs=True,
    )[0]


def predict_public_lp_batch(model: nn.Module, xy_local: np.ndarray, times: List[float], device: torch.device) -> np.ndarray:
    xy_t = torch.from_numpy(xy_local).to(device=device, dtype=torch.float32)
    x_base = xy_t[:, 0:1]
    y_base = xy_t[:, 1:2]
    x = x_base.repeat(len(times), 1).detach().clone().requires_grad_(True)
    y = y_base.repeat(len(times), 1).detach().clone().requires_grad_(True)
    tt = torch.cat([torch.ones_like(x_base) * float(t) for t in times], dim=0).detach().clone().requires_grad_(True)
    out = model(torch.cat((x, y, tt), dim=1))
    psi = out[:, 0:1]
    u = grad(psi, y)
    v = -grad(psi, x)
    n_points = xy_local.shape[0]
    return torch.cat([u, v], dim=1).reshape(len(times), n_points, 2).detach().cpu().numpy().astype(np.float32)


def metric_block(true_uv: np.ndarray, pred_uv: np.ndarray) -> Dict[str, float]:
    err = pred_uv - true_uv
    speed_true = np.linalg.norm(true_uv, axis=-1)
    speed_pred = np.linalg.norm(pred_uv, axis=-1)
    out = {
        "rmse_u": float(np.sqrt(np.mean(err[..., 0] ** 2))),
        "rmse_v": float(np.sqrt(np.mean(err[..., 1] ** 2))),
        "rmse_vector": float(np.sqrt(np.mean(np.sum(err**2, axis=-1)))),
        "rmse_speed": float(np.sqrt(np.mean((speed_pred - speed_true) ** 2))),
        "mae_vector": float(np.mean(np.linalg.norm(err, axis=-1))),
    }
    eps = 1e-12
    for idx, name in [(0, "u"), (1, "v")]:
        yt = true_uv[..., idx].reshape(-1)
        yp = pred_uv[..., idx].reshape(-1)
        out[f"r2_{name}"] = float(1.0 - np.sum((yt - yp) ** 2) / (np.sum((yt - np.mean(yt)) ** 2) + eps))
    yt = speed_true.reshape(-1)
    yp = speed_pred.reshape(-1)
    out["r2_speed"] = float(1.0 - np.sum((yt - yp) ** 2) / (np.sum((yt - np.mean(yt)) ** 2) + eps))
    return out


def window_metrics(times: np.ndarray, true_uv: np.ndarray, pred_uv: np.ndarray) -> List[Dict[str, float]]:
    window_defs = [
        ("training_or_interpolation_0_5s", 0.0, 5.0, True),
        ("extrapolation_5_7s", 5.0, 7.0, False),
        ("overall_0_7s", 0.0, 7.0, True),
    ]
    rows = []
    for name, lo, hi, include_lo in window_defs:
        mask = times >= lo - 1e-9 if include_lo else times > lo + 1e-9
        mask &= times <= hi + 1e-9
        if not np.any(mask):
            continue
        row = metric_block(true_uv[mask], pred_uv[mask])
        row.update({"window": name, "start_time": float(times[mask][0]), "end_time": float(times[mask][-1]), "n_steps": int(mask.sum())})
        rows.append(row)
    return rows


def plot_error_growth(case_dir: Path, title: str, times: np.ndarray, true_uv: np.ndarray, pred_uv: np.ndarray) -> None:
    err_vec = np.sqrt(np.mean(np.sum((pred_uv - true_uv) ** 2, axis=-1), axis=1))
    err_u = np.sqrt(np.mean((pred_uv[..., 0] - true_uv[..., 0]) ** 2, axis=1))
    err_v = np.sqrt(np.mean((pred_uv[..., 1] - true_uv[..., 1]) ** 2, axis=1))
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    ax.axvspan(0, 5, color="#d7e7f3", alpha=0.35, lw=0, label="0-5 s input interval")
    ax.axvspan(5, 7, color="#f4dfc1", alpha=0.35, lw=0, label="5-7 s extrapolation")
    ax.plot(times, err_vec, color="#1f5a85", lw=1.8, label="Vector RMSE")
    ax.plot(times, err_u, color="#d17a22", lw=1.2, label="u RMSE")
    ax.plot(times, err_v, color="#3f8f5f", lw=1.2, label="v RMSE")
    ax.set_title(f"{title}: public PINN-LSTM on integrated data")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RMSE")
    ax.grid(alpha=0.22, lw=0.5)
    ax.legend(ncol=2, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(case_dir / "error_growth.png")
    fig.savefig(case_dir / "error_growth.pdf")
    plt.close(fig)


def plot_snapshot(case_dir: Path, title: str, xy: np.ndarray, t: float, true_uv: np.ndarray, pred_uv: np.ndarray) -> None:
    true_speed = np.linalg.norm(true_uv, axis=1)
    pred_speed = np.linalg.norm(pred_uv, axis=1)
    err_speed = np.abs(pred_speed - true_speed)
    vmax = float(np.percentile(np.concatenate([true_speed, pred_speed]), 99.0))
    emax = float(np.percentile(err_speed, 99.0))
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.9), constrained_layout=True)
    panels = [
        (true_speed, "Integrated data speed", "turbo", 0.0, vmax),
        (pred_speed, "Public PINN-LSTM speed", "turbo", 0.0, vmax),
        (err_speed, "|speed error|", "magma", 0.0, emax),
    ]
    for ax, (field, subtitle, cmap, vmin, vmax_i) in zip(axes, panels):
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=field, s=2.0, cmap=cmap, vmin=vmin, vmax=vmax_i, linewidths=0)
        ax.set_title(subtitle)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(sc, ax=ax, fraction=0.047, pad=0.02)
    fig.suptitle(f"{title}, t={t:.2f} s")
    fig.savefig(case_dir / f"snapshot_t{t:.2f}.png")
    fig.savefig(case_dir / f"snapshot_t{t:.2f}.pdf")
    plt.close(fig)


def plot_probe_series(case_dir: Path, title: str, times: np.ndarray, true_uv: np.ndarray, pred_uv: np.ndarray) -> None:
    n_points = true_uv.shape[1]
    probe_indices = [n_points // 5, n_points // 2, min(4 * n_points // 5, n_points - 1)]
    fig, axes = plt.subplots(3, 1, figsize=(6.8, 5.2), sharex=True)
    for ax, idx in zip(axes, probe_indices):
        ax.plot(times, true_uv[:, idx, 0], color="#222222", lw=1.2, label="Integrated data u")
        ax.plot(times, pred_uv[:, idx, 0], color="#c55a11", lw=1.0, ls="--", label="Public PINN-LSTM u")
        ax.axvline(5.0, color="#1f5a85", lw=1.0, alpha=0.8)
        ax.set_ylabel(f"P{idx}")
        ax.grid(alpha=0.22, lw=0.5)
    axes[0].legend(frameon=False, ncol=2, loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{title}: representative probe time series")
    fig.tight_layout()
    fig.savefig(case_dir / "probe_u_timeseries.png")
    fig.savefig(case_dir / "probe_u_timeseries.pdf")
    plt.close(fig)


def save_snapshot_csv(case_dir: Path, xy: np.ndarray, t: float, true_uv: np.ndarray, pred_uv: np.ndarray) -> None:
    out_dir = case_dir / "snapshot_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "x(mm)": xy[:, 0],
            "y(mm)": xy[:, 1],
            "u_integrated": true_uv[:, 0],
            "v_integrated": true_uv[:, 1],
            "u_public_pinn_lstm": pred_uv[:, 0],
            "v_public_pinn_lstm": pred_uv[:, 1],
            "err_u": pred_uv[:, 0] - true_uv[:, 0],
            "err_v": pred_uv[:, 1] - true_uv[:, 1],
        }
    ).to_csv(out_dir / f"snapshot_t{t:.2f}.csv", index=False)


def run_case(spec: CaseSpec, args: argparse.Namespace, device: torch.device) -> Tuple[List[Dict[str, float]], Dict[str, object]]:
    data_dir = spec.data_dir
    if not data_dir.exists():
        raise FileNotFoundError(f"Training-code data directory does not exist: {data_dir}")
    times = available_times(data_dir, spec, args.eval_end)
    case_dir = args.outdir / spec.key
    case_dir.mkdir(parents=True, exist_ok=True)

    print(f"{spec.key}: data_dir={data_dir}", flush=True)
    t_arr, xy, true_uv = load_case_truth(data_dir, spec, times)
    xy_local = localize_xy_for_public_model(xy)
    model = load_public_model(spec.public_model_path, device)

    pred_batches = []
    tic = time.time()
    for start in range(0, len(times), args.batch_times):
        batch_times = times[start : start + args.batch_times]
        pred_batches.append(predict_public_lp_batch(model, xy_local, batch_times, device))
        print(f"{spec.key}: predicted {min(start + len(batch_times), len(times))}/{len(times)} frames", flush=True)
    pred_uv = np.concatenate(pred_batches, axis=0)
    runtime_s = time.time() - tic

    rows = window_metrics(t_arr, true_uv, pred_uv)
    for row in rows:
        row.update({"case": spec.key, "method": "Public_PINN_LSTM_integrated_data"})
    pd.DataFrame(rows).to_csv(case_dir / "metrics_by_window.csv", index=False)
    np.savez_compressed(case_dir / "predictions_and_truth_0_7s.npz", times=t_arr, xy=xy, true_uv=true_uv, pred_uv=pred_uv)

    error_rows = []
    for i, t in enumerate(t_arr):
        row = metric_block(true_uv[i : i + 1], pred_uv[i : i + 1])
        row.update({"time": float(t), "case": spec.key, "method": "Public_PINN_LSTM_integrated_data"})
        error_rows.append(row)
    pd.DataFrame(error_rows).to_csv(case_dir / "error_growth.csv", index=False)

    plot_error_growth(case_dir, spec.title, t_arr, true_uv, pred_uv)
    plot_probe_series(case_dir, spec.title, t_arr, true_uv, pred_uv)
    for snap in args.snapshots:
        idx = int(np.argmin(np.abs(t_arr - snap)))
        if abs(float(t_arr[idx]) - snap) <= 0.011:
            plot_snapshot(case_dir, spec.title, xy, float(t_arr[idx]), true_uv[idx], pred_uv[idx])
            save_snapshot_csv(case_dir, xy, float(t_arr[idx]), true_uv[idx], pred_uv[idx])

    meta = {
        "case": spec.key,
        "title": spec.title,
        "method": "Public_PINN_LSTM_integrated_data",
        "data_dir": str(data_dir),
        "public_model_path": str(spec.public_model_path),
        "coordinate_mapping": "x,y from integrated text data are linearly mapped to the public model domain x_local in [0,8], y_local in [0,4]",
        "n_frames": int(len(t_arr)),
        "n_points": int(xy.shape[0]),
        "time_start": float(t_arr[0]),
        "time_end": float(t_arr[-1]),
        "runtime_s": runtime_s,
        "batch_times": int(args.batch_times),
        "device": str(device),
    }
    (case_dir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return rows, meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the public PINN-LSTM algorithm on integrated EAAI text data.")
    parser.add_argument("--cases", default="single,triple")
    parser.add_argument("--eval-end", type=float, default=7.0)
    parser.add_argument("--outdir", type=Path, default=OUT_ROOT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-times", type=int, default=16)
    parser.add_argument("--snapshots", type=float, nargs="+", default=[5.0, 6.0, 7.0])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_style()
    args.outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    all_rows: List[Dict[str, float]] = []
    all_meta: List[Dict[str, object]] = []
    for case_key in [x.strip() for x in args.cases.split(",") if x.strip()]:
        rows, meta = run_case(CASES[case_key], args, device)
        all_rows.extend(rows)
        all_meta.append(meta)
        print(pd.DataFrame(rows).to_string(index=False), flush=True)
    pd.DataFrame(all_rows).to_csv(args.outdir / "public_pinn_lstm_metrics_summary.csv", index=False)
    (args.outdir / "run_metadata.json").write_text(json.dumps(all_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved public PINN-LSTM integrated-data baseline outputs to {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
