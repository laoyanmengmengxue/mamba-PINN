from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn


PUBLIC_ROOT = Path(os.environ.get("PINN_LSTM_REPO_ROOT", "external/PINN-combined-with-LSTM-main")).resolve()
SINGLE_CFD_DIR = Path(os.environ.get("EAAI_SINGLE_CFD_DIR", "data/single_cylinder/CFD")).resolve()
THREE_CFD_DIR = Path(os.environ.get("EAAI_THREE_CFD_DIR", "data/three_cylinder/CFD")).resolve()
OUT_ROOT = Path(os.environ.get("EAAI_PINN_LSTM_OUT", "results/revision_studies/baselines/pinn_lstm")).resolve()


@dataclass(frozen=True)
class CaseSpec:
    key: str
    title: str
    model_path: Path
    cfd_dir: Path
    file_style: str
    velocity_y_prefix: str = "w"


CASES: Dict[str, CaseSpec] = {
    "single": CaseSpec(
        key="single",
        title="Single-cylinder",
        model_path=PUBLIC_ROOT / "Single-cylinder case" / "Data" / "LSTM-PINN" / "model_LP.pt",
        cfd_dir=SINGLE_CFD_DIR,
        file_style="original_index",
    ),
    "triple": CaseSpec(
        key="triple",
        title="Three-cylinder",
        model_path=PUBLIC_ROOT / "Three-cylinder case" / "Data" / "LSTM-PINN" / "model_LP.pt",
        cfd_dir=THREE_CFD_DIR,
        file_style="decimal_reindexed",
    ),
}


class PublicLPNet(nn.Module):
    """Network copied from the public PINN-LSTM Data_output.py."""

    def __init__(self) -> None:
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(3, 20), nn.Tanh()]
        for _ in range(8):
            layers.extend([nn.Linear(20, 20), nn.Tanh()])
        layers.append(nn.Linear(20, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def grad(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u),
        create_graph=False,
        retain_graph=True,
        only_inputs=True,
    )[0]


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


def time_to_suffix(spec: CaseSpec, t: float, prefix: str) -> str:
    if spec.file_style == "original_index":
        return f"{prefix}{int(round(8500 + 100 * t))}"
    if spec.file_style == "decimal_reindexed":
        return f"{prefix}{t:.2f}"
    raise ValueError(f"Unknown file style: {spec.file_style}")


def read_cfd_component(spec: CaseSpec, t: float, component: str) -> np.ndarray:
    prefix = component if component != "v" else spec.velocity_y_prefix
    path = spec.cfd_dir / time_to_suffix(spec, t, prefix)
    if not path.exists():
        raise FileNotFoundError(f"Missing CFD file: {path}")
    return np.loadtxt(path)


def crop_original_region(arr: np.ndarray) -> np.ndarray:
    # MATLAB source code used y(40:120), x(200:360), with 1-based inclusive indices.
    if arr.shape[0] < 120 or arr.shape[1] < 360:
        raise ValueError(f"Unexpected CFD matrix shape: {arr.shape}")
    return arr[39:120, 199:360].astype(np.float32)


def read_true_uv(spec: CaseSpec, t: float) -> np.ndarray:
    u = crop_original_region(read_cfd_component(spec, t, "u"))
    v = crop_original_region(read_cfd_component(spec, t, "v"))
    return np.stack([u.reshape(-1), v.reshape(-1)], axis=1)


def load_public_model(path: Path, device: torch.device) -> nn.Module:
    if not path.exists():
        raise FileNotFoundError(f"Missing public model: {path}")
    model = PublicLPNet().net.to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_public_lp_batch(model: nn.Module, times: List[float], device: torch.device) -> np.ndarray:
    x_line = torch.linspace(0.0, 8.0, 161, device=device, dtype=torch.float32).reshape(161, 1)
    y_line = torch.linspace(0.0, 4.0, 81, device=device, dtype=torch.float32).reshape(81, 1)
    base_x = x_line.repeat(81, 1)
    base_y = y_line.repeat_interleave(161).reshape(13041, 1)
    x = base_x.repeat(len(times), 1).detach().clone().requires_grad_(True)
    y = base_y.repeat(len(times), 1).detach().clone().requires_grad_(True)
    tt_chunks = [torch.ones_like(base_x) * float(t) for t in times]
    tt = torch.cat(tt_chunks, dim=0).detach().clone().requires_grad_(True)
    out = model(torch.cat((x, y, tt), dim=1))
    psi = out[:, 0:1]
    u = grad(psi, y)
    v = -grad(psi, x)
    uv = torch.cat([u, v], dim=1).reshape(len(times), 13041, 2)
    return uv.detach().cpu().numpy().astype(np.float32)


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


def make_times(eval_end: float) -> List[float]:
    n = int(round(eval_end * 100))
    return [round(i / 100.0, 2) for i in range(n + 1)]


def window_metrics(times: np.ndarray, true_uv: np.ndarray, pred_uv: np.ndarray) -> List[Dict[str, float]]:
    windows = [
        ("training_or_interpolation_0_5s", 0.0, 5.0, True, True),
        ("extrapolation_5_7s", 5.0, 7.0, False, True),
        ("overall_0_7s", 0.0, 7.0, True, True),
    ]
    rows = []
    for name, lo, hi, include_lo, include_hi in windows:
        if include_lo:
            mask = times >= lo - 1e-9
        else:
            mask = times > lo + 1e-9
        if include_hi:
            mask &= times <= hi + 1e-9
        else:
            mask &= times < hi - 1e-9
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
    ax.plot(times, err_vec, color="#1f5a85", lw=1.8, label="Vector RMSE")
    ax.plot(times, err_u, color="#d17a22", lw=1.2, label="u RMSE")
    ax.plot(times, err_v, color="#3f8f5f", lw=1.2, label="v RMSE")
    ax.axvspan(0, 5, color="#d7e7f3", alpha=0.35, lw=0, label="0-5 s input interval")
    ax.axvspan(5, 7, color="#f4dfc1", alpha=0.35, lw=0, label="5-7 s extrapolation")
    ax.set_title(f"{title}: public PINN-LSTM error growth")
    ax.set_xlabel("Re-indexed time (s)")
    ax.set_ylabel("RMSE")
    ax.legend(ncol=2, frameon=False, loc="upper left")
    ax.grid(alpha=0.22, lw=0.5)
    fig.tight_layout()
    fig.savefig(case_dir / "error_growth.png")
    fig.savefig(case_dir / "error_growth.pdf")
    plt.close(fig)


def plot_snapshot(case_dir: Path, title: str, t: float, true_uv: np.ndarray, pred_uv: np.ndarray) -> None:
    true_speed = np.linalg.norm(true_uv, axis=1).reshape(81, 161)
    pred_speed = np.linalg.norm(pred_uv, axis=1).reshape(81, 161)
    err_speed = np.abs(pred_speed - true_speed)
    vmax = float(np.percentile(np.concatenate([true_speed.ravel(), pred_speed.ravel()]), 99.0))
    emax = float(np.percentile(err_speed.ravel(), 99.0))
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.7), constrained_layout=True)
    panels = [
        (true_speed, "CFD speed", "turbo", 0.0, vmax),
        (pred_speed, "Public PINN-LSTM speed", "turbo", 0.0, vmax),
        (err_speed, "|speed error|", "magma", 0.0, emax),
    ]
    for ax, (field, subtitle, cmap, vmin, vmax_i) in zip(axes, panels):
        im = ax.imshow(field, origin="lower", extent=[0, 8, 0, 4], aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax_i)
        ax.set_title(subtitle)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.047, pad=0.02)
    fig.suptitle(f"{title}, t={t:.2f} s")
    fig.savefig(case_dir / f"snapshot_t{t:.2f}.png")
    fig.savefig(case_dir / f"snapshot_t{t:.2f}.pdf")
    plt.close(fig)


def plot_probe_series(case_dir: Path, title: str, times: np.ndarray, true_uv: np.ndarray, pred_uv: np.ndarray) -> None:
    probe_indices = [161 * 20 + 35, 161 * 40 + 80, 161 * 60 + 125]
    probe_indices = [min(max(i, 0), true_uv.shape[1] - 1) for i in probe_indices]
    fig, axes = plt.subplots(len(probe_indices), 1, figsize=(6.8, 5.2), sharex=True)
    if len(probe_indices) == 1:
        axes = [axes]
    for ax, idx in zip(axes, probe_indices):
        ax.plot(times, true_uv[:, idx, 0], color="#222222", lw=1.2, label="CFD u")
        ax.plot(times, pred_uv[:, idx, 0], color="#c55a11", lw=1.0, ls="--", label="PINN-LSTM u")
        ax.axvline(5.0, color="#1f5a85", lw=1.0, alpha=0.8)
        ax.set_ylabel(f"P{idx}")
        ax.grid(alpha=0.22, lw=0.5)
    axes[0].legend(frameon=False, ncol=2, loc="upper right")
    axes[-1].set_xlabel("Re-indexed time (s)")
    fig.suptitle(f"{title}: representative probe time series")
    fig.tight_layout()
    fig.savefig(case_dir / "probe_u_timeseries.png")
    fig.savefig(case_dir / "probe_u_timeseries.pdf")
    plt.close(fig)


def save_snapshot_data(case_dir: Path, t: float, true_uv: np.ndarray, pred_uv: np.ndarray) -> None:
    data_dir = case_dir / "snapshot_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    xy_x = np.tile(np.linspace(0.0, 8.0, 161), 81)
    xy_y = np.repeat(np.linspace(0.0, 4.0, 81), 161)
    df = pd.DataFrame(
        {
            "x": xy_x,
            "y": xy_y,
            "u_cfd": true_uv[:, 0],
            "v_cfd": true_uv[:, 1],
            "u_pinn_lstm": pred_uv[:, 0],
            "v_pinn_lstm": pred_uv[:, 1],
            "err_u": pred_uv[:, 0] - true_uv[:, 0],
            "err_v": pred_uv[:, 1] - true_uv[:, 1],
        }
    )
    df.to_csv(data_dir / f"snapshot_t{t:.2f}.csv", index=False)


def run_case(spec: CaseSpec, args: argparse.Namespace, device: torch.device) -> Tuple[List[Dict[str, float]], Dict[str, object]]:
    case_dir = args.outdir / spec.key
    case_dir.mkdir(parents=True, exist_ok=True)
    model = load_public_model(spec.model_path, device)
    times_list = make_times(args.eval_end)
    true_frames = []
    pred_frames = []
    tic = time.time()
    for start in range(0, len(times_list), args.batch_times):
        batch_times = times_list[start : start + args.batch_times]
        for t in batch_times:
            true_frames.append(read_true_uv(spec, t))
        pred_batch = predict_public_lp_batch(model, batch_times, device)
        pred_frames.extend([pred_batch[i] for i in range(pred_batch.shape[0])])
        print(f"{spec.key}: {min(start + len(batch_times), len(times_list))}/{len(times_list)} frames", flush=True)
    runtime_s = time.time() - tic
    times = np.asarray(times_list, dtype=np.float32)
    true_uv = np.stack(true_frames, axis=0)
    pred_uv = np.stack(pred_frames, axis=0)

    rows = window_metrics(times, true_uv, pred_uv)
    for row in rows:
        row.update({"case": spec.key, "method": "Public_PINN_LSTM"})
    pd.DataFrame(rows).to_csv(case_dir / "metrics_by_window.csv", index=False)
    np.savez_compressed(case_dir / "predictions_and_truth_0_7s.npz", times=times, true_uv=true_uv, pred_uv=pred_uv)

    plot_error_growth(case_dir, spec.title, times, true_uv, pred_uv)
    plot_probe_series(case_dir, spec.title, times, true_uv, pred_uv)
    for snap in args.snapshots:
        idx = int(round(snap * 100))
        if 0 <= idx < len(times):
            plot_snapshot(case_dir, spec.title, float(times[idx]), true_uv[idx], pred_uv[idx])
            save_snapshot_data(case_dir, float(times[idx]), true_uv[idx], pred_uv[idx])

    meta = {
        "case": spec.key,
        "title": spec.title,
        "method": "Public_PINN_LSTM",
        "public_model_path": str(spec.model_path),
        "cfd_dir": str(spec.cfd_dir),
        "file_style": spec.file_style,
        "eval_end": float(args.eval_end),
        "dt": 0.01,
        "grid": {"nx": 161, "ny": 81, "n_points": 13041, "x_local": "0:0.05:8", "y_local": "0:0.05:4"},
        "original_crop": {"matlab_x_indices": "200:360", "matlab_y_indices": "40:120"},
        "runtime_s": runtime_s,
    }
    (case_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return rows, meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the public PINN-LSTM baseline on current EAAI CFD datasets.")
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
    (args.outdir / "run_metadata.json").write_text(json.dumps(all_meta, indent=2), encoding="utf-8")
    print(f"Saved public PINN-LSTM baseline outputs to {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
