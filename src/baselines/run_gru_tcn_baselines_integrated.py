from __future__ import annotations

import argparse
import json
import os
import random
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
from torch.utils.data import DataLoader, Dataset


RAW_DATA_ROOT = Path(os.environ.get("EAAI_RAW_DATA_ROOT", "data")).resolve()
SINGLE_DATA_DIR = Path(os.environ.get("EAAI_SINGLE_TEXT_DATA", RAW_DATA_ROOT / "single_cylinder")).resolve()
THREE_DATA_DIR = Path(os.environ.get("EAAI_THREE_TEXT_DATA", RAW_DATA_ROOT / "three_cylinder")).resolve()
OUT_ROOT = Path(os.environ.get("EAAI_GRU_TCN_OUT", "results/revision_studies/baselines/gru_tcn")).resolve()


@dataclass(frozen=True)
class CaseSpec:
    key: str
    title: str
    data_dir: Path
    filename_template: str


CASES: Dict[str, CaseSpec] = {
    "single": CaseSpec(
        key="single",
        title="Single-cylinder",
        data_dir=SINGLE_DATA_DIR,
        filename_template="sigle_t={time:.2f}s.txt",
    ),
    "triple": CaseSpec(
        key="triple",
        title="Three-cylinder",
        data_dir=THREE_DATA_DIR,
        filename_template="t={time:.2f}s.txt",
    ),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def read_frame(path: Path, sort_xy: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+")
    required = ["x(mm)", "y(mm)", "u(m/s)", "v(m/s)"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    if sort_xy:
        df = df.sort_values(["x(mm)", "y(mm)"], kind="mergesort").reset_index(drop=True)
    return df


def choose_point_indices(n_points: int, max_points: int, seed: int) -> np.ndarray:
    if max_points <= 0 or max_points >= n_points:
        return np.arange(n_points, dtype=np.int32)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_points, size=max_points, replace=False)).astype(np.int32)


def load_case_data(spec: CaseSpec, max_points: int, seed: int, eval_end: float, train_end: float) -> Dict[str, np.ndarray]:
    if not spec.data_dir.exists():
        raise FileNotFoundError(f"Missing data directory: {spec.data_dir}")
    times = [round(i / 100.0, 2) for i in range(int(round(eval_end * 100)) + 1)]
    first_path = spec.data_dir / spec.filename_template.format(time=0.0)
    first = read_frame(first_path)
    point_idx = choose_point_indices(len(first), max_points, seed)
    xy = first.loc[point_idx, ["x(mm)", "y(mm)"]].to_numpy(dtype=np.float32)
    uv_frames = []
    for t in times:
        path = spec.data_dir / spec.filename_template.format(time=t)
        if not path.exists():
            raise FileNotFoundError(f"Missing frame: {path}")
        df = read_frame(path)
        if len(df) != len(first):
            raise ValueError(f"Point count changed at t={t:.2f}: {len(df)} vs {len(first)}")
        uv_frames.append(df.loc[point_idx, ["u(m/s)", "v(m/s)"]].to_numpy(dtype=np.float32))
    uv = np.stack(uv_frames, axis=0)
    times_arr = np.asarray(times, dtype=np.float32)
    train_mask = times_arr <= train_end + 1e-9
    train_uv = uv[train_mask]
    xy_m = xy / 1000.0
    xy_min = xy_m.min(axis=0)
    xy_max = xy_m.max(axis=0)
    xy_norm = 2.0 * (xy_m - xy_min) / np.maximum(xy_max - xy_min, 1e-8) - 1.0
    uv_mean = train_uv.reshape(-1, 2).mean(axis=0)
    uv_std = train_uv.reshape(-1, 2).std(axis=0)
    uv_std = np.where(uv_std < 1e-6, 1.0, uv_std)
    uv_norm = (uv - uv_mean) / uv_std
    return {
        "case": spec.key,
        "data_dir": str(spec.data_dir),
        "times": times_arr,
        "xy": xy.astype(np.float32),
        "xy_norm": xy_norm.astype(np.float32),
        "uv": uv.astype(np.float32),
        "uv_norm": uv_norm.astype(np.float32),
        "uv_mean": uv_mean.astype(np.float32),
        "uv_std": uv_std.astype(np.float32),
        "point_indices": point_idx,
        "n_points_total_file": np.int32(len(first)),
        "train_end": np.float32(train_end),
    }


class SequenceDataset(Dataset):
    def __init__(self, uv_norm: np.ndarray, xy_norm: np.ndarray, times: np.ndarray, train_end: float, input_len: int, pred_len: int) -> None:
        train_count = int(np.searchsorted(times, train_end + 1e-9, side="right"))
        max_start = train_count - input_len - pred_len
        if max_start < 0:
            raise ValueError("Not enough training frames for input_len + pred_len")
        self.uv = uv_norm
        self.xy = xy_norm
        self.input_len = input_len
        self.pred_len = pred_len
        self.starts = np.arange(max_start + 1, dtype=np.int32)
        self.n_points = uv_norm.shape[1]

    def __len__(self) -> int:
        return len(self.starts) * self.n_points

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        point = idx % self.n_points
        start = self.starts[idx // self.n_points]
        seq_uv = self.uv[start : start + self.input_len, point, :]
        xy_rep = np.repeat(self.xy[point][None, :], self.input_len, axis=0)
        x = np.concatenate([seq_uv, xy_rep], axis=1)
        y = self.uv[start + self.input_len : start + self.input_len + self.pred_len, point, :].reshape(-1)
        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32))


class GRUDirect(nn.Module):
    def __init__(self, input_dim: int, hidden: int, layers: int, pred_len: int, dropout: float) -> None:
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, pred_len * 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])


class TCNDirect(nn.Module):
    def __init__(self, input_dim: int, hidden: int, layers: int, pred_len: int, dropout: float) -> None:
        super().__init__()
        mods: List[nn.Module] = []
        in_ch = input_dim
        for i in range(layers):
            dilation = 2**i
            mods.extend([nn.Conv1d(in_ch, hidden, kernel_size=3, padding=dilation, dilation=dilation), nn.GELU(), nn.Dropout(dropout)])
            in_ch = hidden
        self.net = nn.Sequential(*mods)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, pred_len * 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x.transpose(1, 2))
        return self.head(z[:, :, -1])


def build_model(name: str, input_dim: int, hidden: int, layers: int, pred_len: int, dropout: float) -> nn.Module:
    if name == "gru":
        return GRUDirect(input_dim, hidden, layers, pred_len, dropout)
    if name == "tcn":
        return TCNDirect(input_dim, hidden, layers, pred_len, dropout)
    raise ValueError(f"Unknown model: {name}")


def train_model(model: nn.Module, dataset: Dataset, device: torch.device, epochs: int, batch_size: int, lr: float, max_batches: int) -> List[float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    losses = []
    model.train()
    for epoch in range(1, epochs + 1):
        total = 0.0
        n = 0
        for b, (x, y) in enumerate(loader):
            if max_batches > 0 and b >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach().cpu()) * x.shape[0]
            n += x.shape[0]
        losses.append(total / max(n, 1))
        print(f"epoch={epoch}/{epochs} loss={losses[-1]:.6f}", flush=True)
    return losses


def make_initial_sequence(data: Dict[str, np.ndarray], input_len: int) -> np.ndarray:
    times = data["times"]
    train_end = float(data["train_end"])
    end_idx = int(np.searchsorted(times, train_end + 1e-9, side="right")) - 1
    start_idx = end_idx - input_len + 1
    if start_idx < 0:
        raise ValueError("input_len too large")
    return data["uv_norm"][start_idx : end_idx + 1]


def predict_rollout(model: nn.Module, data: Dict[str, np.ndarray], input_len: int, pred_len: int, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    times = data["times"]
    first_eval_index = int(np.searchsorted(times, float(data["train_end"]) + 1e-9, side="right"))
    eval_steps = len(times) - first_eval_index
    seq = make_initial_sequence(data, input_len)
    xy = data["xy_norm"]
    n_points = seq.shape[1]
    preds = []
    with torch.no_grad():
        while len(preds) < eval_steps:
            chunk_len = min(pred_len, eval_steps - len(preds))
            chunks = []
            for start in range(0, n_points, batch_size):
                stop = min(start + batch_size, n_points)
                seq_uv = seq[-input_len:, start:stop, :].transpose(1, 0, 2)
                xy_rep = np.repeat(xy[start:stop, None, :], input_len, axis=1)
                inp = np.concatenate([seq_uv, xy_rep], axis=2)
                out = model(torch.from_numpy(inp.astype(np.float32)).to(device)).cpu().numpy().reshape(stop - start, pred_len, 2)
                chunks.append(out[:, :chunk_len, :])
            pred_chunk = np.concatenate(chunks, axis=0).transpose(1, 0, 2)
            preds.append(pred_chunk)
            seq = np.concatenate([seq, pred_chunk], axis=0)
    pred_norm = np.concatenate(preds, axis=0)[:eval_steps]
    return pred_norm * data["uv_std"][None, None, :] + data["uv_mean"][None, None, :]


def metric_block(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    eps = 1e-12
    err = y_pred - y_true
    speed_true = np.linalg.norm(y_true, axis=-1)
    speed_pred = np.linalg.norm(y_pred, axis=-1)
    out = {
        "rmse_u": float(np.sqrt(np.mean(err[..., 0] ** 2))),
        "rmse_v": float(np.sqrt(np.mean(err[..., 1] ** 2))),
        "rmse_vector": float(np.sqrt(np.mean(np.sum(err**2, axis=-1)))),
        "rmse_speed": float(np.sqrt(np.mean((speed_pred - speed_true) ** 2))),
        "mae_vector": float(np.mean(np.linalg.norm(err, axis=-1))),
    }
    for idx, name in [(0, "u"), (1, "v")]:
        yt = y_true[..., idx].reshape(-1)
        yp = y_pred[..., idx].reshape(-1)
        out[f"r2_{name}"] = float(1.0 - np.sum((yt - yp) ** 2) / (np.sum((yt - np.mean(yt)) ** 2) + eps))
    yt = speed_true.reshape(-1)
    yp = speed_pred.reshape(-1)
    out["r2_speed"] = float(1.0 - np.sum((yt - yp) ** 2) / (np.sum((yt - np.mean(yt)) ** 2) + eps))
    return out


def compute_windows(times: np.ndarray, true_uv: np.ndarray, pred_uv: np.ndarray) -> List[Dict[str, float]]:
    windows = [
        ("extrapolation_5_7s", 5.0, 7.0, False),
        ("overall_5_7s", 5.0, 7.0, False),
    ]
    rows = []
    for name, lo, hi, include_lo in windows:
        mask = times >= lo - 1e-9 if include_lo else times > lo + 1e-9
        mask &= times <= hi + 1e-9
        if np.any(mask):
            row = metric_block(true_uv[mask], pred_uv[mask])
            row.update({"window": name, "start_time": float(times[mask][0]), "end_time": float(times[mask][-1]), "n_steps": int(mask.sum())})
            rows.append(row)
    return rows


def plot_loss(outdir: Path, losses: List[float], title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    ax.plot(np.arange(1, len(losses) + 1), losses, color="#1f5a85", lw=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training MSE")
    ax.set_title(title)
    ax.grid(alpha=0.22, lw=0.5)
    fig.tight_layout()
    fig.savefig(outdir / "loss_curve.png")
    fig.savefig(outdir / "loss_curve.pdf")
    plt.close(fig)


def plot_error_growth(outdir: Path, title: str, times: np.ndarray, true_uv: np.ndarray, pred_uv: np.ndarray) -> None:
    err_vec = np.sqrt(np.mean(np.sum((pred_uv - true_uv) ** 2, axis=-1), axis=1))
    err_u = np.sqrt(np.mean((pred_uv[..., 0] - true_uv[..., 0]) ** 2, axis=1))
    err_v = np.sqrt(np.mean((pred_uv[..., 1] - true_uv[..., 1]) ** 2, axis=1))
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    ax.axvspan(5, 7, color="#f4dfc1", alpha=0.35, lw=0, label="5-7 s extrapolation")
    ax.plot(times, err_vec, color="#1f5a85", lw=1.8, label="Vector RMSE")
    ax.plot(times, err_u, color="#d17a22", lw=1.2, label="u RMSE")
    ax.plot(times, err_v, color="#3f8f5f", lw=1.2, label="v RMSE")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RMSE")
    ax.set_title(title)
    ax.grid(alpha=0.22, lw=0.5)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(outdir / "error_growth.png")
    fig.savefig(outdir / "error_growth.pdf")
    plt.close(fig)


def plot_probe_series(outdir: Path, title: str, times: np.ndarray, true_uv: np.ndarray, pred_uv: np.ndarray) -> None:
    n_points = true_uv.shape[1]
    probe_indices = [n_points // 5, n_points // 2, min(4 * n_points // 5, n_points - 1)]
    fig, axes = plt.subplots(3, 1, figsize=(6.8, 5.2), sharex=True)
    for ax, idx in zip(axes, probe_indices):
        ax.plot(times, true_uv[:, idx, 0], color="#222222", lw=1.2, label="Data u")
        ax.plot(times, pred_uv[:, idx, 0], color="#c55a11", lw=1.0, ls="--", label="Baseline u")
        ax.axvline(5.0, color="#1f5a85", lw=1.0, alpha=0.8)
        ax.set_ylabel(f"P{idx}")
        ax.grid(alpha=0.22, lw=0.5)
    axes[0].legend(frameon=False, ncol=2, loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outdir / "probe_u_timeseries.png")
    fig.savefig(outdir / "probe_u_timeseries.pdf")
    plt.close(fig)


def plot_snapshot(outdir: Path, title: str, xy: np.ndarray, t: float, true_uv: np.ndarray, pred_uv: np.ndarray) -> None:
    true_speed = np.linalg.norm(true_uv, axis=1)
    pred_speed = np.linalg.norm(pred_uv, axis=1)
    err_speed = np.abs(pred_speed - true_speed)
    vmax = float(np.percentile(np.concatenate([true_speed, pred_speed]), 99))
    emax = float(np.percentile(err_speed, 99))
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.9), constrained_layout=True)
    panels = [
        (true_speed, "Data speed", "turbo", 0.0, vmax),
        (pred_speed, "Baseline speed", "turbo", 0.0, vmax),
        (err_speed, "|speed error|", "magma", 0.0, emax),
    ]
    for ax, (field, label, cmap, vmin, vmax_i) in zip(axes, panels):
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=field, s=2.0, cmap=cmap, vmin=vmin, vmax=vmax_i, linewidths=0)
        ax.set_title(label)
        ax.set_xlabel("x(mm)")
        ax.set_ylabel("y(mm)")
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(sc, ax=ax, fraction=0.047, pad=0.02)
    fig.suptitle(f"{title}, t={t:.2f} s")
    fig.savefig(outdir / f"snapshot_t{t:.2f}.png")
    fig.savefig(outdir / f"snapshot_t{t:.2f}.pdf")
    plt.close(fig)


def run_case_model(args: argparse.Namespace, spec: CaseSpec, model_name: str, device: torch.device) -> Dict[str, object]:
    data = load_case_data(spec, args.max_points, args.seed, args.eval_end, args.train_end)
    outdir = args.outdir / spec.key / model_name
    outdir.mkdir(parents=True, exist_ok=True)
    dataset = SequenceDataset(data["uv_norm"], data["xy_norm"], data["times"], args.train_end, args.input_len, args.pred_len)
    model = build_model(model_name, 4, args.hidden, args.layers, args.pred_len, args.dropout).to(device)
    tic = time.time()
    losses = train_model(model, dataset, device, args.epochs, args.batch_size, args.lr, args.max_batches_per_epoch)
    train_seconds = time.time() - tic
    pred_uv = predict_rollout(model, data, args.input_len, args.pred_len, args.batch_size, device)
    first_eval_index = int(np.searchsorted(data["times"], args.train_end + 1e-9, side="right"))
    eval_times = data["times"][first_eval_index:]
    true_uv = data["uv"][first_eval_index:]
    rows = compute_windows(eval_times, true_uv, pred_uv)
    for row in rows:
        row.update({"case": spec.key, "model": model_name})
    summary = rows[0].copy()
    summary.update({"case": spec.key, "model": model_name, "train_seconds": train_seconds, "final_train_loss": float(losses[-1]), "n_params": sum(p.numel() for p in model.parameters())})

    pd.DataFrame(rows).to_csv(outdir / "metrics_by_window.csv", index=False)
    pd.DataFrame({"epoch": np.arange(1, len(losses) + 1), "loss": losses}).to_csv(outdir / "loss.csv", index=False)
    per_time_rows = []
    for i, t in enumerate(eval_times):
        r = metric_block(true_uv[i : i + 1], pred_uv[i : i + 1])
        r.update({"time": float(t), "case": spec.key, "model": model_name})
        per_time_rows.append(r)
    pd.DataFrame(per_time_rows).to_csv(outdir / "error_growth.csv", index=False)
    np.savez_compressed(outdir / "predictions_and_truth_5_7s.npz", times=eval_times, xy=data["xy"], true_uv=true_uv, pred_uv=pred_uv)
    torch.save({"state_dict": model.state_dict(), "args": vars(args), "case": spec.key, "model": model_name}, outdir / "model.pt")

    plot_loss(outdir, losses, f"{spec.title} {model_name.upper()} training loss")
    plot_error_growth(outdir, f"{spec.title} {model_name.upper()} error growth", eval_times, true_uv, pred_uv)
    plot_probe_series(outdir, f"{spec.title} {model_name.upper()} probe time series", eval_times, true_uv, pred_uv)
    for snap in args.snapshots:
        idx = int(np.argmin(np.abs(eval_times - snap)))
        if abs(float(eval_times[idx]) - snap) <= 0.011:
            plot_snapshot(outdir, f"{spec.title} {model_name.upper()}", data["xy"], float(eval_times[idx]), true_uv[idx], pred_uv[idx])

    meta = {
        "case": spec.key,
        "model": model_name,
        "data_dir": data["data_dir"],
        "n_points_total_file": int(data["n_points_total_file"]),
        "n_points_used": int(data["xy"].shape[0]),
        "input_len": int(args.input_len),
        "pred_len": int(args.pred_len),
        "train_end": float(args.train_end),
        "eval_end": float(args.eval_end),
        "epochs": int(args.epochs),
        "max_batches_per_epoch": int(args.max_batches_per_epoch),
        "batch_size": int(args.batch_size),
        "hidden": int(args.hidden),
        "layers": int(args.layers),
        "train_seconds": train_seconds,
    }
    (outdir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standard GRU/TCN temporal baselines on integrated EAAI data.")
    parser.add_argument("--cases", default="single,triple")
    parser.add_argument("--models", default="gru")
    parser.add_argument("--outdir", type=Path, default=OUT_ROOT)
    parser.add_argument("--train-end", type=float, default=5.0)
    parser.add_argument("--eval-end", type=float, default=7.0)
    parser.add_argument("--input-len", type=int, default=50)
    parser.add_argument("--pred-len", type=int, default=50)
    parser.add_argument("--max-points", type=int, default=3000)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-batches-per-epoch", type=int, default=180)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--snapshots", type=float, nargs="+", default=[5.01, 6.0, 7.0])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    set_style()
    args.outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rows = []
    for case_key in [x.strip() for x in args.cases.split(",") if x.strip()]:
        for model_name in [x.strip().lower() for x in args.models.split(",") if x.strip()]:
            print(f"Running {case_key}/{model_name}", flush=True)
            row = run_case_model(args, CASES[case_key], model_name, device)
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    pd.DataFrame(rows).to_csv(args.outdir / "gru_tcn_metrics_summary.csv", index=False)
    (args.outdir / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    print(f"Saved GRU/TCN baseline outputs to {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
