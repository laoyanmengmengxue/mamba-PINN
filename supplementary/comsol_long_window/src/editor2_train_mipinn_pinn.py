from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def r2_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom <= 1e-30:
        return float("nan")
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def rmse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


class FieldDataset:
    def __init__(self, path: Path, obs_end: float, eval_end: float, probe_count: int, seed: int):
        raw = np.load(path, allow_pickle=True)
        self.path = path
        self.case = str(raw["case"].item())
        self.x = raw["x"].astype(np.float32)
        self.y = raw["y"].astype(np.float32)
        self.t = raw["t"].astype(np.float32)
        self.t_abs = raw["t_abs"].astype(np.float32)
        self.u = raw["u"].astype(np.float32)
        self.v = raw["v"].astype(np.float32)
        self.nu = float(raw["nu"].item())
        self.obs_end = float(obs_end)
        self.eval_end = float(eval_end)
        self.x_min = float(np.min(self.x))
        self.x_max = float(np.max(self.x))
        self.y_min = float(np.min(self.y))
        self.y_max = float(np.max(self.y))
        self.t_min = float(np.min(self.t))
        self.t_max = float(eval_end)
        self.t_idx_obs = np.flatnonzero((self.t >= self.t_min - 1e-8) & (self.t <= obs_end + 1e-8))
        self.t_idx_future = np.flatnonzero((self.t > obs_end + 1e-8) & (self.t <= eval_end + 1e-8))
        if self.t_idx_obs.size < 8:
            raise RuntimeError("Observation window is too short for training.")
        if self.t_idx_future.size < 2:
            raise RuntimeError("Future window is too short for evaluation.")
        rng = np.random.default_rng(seed)
        n_probe = min(int(probe_count), self.x.size)
        self.probe_idx = np.sort(rng.choice(self.x.size, size=n_probe, replace=False))
        obs_uv = np.stack([self.u[self.t_idx_obs[:, None], self.probe_idx], self.v[self.t_idx_obs[:, None], self.probe_idx]], axis=-1)
        self.uv_mean = obs_uv.reshape(-1, 2).mean(axis=0).astype(np.float32)
        self.uv_std = (obs_uv.reshape(-1, 2).std(axis=0) + 1e-6).astype(np.float32)

    @property
    def dt(self) -> float:
        return float(np.median(np.diff(self.t)))

    def norm_x(self, x: np.ndarray) -> np.ndarray:
        return (2.0 * (x - self.x_min) / max(self.x_max - self.x_min, 1e-12) - 1.0).astype(np.float32)

    def norm_y(self, y: np.ndarray) -> np.ndarray:
        return (2.0 * (y - self.y_min) / max(self.y_max - self.y_min, 1e-12) - 1.0).astype(np.float32)

    def norm_t(self, t: np.ndarray) -> np.ndarray:
        return (2.0 * (t - self.t_min) / max(self.t_max - self.t_min, 1e-12) - 1.0).astype(np.float32)

    def sample_data(self, batch: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        ti = np.random.choice(self.t_idx_obs, size=batch, replace=True)
        pi = np.random.choice(self.probe_idx, size=batch, replace=True)
        inp = np.stack([self.norm_x(self.x[pi]), self.norm_y(self.y[pi]), self.norm_t(self.t[ti])], axis=1)
        out = np.stack([self.u[ti, pi], self.v[ti, pi]], axis=1)
        return torch.from_numpy(inp).to(device), torch.from_numpy(out).to(device)

    def sample_phys(self, batch: int, device: torch.device) -> torch.Tensor:
        ti_pool = np.flatnonzero((self.t >= self.t_min - 1e-8) & (self.t <= self.eval_end + 1e-8))
        ti = np.random.choice(ti_pool, size=batch, replace=True)
        pi = np.random.choice(self.x.size, size=batch, replace=True)
        inp = np.stack([self.norm_x(self.x[pi]), self.norm_y(self.y[pi]), self.norm_t(self.t[ti])], axis=1)
        return torch.from_numpy(inp).to(device)

    def tensors_for_times(self, time_indices: np.ndarray, device: torch.device) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
        frames = []
        true_u = []
        true_v = []
        xn = self.norm_x(self.x)
        yn = self.norm_y(self.y)
        for ti in time_indices:
            tn = self.norm_t(np.full_like(self.x, self.t[ti], dtype=np.float32))
            frames.append(np.stack([xn, yn, tn], axis=1))
            true_u.append(self.u[ti])
            true_v.append(self.v[ti])
        inp = torch.from_numpy(np.concatenate(frames, axis=0).astype(np.float32)).to(device)
        return inp, np.concatenate(true_u), np.concatenate(true_v)


class PINN2D(nn.Module):
    def __init__(self, width: int = 64, depth: int = 8):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(3, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 3)]
        self.net = nn.Sequential(*layers)

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        return self.net(xyt)


class SelectiveSSMBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.in_proj = nn.Linear(dim, dim)
        self.delta_proj = nn.Linear(dim, dim)
        self.b_proj = nn.Linear(dim, dim)
        self.c_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.a_log = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.silu(self.in_proj(x))
        delta = F.softplus(self.delta_proj(z)) + 1e-3
        a = -F.softplus(self.a_log).view(1, 1, -1) - 1e-3
        b = self.b_proj(z)
        h = torch.zeros(x.shape[0], x.shape[2], device=x.device, dtype=x.dtype)
        outs = []
        for i in range(x.shape[1]):
            decay = torch.exp(delta[:, i, :] * a.squeeze(1))
            h = decay * h + (1.0 - decay) * b[:, i, :]
            outs.append(self.c_proj(h))
        y = torch.stack(outs, dim=1)
        return self.out_proj(y)


class MambaTemporalPrior(nn.Module):
    def __init__(self, input_dim: int, hidden: int, out_steps: int, blocks: int = 4):
        super().__init__()
        self.out_steps = int(out_steps)
        self.embed = nn.Linear(input_dim, hidden)
        self.blocks = nn.ModuleList([SelectiveSSMBlock(hidden) for _ in range(blocks)])
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, self.out_steps * 2))

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        x = self.embed(seq)
        for block in self.blocks:
            x = x + block(x)
        h = self.norm(x[:, -1, :])
        return self.head(h).view(seq.shape[0], self.out_steps, 2)


def physics_loss(model: nn.Module, xyt: torch.Tensor, ds: FieldDataset) -> torch.Tensor:
    xyt = xyt.detach().clone().requires_grad_(True)
    pred = model(xyt)
    u = pred[:, 0:1]
    v = pred[:, 1:2]
    p = pred[:, 2:3]

    ones = torch.ones_like(u)
    gu = torch.autograd.grad(u, xyt, ones, create_graph=True, retain_graph=True)[0]
    gv = torch.autograd.grad(v, xyt, ones, create_graph=True, retain_graph=True)[0]
    gp = torch.autograd.grad(p, xyt, ones, create_graph=True, retain_graph=True)[0]

    sx = 2.0 / max(ds.x_max - ds.x_min, 1e-12)
    sy = 2.0 / max(ds.y_max - ds.y_min, 1e-12)
    st = 2.0 / max(ds.t_max - ds.t_min, 1e-12)

    u_x, u_y, u_t = gu[:, 0:1] * sx, gu[:, 1:2] * sy, gu[:, 2:3] * st
    v_x, v_y, v_t = gv[:, 0:1] * sx, gv[:, 1:2] * sy, gv[:, 2:3] * st
    p_x, p_y = gp[:, 0:1] * sx, gp[:, 1:2] * sy

    gu_x = torch.autograd.grad(gu[:, 0:1], xyt, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    gu_y = torch.autograd.grad(gu[:, 1:2], xyt, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    gv_x = torch.autograd.grad(gv[:, 0:1], xyt, torch.ones_like(v), create_graph=True, retain_graph=True)[0]
    gv_y = torch.autograd.grad(gv[:, 1:2], xyt, torch.ones_like(v), create_graph=True, retain_graph=True)[0]
    u_xx, u_yy = gu_x[:, 0:1] * sx * sx, gu_y[:, 1:2] * sy * sy
    v_xx, v_yy = gv_x[:, 0:1] * sx * sx, gv_y[:, 1:2] * sy * sy

    cont = u_x + v_y
    mom_u = u_t + u * u_x + v * u_y + p_x - ds.nu * (u_xx + u_yy)
    mom_v = v_t + u * v_x + v * v_y + p_y - ds.nu * (v_xx + v_yy)
    return (cont.square().mean() + mom_u.square().mean() + mom_v.square().mean())


def train_mamba(ds: FieldDataset, args: argparse.Namespace, device: torch.device, out_dir: Path) -> tuple[dict, dict[str, np.ndarray]]:
    obs_times = ds.t_idx_obs
    future_times = ds.t_idx_future
    probe_idx = ds.probe_idx
    uv_obs = np.stack([ds.u[obs_times[:, None], probe_idx], ds.v[obs_times[:, None], probe_idx]], axis=-1).transpose(1, 0, 2)
    uv_obs_n = (uv_obs - ds.uv_mean) / ds.uv_std
    xyn = np.stack([ds.norm_x(ds.x[probe_idx]), ds.norm_y(ds.y[probe_idx])], axis=1)

    out_steps = min(args.mamba_out_steps, max(1, obs_times.size - args.mamba_input_steps - 1))
    model = MambaTemporalPrior(input_dim=4, hidden=args.mamba_hidden, out_steps=out_steps, blocks=args.mamba_blocks).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.mamba_lr, weight_decay=1e-5)
    max_start = obs_times.size - args.mamba_input_steps - out_steps
    if max_start < 1:
        raise RuntimeError("Observation window is too short for the requested Mamba target horizon.")

    losses = []
    for epoch in range(1, args.epochs_mamba + 1):
        pids = np.random.randint(0, probe_idx.size, size=args.mamba_batch)
        starts = np.random.randint(0, max_start + 1, size=args.mamba_batch)
        seq = np.empty((args.mamba_batch, args.mamba_input_steps, 4), dtype=np.float32)
        target = np.empty((args.mamba_batch, out_steps, 2), dtype=np.float32)
        seq[:, :, 0:2] = xyn[pids, None, :]
        for i, (pid, s) in enumerate(zip(pids, starts)):
            seq[i, :, 2:4] = uv_obs_n[pid, s : s + args.mamba_input_steps, :]
            target[i] = uv_obs_n[pid, s + args.mamba_input_steps : s + args.mamba_input_steps + out_steps, :]
        seq_t = torch.from_numpy(seq).to(device)
        target_t = torch.from_numpy(target).to(device)
        opt.zero_grad(set_to_none=True)
        pred = model(seq_t)
        loss = F.mse_loss(pred, target_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach().cpu()))
        if epoch % args.report_every == 0 or epoch == 1:
            print(f"[Mamba] epoch={epoch} loss={losses[-1]:.6e}", flush=True)

    total_future = future_times.size
    with torch.no_grad():
        last = uv_obs_n[:, -args.mamba_input_steps :, :].copy()
        generated = []
        remaining = total_future
        while remaining > 0:
            seq = np.empty((probe_idx.size, args.mamba_input_steps, 4), dtype=np.float32)
            seq[:, :, 0:2] = xyn[:, None, :]
            seq[:, :, 2:4] = last[:, -args.mamba_input_steps :, :]
            pred_block = model(torch.from_numpy(seq).to(device)).cpu().numpy()
            take = min(remaining, pred_block.shape[1])
            generated.append(pred_block[:, :take, :])
            last = np.concatenate([last, pred_block[:, :take, :]], axis=1)
            remaining -= take
        future_n = np.concatenate(generated, axis=1)
    future_uv = future_n * ds.uv_std + ds.uv_mean
    future_uv = future_uv.transpose(1, 0, 2).astype(np.float32)

    true_future = np.stack([ds.u[future_times[:, None], probe_idx], ds.v[future_times[:, None], probe_idx]], axis=-1)
    metrics = {
        "mamba_input_steps": args.mamba_input_steps,
        "mamba_out_steps_trained": out_steps,
        "mamba_direct_horizon_s": float(out_steps * ds.dt),
        "mamba_rollout_steps": int(total_future),
        "mamba_final_train_loss": losses[-1],
    }
    for end in args.eval_ends:
        tids = np.flatnonzero((ds.t[future_times] > ds.obs_end + 1e-8) & (ds.t[future_times] <= float(end) + 1e-8))
        if tids.size == 0:
            continue
        pred = future_uv[tids]
        true = true_future[tids]
        pred_speed = np.sqrt(pred[..., 0] ** 2 + pred[..., 1] ** 2)
        true_speed = np.sqrt(true[..., 0] ** 2 + true[..., 1] ** 2)
        metrics[f"mamba_speed_r2_5_{end:g}"] = r2_np(true_speed.ravel(), pred_speed.ravel())
        metrics[f"mamba_speed_rmse_5_{end:g}"] = rmse_np(true_speed.ravel(), pred_speed.ravel())

    torch.save(model.state_dict(), out_dir / "mamba_temporal_prior.pt")
    with (out_dir / "mamba_training_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss"])
        for i, loss in enumerate(losses, 1):
            writer.writerow([i, loss])
    return metrics, {"future_uv": future_uv, "future_times": ds.t[future_times], "probe_idx": probe_idx}


def sample_pseudo(ds: FieldDataset, pseudo: dict[str, np.ndarray], batch: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    future_uv = pseudo["future_uv"]
    probe_idx = pseudo["probe_idx"]
    future_times = pseudo["future_times"]
    ti = np.random.randint(0, future_uv.shape[0], size=batch)
    pi_local = np.random.randint(0, probe_idx.size, size=batch)
    pi = probe_idx[pi_local]
    inp = np.stack([ds.norm_x(ds.x[pi]), ds.norm_y(ds.y[pi]), ds.norm_t(future_times[ti].astype(np.float32))], axis=1)
    out = future_uv[ti, pi_local, :]
    return torch.from_numpy(inp.astype(np.float32)).to(device), torch.from_numpy(out.astype(np.float32)).to(device)


def evaluate_model(model: nn.Module, ds: FieldDataset, args: argparse.Namespace, device: torch.device, out_dir: Path) -> dict:
    model.eval()
    metrics: dict[str, float] = {}
    growth_rows = []
    with torch.no_grad():
        for end in args.eval_ends:
            tidx = np.flatnonzero((ds.t > ds.obs_end + 1e-8) & (ds.t <= float(end) + 1e-8))
            if tidx.size == 0:
                continue
            inp, true_u, true_v = ds.tensors_for_times(tidx, device)
            preds = []
            for start in range(0, inp.shape[0], args.eval_batch):
                preds.append(model(inp[start : start + args.eval_batch]).detach().cpu().numpy()[:, :2])
            pred = np.concatenate(preds, axis=0)
            pred_u, pred_v = pred[:, 0], pred[:, 1]
            true_speed = np.sqrt(true_u**2 + true_v**2)
            pred_speed = np.sqrt(pred_u**2 + pred_v**2)
            suffix = f"5_{end:g}"
            metrics[f"u_r2_{suffix}"] = r2_np(true_u, pred_u)
            metrics[f"v_r2_{suffix}"] = r2_np(true_v, pred_v)
            metrics[f"speed_r2_{suffix}"] = r2_np(true_speed, pred_speed)
            metrics[f"u_rmse_{suffix}"] = rmse_np(true_u, pred_u)
            metrics[f"v_rmse_{suffix}"] = rmse_np(true_v, pred_v)
            metrics[f"speed_rmse_{suffix}"] = rmse_np(true_speed, pred_speed)
            metrics[f"speed_mae_{suffix}"] = mae_np(true_speed, pred_speed)

        for ti in ds.t_idx_future:
            inp, true_u, true_v = ds.tensors_for_times(np.array([ti]), device)
            preds = []
            for start in range(0, inp.shape[0], args.eval_batch):
                preds.append(model(inp[start : start + args.eval_batch]).detach().cpu().numpy()[:, :2])
            pred = np.concatenate(preds, axis=0)
            true_speed = np.sqrt(true_u**2 + true_v**2)
            pred_speed = np.sqrt(pred[:, 0] ** 2 + pred[:, 1] ** 2)
            growth_rows.append(
                {
                    "t_shift_s": float(ds.t[ti]),
                    "t_abs_s": float(ds.t_abs[ti]),
                    "speed_r2": r2_np(true_speed, pred_speed),
                    "speed_rmse": rmse_np(true_speed, pred_speed),
                    "speed_mae": mae_np(true_speed, pred_speed),
                }
            )

    with (out_dir / "error_growth_by_time.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(growth_rows[0].keys()))
        writer.writeheader()
        writer.writerows(growth_rows)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.2), dpi=180)
    t = [r["t_abs_s"] for r in growth_rows]
    axes[0].plot(t, [r["speed_rmse"] for r in growth_rows], color="#2D6F8E", lw=1.8)
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("speed RMSE")
    axes[0].grid(alpha=0.25)
    axes[1].plot(t, [r["speed_r2"] for r in growth_rows], color="#D0644B", lw=1.8)
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("speed R2")
    axes[1].grid(alpha=0.25)
    fig.suptitle(f"{ds.case} {args.variant}: 65--75 s error growth")
    fig.tight_layout()
    fig.savefig(out_dir / "error_growth.png", bbox_inches="tight")
    plt.close(fig)
    return metrics


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    data_path = Path(args.data_root) / f"editor2_{args.case}_roi_grid_60_75s.npz"
    ds = FieldDataset(data_path, obs_end=args.obs_end, eval_end=max(args.eval_ends), probe_count=args.probe_count, seed=args.seed)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / args.case / args.variant / f"run_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"RUN_DIR {out_dir}", flush=True)
    print(f"DEVICE {device}", flush=True)
    print(f"CASE {ds.case} mesh={ds.x.size} times={ds.t.size} dt={ds.dt:.4f} probes={ds.probe_idx.size}", flush=True)
    print(f"WINDOW obs=0--{args.obs_end:g}s extrap=5--{max(args.eval_ends):g}s", flush=True)

    run_config = vars(args).copy()
    run_config.update(
        {
            "data_path": str(data_path),
            "device": str(device),
            "mesh_points": int(ds.x.size),
            "roi": [ds.x_min, ds.x_max, ds.y_min, ds.y_max],
            "time_count": int(ds.t.size),
            "dt": ds.dt,
            "nu": ds.nu,
            "x_bounds": [ds.x_min, ds.x_max],
            "y_bounds": [ds.y_min, ds.y_max],
            "dataset_status": "independent_COMSOL_generated_case_not_public_dataset_extension",
            "truth_usage": "velocity labels used only for observation-window data loss and final evaluation",
        }
    )
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    pseudo = None
    mamba_metrics = {}
    if args.variant == "mipinn":
        mamba_metrics, pseudo = train_mamba(ds, args, device, out_dir)
        (out_dir / "mamba_metrics.json").write_text(json.dumps(mamba_metrics, indent=2), encoding="utf-8")

    model = PINN2D(width=args.width, depth=args.depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs_pinn, 1), eta_min=args.lr * 0.05)
    history = []
    start_time = time.time()
    for epoch in range(1, args.epochs_pinn + 1):
        model.train()
        x_data, y_data = ds.sample_data(args.batch_data, device)
        x_phys = ds.sample_phys(args.batch_phys, device)
        pred_data = model(x_data)[:, :2]
        l_data = F.mse_loss(pred_data, y_data)
        l_phys = physics_loss(model, x_phys, ds)
        l_pseudo = torch.tensor(0.0, device=device)
        if args.variant == "mipinn" and pseudo is not None:
            x_ps, y_ps = sample_pseudo(ds, pseudo, args.batch_pseudo, device)
            l_pseudo = F.mse_loss(model(x_ps)[:, :2], y_ps)
        loss = args.w_data * l_data + args.w_phys * l_phys + args.w_pseudo * l_pseudo
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        scheduler.step()
        row = {
            "epoch": epoch,
            "loss": float(loss.detach().cpu()),
            "data_loss": float(l_data.detach().cpu()),
            "phys_loss": float(l_phys.detach().cpu()),
            "pseudo_loss": float(l_pseudo.detach().cpu()),
            "lr": float(scheduler.get_last_lr()[0]),
            "elapsed_s": time.time() - start_time,
        }
        history.append(row)
        if epoch % args.report_every == 0 or epoch == 1:
            print(
                f"[PINN] epoch={epoch} loss={row['loss']:.6e} data={row['data_loss']:.3e} "
                f"phys={row['phys_loss']:.3e} pseudo={row['pseudo_loss']:.3e}",
                flush=True,
            )

    torch.save(model.state_dict(), out_dir / "final_model.pt")
    with (out_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    metrics = evaluate_model(model, ds, args, device, out_dir)
    metrics.update(mamba_metrics)
    metrics.update(
        {
            "case": ds.case,
            "variant": args.variant,
            "epochs_pinn": args.epochs_pinn,
            "epochs_mamba": args.epochs_mamba if args.variant == "mipinn" else 0,
            "probe_count": int(ds.probe_idx.size),
            "obs_window_shift_s": [0.0, args.obs_end],
            "eval_window_shift_s": [args.obs_end, max(args.eval_ends)],
            "obs_window_abs_s": [float(ds.t_abs[0]), float(ds.t_abs[ds.t_idx_obs[-1]])],
            "eval_window_abs_s": [float(ds.t_abs[ds.t_idx_future[0]]), float(ds.t_abs[ds.t_idx_future[-1]])],
            "wall_time_s": time.time() - start_time,
        }
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (out_dir / "final_metrics.txt").open("w", encoding="utf-8") as f:
        for key in sorted(metrics):
            f.write(f"{key}: {metrics[key]}\n")
    print("FINAL_METRICS", json.dumps(metrics, ensure_ascii=False), flush=True)
    print("DONE", out_dir, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PINN and MI-PINN on independent COMSOL long-window data.")
    parser.add_argument("--case", choices=["single", "three"], required=True)
    parser.add_argument("--variant", choices=["pinn", "mipinn"], required=True)
    parser.add_argument("--data-root", default="outputs/comsol_long_window/data")
    parser.add_argument("--out-root", default="outputs/comsol_long_window/runs")
    parser.add_argument("--obs-end", type=float, default=5.0)
    parser.add_argument("--eval-ends", type=float, nargs="+", default=[7.0, 10.0, 15.0])
    parser.add_argument("--probe-count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--epochs-pinn", type=int, default=5000)
    parser.add_argument("--epochs-mamba", type=int, default=2000)
    parser.add_argument("--batch-data", type=int, default=4096)
    parser.add_argument("--batch-phys", type=int, default=20000)
    parser.add_argument("--batch-pseudo", type=int, default=1140)
    parser.add_argument("--mamba-batch", type=int, default=256)
    parser.add_argument("--mamba-input-steps", type=int, default=5)
    parser.add_argument("--mamba-out-steps", type=int, default=40)
    parser.add_argument("--mamba-hidden", type=int, default=128)
    parser.add_argument("--mamba-blocks", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mamba-lr", type=float, default=1e-3)
    parser.add_argument("--w-data", type=float, default=1.0)
    parser.add_argument("--w-phys", type=float, default=1.0)
    parser.add_argument("--w-pseudo", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--report-every", type=int, default=200)
    parser.add_argument("--eval-batch", type=int, default=65536)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
