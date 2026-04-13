#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import math
import os
import random
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.io
from scipy.ndimage import uniform_filter1d
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except Exception:
    pd = None
    PANDAS_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    TORCH_AVAILABLE = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def load_complex_matrix(mat_path: str, var_name: str = "H_active_matrix") -> np.ndarray:
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"MAT file not found: {mat_path}")
    mat = scipy.io.loadmat(mat_path)

    if var_name in mat:
        H = mat[var_name]
    else:
        candidates = []
        for k, v in mat.items():
            if k.startswith("__"):
                continue
            if isinstance(v, np.ndarray) and v.ndim == 2 and np.iscomplexobj(v):
                candidates.append((k, v.size))
        if not candidates:
            raise ValueError(
                f"Variable '{var_name}' not found and no complex 2D array candidate in MAT file."
            )
        best_key = sorted(candidates, key=lambda x: x[1], reverse=True)[0][0]
        H = mat[best_key]
        print(f"[INFO] '{var_name}' not found, using '{best_key}' instead.")

    H = np.asarray(H)
    if H.ndim != 2:
        raise ValueError(f"Expected a 2D complex matrix, got shape: {H.shape}")
    if not np.iscomplexobj(H):
        raise ValueError("Loaded matrix is not complex. Please provide complex CSI matrix.")
    return H


def estimate_common_phase(H: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    unit = H / np.maximum(np.abs(H), eps)
    return np.angle(np.sum(unit, axis=1))


def _fit_cfo_linear(train_H: np.ndarray) -> Tuple[float, float]:
    if train_H.shape[0] < 2:
        return 0.0, 0.0
    t = np.arange(train_H.shape[0], dtype=float)
    cpe_train = np.unwrap(estimate_common_phase(train_H))
    slope, intercept = np.polyfit(t, cpe_train, deg=1)
    return float(slope), float(intercept)


def _apply_cfo_linear(H: np.ndarray, slope: float, intercept: float, start_idx: int) -> np.ndarray:
    if H.shape[0] == 0:
        return H.copy()
    t = np.arange(start_idx, start_idx + H.shape[0], dtype=float)
    cfo_phase = slope * t + intercept
    return H * np.exp(-1j * cfo_phase[:, None])


def _remove_residual_cpe_per_frame(H: np.ndarray) -> Tuple[np.ndarray, float]:
    if H.shape[0] == 0:
        return H.copy(), 0.0
    cpe_res = estimate_common_phase(H)
    cpe_res_unwrapped = np.unwrap(cpe_res)
    return H * np.exp(-1j * cpe_res_unwrapped[:, None]), float(np.std(cpe_res_unwrapped))


def phase_smooth(H: np.ndarray, window: int = 11) -> np.ndarray:
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    if H.shape[0] < window:
        return H.copy()

    mag = np.abs(H)
    phase = np.unwrap(np.angle(H), axis=0)
    phase_sm = uniform_filter1d(phase, size=window, axis=0, mode="nearest")
    return mag * np.exp(1j * phase_sm)


def phase_smooth_causal(H: np.ndarray, window: int = 11) -> np.ndarray:
    # Causal smoothing avoids future look-ahead within each split.
    window = max(3, int(window))
    if H.shape[0] < 2:
        return H.copy()
    mag = np.abs(H)
    phase = np.unwrap(np.angle(H), axis=0)
    csum = np.cumsum(phase, axis=0)
    sm = csum.copy()
    if H.shape[0] > window:
        sm[window:] = csum[window:] - csum[:-window]
    counts = np.minimum(np.arange(1, H.shape[0] + 1), window).astype(np.float64)[:, None]
    phase_sm = sm / counts
    return mag * np.exp(1j * phase_sm)


def _split_by_ratio(H: np.ndarray, train_ratio: float, val_ratio: float):
    n = H.shape[0]
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return H[:train_end], H[train_end:val_end], H[val_end:], train_end, val_end


def build_variants_no_leakage(
    H_raw: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    smooth_window: int = 11,
) -> Dict[str, np.ndarray]:
    H_train, H_val, H_test, train_end, val_end = _split_by_ratio(H_raw, train_ratio, val_ratio)
    slope, intercept = _fit_cfo_linear(H_train)

    H_train_cfo = _apply_cfo_linear(H_train, slope, intercept, start_idx=0)
    H_val_cfo = _apply_cfo_linear(H_val, slope, intercept, start_idx=train_end)
    H_test_cfo = _apply_cfo_linear(H_test, slope, intercept, start_idx=val_end)

    H_train_comp, cpe_std_train = _remove_residual_cpe_per_frame(H_train_cfo)
    H_val_comp, cpe_std_val = _remove_residual_cpe_per_frame(H_val_cfo)
    H_test_comp, cpe_std_test = _remove_residual_cpe_per_frame(H_test_cfo)

    H_cfo_cpe = np.concatenate([H_train_comp, H_val_comp, H_test_comp], axis=0)
    H_cfo_cpe_smooth = np.concatenate(
        [
            phase_smooth_causal(H_train_comp, window=smooth_window),
            phase_smooth_causal(H_val_comp, window=smooth_window),
            phase_smooth_causal(H_test_comp, window=smooth_window),
        ],
        axis=0,
    )
    print(
        "[INFO] Leakage-safe compensation stats: "
        f"CFO slope(train-fit)={slope:.6f} rad/step, "
        f"residual CPE std train/val/test="
        f"{cpe_std_train:.6f}/{cpe_std_val:.6f}/{cpe_std_test:.6f} rad"
    )
    return {
        "raw": H_raw,
        "cfo_cpe_comp": H_cfo_cpe,
        "cfo_cpe_comp_phase_smooth": H_cfo_cpe_smooth,
    }


def complex_to_features(H: np.ndarray) -> np.ndarray:
    return np.hstack([np.real(H), np.imag(H)]).astype(np.float32)


def features_to_complex(F: np.ndarray, n_sc: int) -> np.ndarray:
    return F[..., :n_sc] + 1j * F[..., n_sc:]


def split_contiguous(data: np.ndarray, train_ratio: float, val_ratio: float):
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return data[:train_end], data[train_end:val_end], data[val_end:]


def make_windows(data: np.ndarray, lookback: int, horizon: int):
    n = len(data) - lookback - horizon + 1
    if n <= 0:
        return np.empty((0, lookback, data.shape[1]), dtype=np.float32), np.empty(
            (0, horizon, data.shape[1]), dtype=np.float32
        )
    X = []
    y = []
    for i in range(n):
        X.append(data[i : i + lookback])
        y.append(data[i + lookback : i + lookback + horizon])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


if TORCH_AVAILABLE:
    class PositionalEncoder(nn.Module):
        def __init__(self, dropout: float = 0.1, max_seq_len: int = 5000, d_model: int = 256, batch_first: bool = True):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            self.batch_first = batch_first
            self.x_dim = 1 if batch_first else 0
            position = torch.arange(max_seq_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(max_seq_len, d_model)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.dropout(x + self.pe[: x.size(self.x_dim)])


    class TimeSeriesTransformer(nn.Module):
        def __init__(
            self,
            input_size: int,
            dim_val: int,
            n_layers: int,
            n_heads: int,
            dropout: float,
            out_seq_len: int,
            dim_feedforward: Optional[int] = None,
        ):
            super().__init__()
            self.out_seq_len = out_seq_len
            self.encoder_input_layer = nn.Linear(in_features=input_size, out_features=dim_val)
            self.decoder_input_layer = nn.Linear(in_features=input_size, out_features=dim_val)
            self.positional_encoding_layer = PositionalEncoder(d_model=dim_val, dropout=dropout)
            ff_dim = int(dim_feedforward) if dim_feedforward and int(dim_feedforward) > 0 else 4 * dim_val

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=dim_val,
                nhead=n_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                batch_first=True,
            )
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=dim_val,
                nhead=n_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_layers, norm=None)
            self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=n_layers, norm=None)
            self.linear_mapping = nn.Linear(in_features=dim_val, out_features=input_size)

        def forward(self, src: torch.Tensor, tgt: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            src = self.encoder_input_layer(src)
            src = self.positional_encoding_layer(src)
            memory = self.encoder(src=src)
            tgt = self.decoder_input_layer(tgt)
            tgt = self.positional_encoding_layer(tgt)
            out = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
            out = self.linear_mapping(out)
            return out[:, -self.out_seq_len :, :]


    class ChannelLSTM(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float = 0.1):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.n_layers = num_layers
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
            self.fc = nn.Linear(hidden_dim, input_dim)

        def init_hidden(self, batch_size: int, device: torch.device):
            w = next(self.parameters()).data
            return (
                w.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                w.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
            )

        def forward(self, x: torch.Tensor, hidden=None):
            out, hidden = self.lstm(x, hidden)
            return self.fc(out), hidden


    class ChannelGRU(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float = 0.1):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.n_layers = num_layers
            self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
            self.fc = nn.Linear(hidden_dim, input_dim)

        def init_hidden(self, batch_size: int, device: torch.device):
            w = next(self.parameters()).data
            return w.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)

        def forward(self, x: torch.Tensor, hidden=None):
            out, hidden = self.gru(x, hidden)
            return self.fc(out), hidden


    class ChannelCNN(nn.Module):
        def __init__(self, lookback: int, input_channels: int, hidden_dim: int = 128):
            super().__init__()
            self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=hidden_dim, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.pool = nn.AdaptiveMaxPool1d(1)
            self.fc = nn.Linear(hidden_dim, input_channels)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x.permute(0, 2, 1)
            x = self.conv1(x)
            x = self.relu(x)
            x = self.pool(x).squeeze(-1)
            return self.fc(x)


    class ChannelMLP(nn.Module):
        def __init__(self, lookback: int, input_channels: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_channels * lookback, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, input_channels),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


def generate_square_subsequent_mask(dim: int, device) -> "torch.Tensor":
    mask = (torch.triu(torch.ones(dim, dim), diagonal=1) == 1)
    mask = mask.float().masked_fill(mask, float("-inf"))
    return mask.to(device)


def nmse_db(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = np.sum(np.abs(y_true - y_pred) ** 2)
    sig = np.sum(np.abs(y_true) ** 2)
    if sig <= 0:
        return np.nan
    return float(10.0 * np.log10(err / sig))


def _build_torch_model(model_name: str, input_dim: int, args):
    if model_name == "gru":
        return ChannelGRU(input_dim=input_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout)
    if model_name == "lstm":
        return ChannelLSTM(input_dim=input_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout)
    if model_name == "transformer":
        return TimeSeriesTransformer(
            input_size=input_dim,
            dim_val=args.hidden_dim,
            n_layers=args.num_layers,
            n_heads=args.n_heads,
            dropout=args.dropout,
            out_seq_len=args.horizon,
            dim_feedforward=(args.transformer_ff_dim if args.transformer_ff_dim > 0 else None),
        )
    if model_name == "cnn":
        return ChannelCNN(lookback=args.lookback, input_channels=input_dim, hidden_dim=args.hidden_dim)
    if model_name == "mlp":
        return ChannelMLP(lookback=args.lookback, input_channels=input_dim)
    raise ValueError(f"Unknown torch model: {model_name}")


def _torch_forward_train(model, model_name: str, xb: "torch.Tensor", yb: "torch.Tensor", horizon: int, device):
    if model_name == "transformer":
        sos = xb[:, -1:, :]
        dec_in = sos if horizon == 1 else torch.cat([sos, yb[:, :-1, :]], dim=1)
        tgt_mask = generate_square_subsequent_mask(dec_in.size(1), device=device)
        out = model(xb.float(), dec_in.float(), tgt_mask=tgt_mask)
        return out, yb.float()
    if model_name in {"gru", "lstm"}:
        h = model.init_hidden(xb.size(0), device)
        out, _ = model(xb.float(), h)
        out = out[:, -horizon:, :]
        return out, yb.float()
    out = model(xb.float())
    target = yb[:, -1, :].float()
    return out, target


def _torch_predict_sequence(model, model_name: str, xb: "torch.Tensor", horizon: int, device):
    # Returns [B, H, F]
    if model_name == "transformer":
        src = model.encoder_input_layer(xb.float())
        src = model.positional_encoding_layer(src)
        memory = model.encoder(src=src)
        dec = xb[:, -1:, :].float()
        preds = []
        for _ in range(horizon):
            tgt_mask = generate_square_subsequent_mask(dec.size(1), device=device)
            tgt = model.decoder_input_layer(dec)
            tgt = model.positional_encoding_layer(tgt)
            out = model.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
            out = model.linear_mapping(out)
            next_step = out[:, -1:, :]
            preds.append(next_step)
            dec = torch.cat([dec, next_step], dim=1)
        return torch.cat(preds, dim=1)

    if model_name in {"gru", "lstm"}:
        h = model.init_hidden(xb.size(0), device)
        _, h = model(xb.float(), h)
        dec = xb[:, -1:, :].float()
        preds = []
        for _ in range(horizon):
            out, h = model(dec, h)
            next_step = out[:, -1:, :]
            preds.append(next_step)
            dec = next_step
        return torch.cat(preds, dim=1)

    # MLP/CNN: recursive one-step rollout
    cur = xb.float()
    preds = []
    for _ in range(horizon):
        next_step = model(cur).unsqueeze(1)
        preds.append(next_step)
        cur = torch.cat([cur[:, 1:, :], next_step], dim=1)
    return torch.cat(preds, dim=1)


def _run_torch_model(X_train, y_train, X_val, y_val, X_test, y_test, args, device, model_name: str, variant_name: str):
    input_dim = X_train.shape[-1]
    model = _build_torch_model(model_name, input_dim=input_dim, args=args).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    best_val = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(args.epochs):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred, target = _torch_forward_train(model, model_name, xb, yb, args.horizon, device)
            loss = criterion(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred, target = _torch_forward_train(model, model_name, xb, yb, args.horizon, device)
                val_loss += criterion(pred, target).item()
        val_loss /= max(1, len(val_loader))

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[{variant_name}/{model_name}] Epoch {epoch+1:03d} | train={tr_loss:.6f} | val={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                break
        scheduler.step(val_loss)

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            pred_seq = _torch_predict_sequence(model, model_name, xb, args.horizon, device)
            preds.append(pred_seq[:, -1, :].cpu().numpy())
    pred_last = np.concatenate(preds, axis=0)
    true_last = y_test[:, -1, :]
    return pred_last, true_last, float(best_val)


def _run_ridge_model(X_train, y_train, X_val, y_val, X_test, y_test, args):
    X_train_2d = X_train.reshape(X_train.shape[0], -1)
    X_val_2d = X_val.reshape(X_val.shape[0], -1)
    X_test_2d = X_test.reshape(X_test.shape[0], -1)
    y_train_last = y_train[:, -1, :]
    y_val_last = y_val[:, -1, :]

    best_val = float("inf")
    best_model = None
    for alpha in [0.1, 1.0, 10.0]:
        reg = Ridge(alpha=alpha, random_state=args.seed)
        reg.fit(X_train_2d, y_train_last)
        pred_val = reg.predict(X_val_2d)
        val_loss = float(np.mean((pred_val - y_val_last) ** 2))
        if val_loss < best_val:
            best_val = val_loss
            best_model = reg

    pred_last = best_model.predict(X_test_2d)
    true_last = y_test[:, -1, :]
    return pred_last, true_last, float(best_val)


def run_single_variant(
    variant_name: str,
    H_variant: np.ndarray,
    args,
    device,
    model_name: str,
) -> Dict[str, float]:
    n_sc = H_variant.shape[1]
    features = complex_to_features(H_variant)
    train_raw, val_raw, test_raw = split_contiguous(features, args.train_ratio, args.val_ratio)

    scaler = StandardScaler()
    train_norm = scaler.fit_transform(train_raw)
    val_norm = scaler.transform(val_raw)
    test_norm = scaler.transform(test_raw)

    X_train, y_train = make_windows(train_norm, args.lookback, args.horizon)
    X_val, y_val = make_windows(val_norm, args.lookback, args.horizon)
    X_test, y_test = make_windows(test_norm, args.lookback, args.horizon)
    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise ValueError("Not enough data for window generation. Increase data length or reduce lookback/horizon.")

    if model_name == "ridge":
        pred_norm, true_norm, best_val = _run_ridge_model(X_train, y_train, X_val, y_val, X_test, y_test, args)
    else:
        if not TORCH_AVAILABLE:
            raise RuntimeError(f"Model '{model_name}' needs torch, but torch is not available.")
        pred_norm, true_norm, best_val = _run_torch_model(
            X_train, y_train, X_val, y_val, X_test, y_test, args, device, model_name=model_name, variant_name=variant_name
        )

    pred_denorm = scaler.inverse_transform(pred_norm)
    true_denorm = scaler.inverse_transform(true_norm)

    pred_complex = features_to_complex(pred_denorm, n_sc)
    true_complex = features_to_complex(true_denorm, n_sc)
    model_nmse = nmse_db(true_complex, pred_complex)

    test_start = len(train_raw) + len(val_raw)
    idx = np.arange(len(X_test)) + args.lookback
    true_naive = H_variant[test_start + idx]
    pred_naive = H_variant[test_start + idx - 1]
    naive_nmse = nmse_db(true_naive, pred_naive)

    return {
        "variant": variant_name,
        "model_name": model_name,
        "naive_nmse_db": naive_nmse,
        "model_nmse_db": model_nmse,
        "delta_vs_naive_db": model_nmse - naive_nmse,
        "best_val_loss": float(best_val),
    }


def _auto_find_doppler_csv() -> Optional[str]:
    candidates = []
    search_dir = "paper_results"
    if os.path.isdir(search_dir):
        for name in os.listdir(search_dir):
            if name.endswith("_track_comparison_wide.csv"):
                candidates.append(os.path.join(search_dir, name))
    if not candidates and os.path.isdir(search_dir):
        for name in os.listdir(search_dir):
            if name.endswith("_track_comparison_long.csv"):
                candidates.append(os.path.join(search_dir, name))
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def _load_doppler_table(path: str):
    if not PANDAS_AVAILABLE:
        raise RuntimeError("pandas is required for joint NMSE+Doppler table export.")
    df = pd.read_csv(path)

    if {"preprocessing", "peak_track_mode", "mae_hz", "rmse_hz", "bias_hz"}.issubset(df.columns):
        wide = df.pivot(
            index="preprocessing",
            columns="peak_track_mode",
            values=["mae_hz", "rmse_hz", "bias_hz"],
        )
        wide.columns = [f"{metric}_{mode}" for metric, mode in wide.columns]
        wide = wide.reset_index().rename(columns={"preprocessing": "variant"})
        return wide

    if "preprocessing" in df.columns:
        df = df.rename(columns={"preprocessing": "variant"})
    if "variant" not in df.columns:
        raise ValueError(f"Doppler table '{path}' must contain 'variant' or 'preprocessing' column.")
    return df


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare raw vs compensated CSI variants on the same real dataset."
    )
    p.add_argument("--mat-path", type=str, required=True, help="Path to .mat file containing complex CSI matrix.")
    p.add_argument("--mat-var-name", type=str, default="H_active_matrix", help="Variable name in MAT file.")
    p.add_argument("--lookback", type=int, default=10)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--transformer-ff-dim",
        type=int,
        default=0,
        help="Transformer feedforward width. 0 means use 4*hidden_dim. "
             "Set 1024 to match the legacy magnitude script.",
    )
    p.add_argument("--smooth-window", type=int, default=11)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-csv", type=str, default="results/raw_vs_compensation.csv")
    p.add_argument(
        "--doppler-csv",
        type=str,
        default="",
        help="Optional path to Doppler-consistency CSV (wide or long) from CPE analysis. "
             "If omitted, script auto-searches paper_results/*_track_comparison_*.csv.",
    )
    p.add_argument(
        "--joint-out-csv",
        type=str,
        default="",
        help="Output path for merged NMSE + Doppler-consistency table. "
             "Default: <out-csv> with suffix _joint.csv",
    )
    p.add_argument(
        "--model",
        type=str,
        default="auto",
        choices=["auto", "gru", "lstm", "transformer", "cnn", "mlp", "ridge"],
        help="Single model to run. Use --models to run multiple models in one pass.",
    )
    p.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated model list, e.g. 'gru,lstm,transformer,cnn,mlp'. If set, overrides --model.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    if TORCH_AVAILABLE:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
    else:
        device = None
        print("Device: N/A (torch not installed, using sklearn model)")

    H_raw = load_complex_matrix(args.mat_path, var_name=args.mat_var_name)
    print(f"[INFO] Loaded matrix shape: {H_raw.shape}")

    variants = build_variants_no_leakage(
        H_raw,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        smooth_window=args.smooth_window,
    )
    all_results = []

    if args.models.strip():
        model_list = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    else:
        model_list = [args.model.lower()]
    if model_list == ["auto"]:
        model_list = ["gru"] if TORCH_AVAILABLE else ["ridge"]

    valid_models = {"gru", "lstm", "transformer", "cnn", "mlp", "ridge"}
    for m in model_list:
        if m not in valid_models:
            raise ValueError(f"Unsupported model '{m}'. Valid models: {sorted(valid_models)}")
        if m != "ridge" and not TORCH_AVAILABLE:
            raise RuntimeError(f"Model '{m}' requires torch, but torch is not available.")

    print(f"[INFO] Models to run: {model_list}")

    for variant_name, H_variant in variants.items():
        print(f"\n=== Running variant: {variant_name} ===")
        for model_name in model_list:
            set_seed(args.seed)  # Keep initialization fair across variants/models.
            result = run_single_variant(variant_name, H_variant, args, device, model_name=model_name)
            all_results.append(result)
            print(
                f"[{variant_name}/{model_name}] "
                f"naive_nmse={result['naive_nmse_db']:.4f} dB | "
                f"model_nmse={result['model_nmse_db']:.4f} dB | "
                f"delta={result['delta_vs_naive_db']:.4f} dB"
            )

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["variant", "model_name", "naive_nmse_db", "model_nmse_db", "delta_vs_naive_db", "best_val_loss"],
        )
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nSaved comparison table to: {args.out_csv}")

    doppler_csv = args.doppler_csv.strip() or _auto_find_doppler_csv()
    if doppler_csv:
        try:
            doppler_df = _load_doppler_table(doppler_csv)
            if PANDAS_AVAILABLE:
                nmse_df = pd.DataFrame(all_results)
                joint_df = nmse_df.merge(doppler_df, on="variant", how="left")
                joint_out = args.joint_out_csv.strip()
                if not joint_out:
                    root, ext = os.path.splitext(args.out_csv)
                    joint_out = f"{root}_joint{ext or '.csv'}"
                os.makedirs(os.path.dirname(joint_out) or ".", exist_ok=True)
                joint_df.to_csv(joint_out, index=False)
                print(f"Saved joint NMSE + Doppler table to: {joint_out}")
        except Exception as e:
            print(f"[WARN] Failed to merge Doppler table '{doppler_csv}': {e}")
    else:
        print("[INFO] No Doppler CSV found; skipped joint table export.")

    # Print sorted summary.
    sorted_results = sorted(all_results, key=lambda x: x["model_nmse_db"])
    print("\n=== Summary (lower NMSE is better) ===")
    for r in sorted_results:
        print(
            f"{r['variant']:<28} "
            f"{r['model_name'].upper()} NMSE={r['model_nmse_db']:.4f} dB | "
            f"Naive={r['naive_nmse_db']:.4f} dB | "
            f"Delta={r['delta_vs_naive_db']:.4f} dB"
        )


if __name__ == "__main__":
    main()
