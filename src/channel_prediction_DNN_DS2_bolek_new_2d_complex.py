# -*- coding: utf-8 -*-
import os
import time
import math
import numpy as np
import scipy.io
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
import random
from typing import Tuple, Optional

# ==============================================================================
# 1. 模型定义 (Model Definitions)
# ==============================================================================

class PositionalEncoder(nn.Module):
    def __init__(self, dropout: float = 0.1, max_seq_len: int = 5000, d_model: int = 512, batch_first: bool = True):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        self.x_dim = 1 if batch_first else 0
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(self.x_dim)]
        return self.dropout(x)

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size: int, dim_val: int, n_encoder_layers: int, n_decoder_layers: int,
                 n_heads: int, dropout_encoder: float, dropout_decoder: float, dropout_pos_enc: float,
                 dim_feedforward_encoder: int, dim_feedforward_decoder: int,
                 max_seq_len: int, dec_seq_len: int, out_seq_len: int):
        super().__init__()
        self.dec_seq_len = dec_seq_len
        self.out_seq_len = out_seq_len
        self.encoder_input_layer = nn.Linear(in_features=input_size, out_features=dim_val)
        self.decoder_input_layer = nn.Linear(in_features=input_size, out_features=dim_val)
        self.positional_encoding_layer = PositionalEncoder(d_model=dim_val, dropout=dropout_pos_enc, max_seq_len=max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_val, nhead=n_heads, 
                                                  dim_feedforward=dim_feedforward_encoder, 
                                                  dropout=dropout_encoder, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_encoder_layers, norm=None)
        decoder_layer = nn.TransformerDecoderLayer(d_model=dim_val, nhead=n_heads, 
                                                  dim_feedforward=dim_feedforward_decoder, 
                                                  dropout=dropout_decoder, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=n_decoder_layers, norm=None)
        self.linear_mapping = nn.Linear(in_features=dim_val, out_features=input_size)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None, 
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        src = self.encoder_input_layer(src)
        src = self.positional_encoding_layer(src)
        src = self.encoder(src=src)
        tgt = self.decoder_input_layer(tgt)
        tgt = self.positional_encoding_layer(tgt)
        decoder_output = self.decoder(tgt=tgt, memory=src, tgt_mask=tgt_mask, memory_mask=src_mask)
        decoder_output = self.linear_mapping(decoder_output)
        return decoder_output

# --- LSTM CLASS ---
class ChannelLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float = 0.1, output_dim: int = None):
        super(ChannelLSTM, self).__init__()
        if output_dim is None: output_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size: int, device: torch.device):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden

# --- GRU CLASS ---
class ChannelGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float = 0.1, output_dim: int = None):
        super(ChannelGRU, self).__init__()
        if output_dim is None: output_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        out, hidden = self.gru(x, hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size: int, device: torch.device):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden

class ChannelMLP(nn.Module):
    def __init__(self, lookback: int, input_channels: int):
        super(ChannelMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_channels * lookback, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, input_channels)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ChannelCNN(nn.Module):
    def __init__(self, lookback: int, input_channels: int, hidden_dim: int = 128, dropout: float = 0.1):
        super(ChannelCNN, self).__init__()
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

# ==============================================================================
# 2. 数据处理函数
# ==============================================================================

def load_complex_data(config):
    """
    加载数据并将其拆分为实部和虚部特征。
    """
    print(f"\n--- Loading Data from {config['file_path']} ---")
    if not os.path.exists(config['file_path']):
        local_path = os.path.basename(config['file_path'])
        if os.path.exists(local_path):
            config['file_path'] = local_path
        else:
            print("Error: File not found.")
            return None, None, None, 0

    try:
        mat = scipy.io.loadmat(config['file_path'])
        if config['mat_var_name'] not in mat:
            # 自动寻找最大的变量
            var_name = max(mat.keys(), key=lambda k: mat[k].size if isinstance(mat[k], np.ndarray) else 0)
            print(f"Variable '{config['mat_var_name']}' not found. Using '{var_name}' instead.")
            H_complex = mat[var_name]
        else:
            H_complex = mat[config['mat_var_name']]
    except Exception as e:
        print(f"Error loading MAT file: {e}")
        return None, None, None, 0

    return _process_split(H_complex, config)

def _process_split(H_complex, config):
    if not np.iscomplexobj(H_complex):
        print("Warning: Input data appears to be Real. Treating as Magnitude only?")
    
    print(f"Original Complex Shape: {H_complex.shape}")
    
    # 拆分实部和虚部
    H_real = np.real(H_complex)
    H_imag = np.imag(H_complex)
    
    # 拼接特征: [Batch, Real_Feat + Imag_Feat]
    data_processed = np.hstack([H_real, H_imag])
    
    print(f"Processed Feature Shape (Real+Imag): {data_processed.shape}")
    
    total_len = data_processed.shape[0]
    train_end = int(total_len * config['train_ratio'])
    val_end = int(total_len * (config['train_ratio'] + config['val_ratio']))
    
    train_data = data_processed[:train_end]
    val_data = data_processed[train_end:val_end]
    test_data = data_processed[val_end:]
    
    return train_data, val_data, test_data, H_complex.shape[1]

def generate_samples_from_block_m2o(block_2d, lookback, horizon, step=1):
    num_timesteps, _ = block_2d.shape
    X, y = [], []
    if num_timesteps <= lookback + horizon: return np.array([]), np.array([])
    
    for i in range(0, num_timesteps - lookback - horizon + 1, step):
        X.append(block_2d[i : i+lookback])
        y.append(block_2d[i+lookback + horizon - 1])
    return np.array(X), np.array(y)

def generate_samples_from_block_s2s(block_2d, lookback, horizon, step=1):
    num_timesteps, _ = block_2d.shape
    X, y = [], []
    if num_timesteps <= lookback + horizon: return np.array([]), np.array([])
    
    for i in range(0, num_timesteps - lookback - horizon + 1, step):
        X.append(block_2d[i : i+lookback])
        y.append(block_2d[i+lookback : i+lookback+horizon])
    return np.array(X), np.array(y)

def generate_square_subsequent_mask(dim):
    mask = (torch.triu(torch.ones(dim, dim)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def compute_complex_nmse(y_true, y_pred, num_sc):
    """
    计算复数域的 NMSE
    y_true/y_pred: [Samples, 2*num_sc] (前半部分实部，后半部分虚部)
    """
    # 重建复数
    y_true_c = y_true[:, :num_sc] + 1j * y_true[:, num_sc:]
    y_pred_c = y_pred[:, :num_sc] + 1j * y_pred[:, num_sc:]
    
    error_power = np.sum(np.abs(y_true_c - y_pred_c)**2)
    true_power = np.sum(np.abs(y_true_c)**2)
    
    if true_power == 0: return 0.0
    return 10 * np.log10(error_power / true_power)

def plot_loss_curve(train_losses, val_losses, model_name, save_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss (Norm)')
    plt.plot(val_losses, label='Val Loss (Real MSE)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Training Loss Curve')
    plt.legend()
    plt.grid(True)
    path = os.path.join(save_dir, f'loss_curve_{model_name}.png')
    plt.savefig(path)
    plt.close()
    print(f"Saved loss curve to {path}")

def plot_predictions_complex(y_true, y_pred, model_name, y_naive=None, save_dir='.', num_sc=0):
    sc_idx = 0 
    
    # 绘图时只绘制幅度 (Magnitude in dB) 以便于观察
    mag_true = np.abs(y_true[:, sc_idx] + 1j * y_true[:, sc_idx + num_sc])
    mag_pred = np.abs(y_pred[:, sc_idx] + 1j * y_pred[:, sc_idx + num_sc])
    
    mag_true_db = 20 * np.log10(mag_true + 1e-9)
    mag_pred_db = 20 * np.log10(mag_pred + 1e-9)
    
    plt.figure(figsize=(12, 6))
    limit = 200 
    plt.plot(mag_true_db[:limit], label='Ground Truth', color='black', alpha=0.8)
    plt.plot(mag_pred_db[:limit], label=f'{model_name} Pred', color='red', linestyle='--')
    
    if y_naive is not None:
        mag_naive = np.abs(y_naive[:, sc_idx] + 1j * y_naive[:, sc_idx + num_sc])
        mag_naive_db = 20 * np.log10(mag_naive + 1e-9)
        plt.plot(mag_naive_db[:limit], label='Naive', color='green', linestyle=':', alpha=0.6)
    
    plt.title(f'{model_name} Prediction (SC {sc_idx}) - Magnitude')
    plt.ylabel('dB')
    plt.xlabel('Time Step')
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(save_dir, f'pred_complex_{model_name}.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Prediction plot saved to {save_path}")

# ==============================================================================
# 3. 训练与评估函数
# ==============================================================================

def train(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, 
          optimizer: torch.optim.Optimizer, device: torch.device, horizon: int) -> float:
    model.train()
    epoch_loss = 0.0 
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        if isinstance(model, TimeSeriesTransformer):
            sos = x[:, -1:, :] 
            dec_in = torch.cat([sos, y[:, :-1, :]], dim=1)
            mask = generate_square_subsequent_mask(dec_in.size(1)).to(device)
            out = model(x, dec_in, tgt_mask=mask)
        elif isinstance(model, (ChannelLSTM, ChannelGRU)):
            h = model.init_hidden(x.size(0), device)
            out, _ = model(x, h)
            out = out[:, -horizon:, :]
        else:
            out = model(x)
        
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(train_loader)

def run_inference(model: nn.Module, loader: DataLoader, device: torch.device, horizon: int) -> np.ndarray:
    model.eval()
    all_outputs = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            if isinstance(model, TimeSeriesTransformer):
                curr = x[:, -1:, :]
                for _ in range(horizon):
                    mask = generate_square_subsequent_mask(curr.size(1)).to(device)
                    out = model(x, curr, tgt_mask=mask)
                    next_token = out[:, -1:, :]
                    curr = torch.cat([curr, next_token], dim=1)
                out = curr[:, 1:, :]
            elif isinstance(model, (ChannelLSTM, ChannelGRU)):
                h = model.init_hidden(x.size(0), device)
                _, h = model(x, h)
                dec_in = x[:, -1:, :]
                outs = []
                for _ in range(horizon):
                    o, h = model(dec_in, h)
                    outs.append(o)
                    dec_in = o
                out = torch.cat(outs, dim=1)
            else:
                out = model(x)
            all_outputs.append(out.cpu().numpy())
    return np.concatenate(all_outputs, axis=0)

def predict(model: nn.Module, test_loader: DataLoader, device: torch.device, 
            scaler: StandardScaler, horizon: int) -> Tuple[np.ndarray, float]:
    start_time = time.time()
    predictions_norm = run_inference(model, test_loader, device, horizon)
    total_time = time.time() - start_time
    
    if predictions_norm.ndim == 3: 
        pred_shape = predictions_norm.shape
        predictions_denorm = scaler.inverse_transform(predictions_norm.reshape(-1, pred_shape[-1])).reshape(pred_shape)
        final_preds = predictions_denorm[:, -1, :]
    else:
        final_preds = scaler.inverse_transform(predictions_norm)
    return final_preds, total_time

def evaluate(model: nn.Module, test_loader: DataLoader, device: torch.device, 
             scaler: StandardScaler, horizon: int) -> float:
    preds, _ = predict(model, test_loader, device, scaler, horizon)
    y_true_list = []
    for _, y in test_loader:
        y_np = y.cpu().numpy()
        if y_np.ndim == 3: y_np = y_np[:, -1, :] 
        y_inv = scaler.inverse_transform(y_np)
        y_true_list.append(y_inv)
    y_true = np.concatenate(y_true_list, axis=0)
    return np.mean((preds - y_true)**2)

def epoch_loop(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
               epochs: int, lr_scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau, 
               criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device, 
               scaler: StandardScaler, model_name: str, results_dir: str, 
               horizon: int, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    training_loss, validation_loss = [], []
    best_loss = float('inf')
    epochs_no_improve = 0
    early_stop_patience = 20    

    for epoch in range(epochs):
        start_time = time.time()
        train_loss = train(model, train_loader, criterion, optimizer, device, horizon)
        test_loss = evaluate(model, val_loader, device, scaler, horizon)
        training_loss.append(train_loss)
        validation_loss.append(test_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | "
              f"Val MSE (Real): {test_loss:.6f} | LR: {current_lr:.6f} | "
              f"Time: {time.time()-start_time:.2f}s")

        if test_loss < best_loss:
            best_loss = test_loss
            epochs_no_improve = 0
            model_save_path = os.path.join(results_dir, f'{model_name}_best_model_complex.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f" -> Val Loss improved to {best_loss:.6f}. Saved.")
        else:
            epochs_no_improve += 1

        if current_lr < 1e-6: break
        lr_scheduler.step(test_loss)
        if epochs_no_improve >= early_stop_patience: break
            
    return np.array(training_loss), np.array(validation_loss)

# ==============================================================================
# 4. 主程序
# ==============================================================================

def main():
    # --- 配置 ---
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    project_root = '.'
    data_dir = os.path.join(project_root, 'data')
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Auto-detect
    target_file_path = 'data/ch_est_40m_15mps_autopilot_2d.mat'
    if not os.path.exists(target_file_path):
        available_mats = [f for f in os.listdir(data_dir) if f.endswith('.mat')] if os.path.exists(data_dir) else []
        if available_mats:
            target_file_path = os.path.join(data_dir, available_mats[0])
        else:
            print("Error: No .mat file found.")
            return
    
    # --- CONFIG ---
    CONFIG = {
        'file_path': target_file_path, 
        'mat_var_name': 'H_active_matrix',
        'lookback': 10,       
        'horizons': [1],     
        'ts_step': 1,
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'batch_size': 128,
        'epochs': 100,          
        'learning_rate': 1e-4, 
        'hidden_dim': 128,    
        'num_layers': 2,      
        'dropout': 0.1,
        'nhead': 4,           
        'device': device,
        'learning_rates': {
            'GRU': 0.000025, 'LSTM': 0.0001, 'CNN': 0.0001, 'MLP': 0.0001, 'Transformer': 0.00001
        }
    }
    
    results_file = os.path.join(results_dir, 'results_complex.csv')
    print(f"Data: {CONFIG['file_path']}")

    # --- Load Data ---
    train_raw, val_raw, test_raw, NUM_SC = load_complex_data(CONFIG)
    if train_raw is None: return
    INPUT_DIM = train_raw.shape[1]
    print(f"Features (Real+Imag): {INPUT_DIM}, Subcarriers: {NUM_SC}")

    scaler = StandardScaler()
    train_norm = scaler.fit_transform(train_raw)
    val_norm = scaler.transform(val_raw)
    test_norm = scaler.transform(test_raw)

    performance_df = pd.DataFrame()
    models_to_run = ['Naive Outdated', 'GRU', 'Transformer'] 

    for horizon in CONFIG['horizons']:
        print(f"\n{'='*30}\n Horizon: {horizon} \n{'='*30}")
        
        # Data Generation
        # Seq2Seq (Transformer, LSTM, GRU)
        train_x_s2s, train_y_s2s = generate_samples_from_block_s2s(train_norm, CONFIG['lookback'], horizon, CONFIG['ts_step'])
        val_x_s2s, val_y_s2s = generate_samples_from_block_s2s(val_norm, CONFIG['lookback'], horizon, CONFIG['ts_step'])
        test_x_s2s, test_y_s2s = generate_samples_from_block_s2s(test_norm, CONFIG['lookback'], horizon, CONFIG['ts_step'])
        
        # M2O (Explicitly generated for Naive, CNN, MLP)
        train_x_m2o, train_y_m2o = generate_samples_from_block_m2o(train_norm, CONFIG['lookback'], horizon, CONFIG['ts_step'])
        val_x_m2o, val_y_m2o = generate_samples_from_block_m2o(val_norm, CONFIG['lookback'], horizon, CONFIG['ts_step'])
        test_x_m2o, test_y_m2o = generate_samples_from_block_m2o(test_norm, CONFIG['lookback'], horizon, CONFIG['ts_step'])

        if len(train_x_s2s) == 0:
            print("Not enough data.")
            continue

        # Naive Baseline
        naive_predictions_norm = test_x_m2o[:, -1, :]
        naive_targets_norm = test_y_m2o
        naive_pred = scaler.inverse_transform(naive_predictions_norm)
        naive_true = scaler.inverse_transform(naive_targets_norm)
        
        naive_vis = naive_pred # Define for plotting
        
        naive_nmse = compute_complex_nmse(naive_true, naive_pred, NUM_SC)
        print(f"Naive Baseline NMSE: {naive_nmse:.4f} dB")
        performance_df.loc[horizon, 'Naive Outdated'] = naive_nmse

        for model_name in models_to_run:
            if model_name == 'Naive Outdated': continue
            print(f"--- Training {model_name} ---")
            
            # Dataset Setup
            if model_name in ['GRU', 'LSTM', 'Transformer']:
                train_dataset = TensorDataset(torch.from_numpy(train_x_s2s).float(), torch.from_numpy(train_y_s2s).float())
                val_dataset = TensorDataset(torch.from_numpy(val_x_s2s).float(), torch.from_numpy(val_y_s2s).float())
                test_dataset = TensorDataset(torch.from_numpy(test_x_s2s).float(), torch.from_numpy(test_y_s2s).float())
            else:
                # M2O Dataset (Now correctly using pre-generated variables)
                train_dataset = TensorDataset(torch.from_numpy(train_x_m2o).float(), torch.from_numpy(train_y_m2o).float())
                val_dataset = TensorDataset(torch.from_numpy(val_x_m2o).float(), torch.from_numpy(val_y_m2o).float())
                test_dataset = TensorDataset(torch.from_numpy(test_x_m2o).float(), torch.from_numpy(test_y_m2o).float())

            train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

            lr = CONFIG['learning_rates'].get(model_name, 0.0001)
            model = None
            if model_name == 'Transformer':
                model = TimeSeriesTransformer(
                    input_size=INPUT_DIM, 
                    dim_val=CONFIG['hidden_dim'], 
                    n_encoder_layers=CONFIG['num_layers'], 
                    n_decoder_layers=CONFIG['num_layers'],
                    n_heads=CONFIG['nhead'], 
                    dropout_encoder=CONFIG['dropout'], 
                    dropout_decoder=CONFIG['dropout'], 
                    dropout_pos_enc=CONFIG['dropout'], 
                    dim_feedforward_encoder=512, 
                    dim_feedforward_decoder=512,
                    max_seq_len=CONFIG['lookback']+horizon+100, 
                    dec_seq_len=horizon, 
                    out_seq_len=horizon
                )
            elif model_name == 'LSTM':
                model = ChannelLSTM(INPUT_DIM, CONFIG['hidden_dim'], CONFIG['num_layers'], CONFIG['dropout'], output_dim=INPUT_DIM)
            elif model_name == 'GRU':
                model = ChannelGRU(INPUT_DIM, CONFIG['hidden_dim'], CONFIG['num_layers'], CONFIG['dropout'], output_dim=INPUT_DIM)
            elif model_name == 'CNN':
                model = ChannelCNN(CONFIG['lookback'], INPUT_DIM)
            elif model_name == 'MLP':
                model = ChannelMLP(CONFIG['lookback'], INPUT_DIM)
            
            if model is None: continue
            model.to(device)
            
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

            # Train
            train_losses, val_losses = epoch_loop(
                model, train_loader, val_loader, CONFIG['epochs'], scheduler, criterion, optimizer, device, 
                scaler, model_name, results_dir, horizon, CONFIG['lookback']
            )
            
            plot_loss_curve(train_losses, val_losses, f"{model_name}_H{horizon}", results_dir)

            # Test
            best_path = os.path.join(results_dir, f'{model_name}_best_model_complex.pth')
            if os.path.exists(best_path):
                model.load_state_dict(torch.load(best_path))
                preds, _ = predict(model, test_loader, device, scaler, horizon)
                
                if preds.ndim == 3: preds = preds[:, -1, :] 
                
                # Calculate Test NMSE against Naive Truth (t_final)
                t_final = naive_true
                nmse = compute_complex_nmse(t_final, preds, NUM_SC)
                print(f"Test NMSE ({model_name}): {nmse:.4f} dB")
                performance_df.loc[horizon, model_name] = nmse
                
                # Plot
                plot_predictions_complex(t_final, preds, f"{model_name}_H{horizon}", naive_vis, save_dir=results_dir, num_sc=NUM_SC)

    print(performance_df)
    performance_df.to_csv(results_file)

if __name__ == "__main__":
    main()