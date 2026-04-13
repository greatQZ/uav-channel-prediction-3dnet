# -*- coding: utf-8 -*-
import os
import time
import math
import numpy as np
import scipy.io
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
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
        return self.linear_mapping(decoder_output)

class ChannelLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float = 0.1, output_dim: int = None):
        super(ChannelLSTM, self).__init__()
        if output_dim is None: output_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def init_hidden(self, batch_size: int, device: torch.device):
        weight = next(self.parameters()).data
        return (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))

    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        out, hidden = self.lstm(x, hidden)
        return self.fc(out), hidden

class ChannelGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float = 0.1, output_dim: int = None):
        super(ChannelGRU, self).__init__()
        if output_dim is None: output_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def init_hidden(self, batch_size: int, device: torch.device):
        weight = next(self.parameters()).data
        return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)

    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        out, hidden = self.gru(x, hidden)
        return self.fc(out), hidden

class ChannelCNN(nn.Module):
    def __init__(self, lookback: int, input_channels: int, hidden_dim: int = 128, dropout: float = 0.1):
        super(ChannelCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(hidden_dim, input_channels)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, Seq, Feat] -> [Batch, Feat, Seq]
        x = x.permute(0, 2, 1) 
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

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

# ==============================================================================
# 2. 数据处理 (去旋转)
# ==============================================================================

def load_data_derotated(config):
    print(f"\n--- Loading Data from {config['file_path']} ---")
    if not os.path.exists(config['file_path']):
        print("Error: File not found.")
        return None, None, None, None, 0, 0.0

    try:
        mat = scipy.io.loadmat(config['file_path'])
        if config['mat_var_name'] not in mat:
            var_name = max(mat.keys(), key=lambda k: mat[k].size if isinstance(mat[k], np.ndarray) else 0)
            H_complex = mat[var_name]
        else:
            H_complex = mat[config['mat_var_name']]
    except Exception as e:
        print(f"Error loading MAT file: {e}")
        return None, None, None, None, 0, 0.0

    # A. 计算相位旋转速度
    phase = np.angle(H_complex)
    diff_complex = np.exp(1j * phase[1:]) * np.conj(np.exp(1j * phase[:-1]))
    phase_diff = np.angle(diff_complex)
    avg_slope = np.mean(phase_diff)
    print(f"Detected Phase Slope: {avg_slope:.6f} rad/step")

    # B. 去旋转
    num_timesteps = H_complex.shape[0]
    t_indices = np.arange(num_timesteps).reshape(-1, 1)
    derotation_factor = np.exp(-1j * avg_slope * t_indices)
    H_aligned = H_complex * derotation_factor
    print("Data derotated. Phase should now be stable around 0.")

    # C. 提取特征 [Real, Imag]
    data_processed = np.hstack([np.real(H_aligned), np.imag(H_aligned)])
    
    total_len = data_processed.shape[0]
    train_end = int(total_len * config['train_ratio'])
    val_end = int(total_len * (config['train_ratio'] + config['val_ratio']))
    
    train_data = data_processed[:train_end]
    val_data = data_processed[train_end:val_end]
    test_data = data_processed[val_end:]
    
    test_complex_raw = H_complex[val_end:]
    
    return train_data, val_data, test_data, test_complex_raw, H_complex.shape[1], avg_slope

def generate_windows(data, lookback, horizon, step=1):
    n_samples = len(data) - lookback - horizon + 1
    if n_samples <= 0: return np.array([]), np.array([])
    X, y = [], []
    for i in range(0, n_samples, step):
        X.append(data[i : i+lookback])
        y.append(data[i+lookback : i+lookback+horizon])
    return np.array(X), np.array(y)

def generate_square_subsequent_mask(dim, device):
    mask = (torch.triu(torch.ones(dim, dim)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask.to(device)

# ==============================================================================
# 3. 预测与复原 (包含内存修复)
# ==============================================================================

def predict_and_rerotate(model, loader, device, scaler, test_complex_raw, lookback, horizon, num_sc, avg_slope, test_start_idx):
    model.eval()
    preds_list = []
    
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            if isinstance(model, TimeSeriesTransformer):
                curr_dec_in = x[:, -1:, :] 
                for _ in range(horizon):
                    tgt_mask = generate_square_subsequent_mask(curr_dec_in.size(1), device)
                    out = model(x, curr_dec_in, tgt_mask=tgt_mask)
                    next_step = out[:, -1:, :]
                    if curr_dec_in.size(1) < horizon:
                         curr_dec_in = torch.cat([curr_dec_in, next_step], dim=1)
                pred = out 
            elif isinstance(model, (ChannelLSTM, ChannelGRU)):
                h = model.init_hidden(x.size(0), device)
                out, _ = model(x, h)
                if horizon == 1:
                    pred = out[:, -1:, :] 
                else:
                    pred = out[:, -horizon:, :]
            else:
                pred = model(x)
                if pred.ndim == 2: pred = pred.unsqueeze(1)
            preds_list.append(pred.cpu().numpy())
            
    preds_norm = np.concatenate(preds_list, axis=0)
    
    # 反归一化
    B, H, F = preds_norm.shape
    preds_denorm = scaler.inverse_transform(preds_norm.reshape(-1, F)).reshape(B, H, F)
    
    # 重建复数
    pred_real = preds_denorm[:, :, :num_sc]
    pred_imag = preds_denorm[:, :, num_sc:]
    H_pred_aligned = pred_real + 1j * pred_imag
    
    # 复原旋转
    n_samples = len(H_pred_aligned)
    idx_matrix = np.arange(n_samples)[:, None] + lookback + np.arange(horizon)[None, :]
    t_abs = test_start_idx + idx_matrix
    
    rerotation_phasor = np.exp(1j * avg_slope * t_abs)
    # --- 内存修复: 增加维度 ---
    rerotation_phasor = rerotation_phasor[:, :, None]
    
    H_pred_final = H_pred_aligned * rerotation_phasor
    
    return H_pred_final, idx_matrix

def compute_nmse_complex(y_true, y_pred):
    error_power = np.sum(np.abs(y_true - y_pred)**2)
    true_power = np.sum(np.abs(y_true)**2)
    if true_power == 0: return 0.0
    return 10 * np.log10(error_power / true_power)

# ==============================================================================
# 4. 主程序
# ==============================================================================

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    project_root = '.'
    data_dir = os.path.join(project_root, 'data')
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    target_file_path = 'data/ch_est_40m_15mps_autopilot_2d.mat'
    if not os.path.exists(target_file_path):
         files = [f for f in os.listdir(data_dir) if f.endswith('.mat')]
         if files: target_file_path = os.path.join(data_dir, files[0])

    CONFIG = {
        'file_path': target_file_path, 
        'mat_var_name': 'H_active_matrix',
        'lookback': 10,       
        'horizons': [1],     
        'ts_step': 1,
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'batch_size': 128,
        'epochs': 50,          
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

    # 1. 加载去旋转后的数据
    train_raw, val_raw, test_raw, test_complex_raw, NUM_SC, avg_slope = load_data_derotated(CONFIG)
    if train_raw is None: return
    
    test_start_idx = train_raw.shape[0] + val_raw.shape[0]
    INPUT_DIM = train_raw.shape[1] 
    print(f"Input Dim: {INPUT_DIM} (Real+Imag, Aligned)")

    scaler = StandardScaler()
    train_norm = scaler.fit_transform(train_raw)
    val_norm = scaler.transform(val_raw)
    test_norm = scaler.transform(test_raw)

    results_dict = {}

    for horizon in CONFIG['horizons']:
        print(f"\n{'='*30}\n Horizon: {horizon} \n{'='*30}")
        
        # 生成窗口
        train_x, train_y = generate_windows(train_norm, CONFIG['lookback'], horizon)
        val_x, val_y = generate_windows(val_norm, CONFIG['lookback'], horizon)
        test_x, test_y = generate_windows(test_norm, CONFIG['lookback'], horizon)

        train_loader = DataLoader(TensorDataset(torch.Tensor(train_x), torch.Tensor(train_y)), 
                                  batch_size=CONFIG['batch_size'], shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.Tensor(val_x), torch.Tensor(val_y)), 
                                batch_size=CONFIG['batch_size'], shuffle=False)
        test_loader = DataLoader(TensorDataset(torch.Tensor(test_x), torch.Tensor(test_y)), 
                                 batch_size=CONFIG['batch_size'], shuffle=False)

        # Naive Baseline
        n_test = len(test_y)
        idx_matrix = np.arange(n_test)[:, None] + CONFIG['lookback'] + np.arange(horizon)[None, :]
        valid_mask = idx_matrix < len(test_complex_raw)
        idx_matrix = idx_matrix[np.all(valid_mask, axis=1)]
        
        # Naive: H[t+H] = H[t] (使用原始数据，即去旋转前的)
        naive_input_idx = idx_matrix - horizon
        naive_pred = test_complex_raw[naive_input_idx]
        naive_true = test_complex_raw[idx_matrix]
        
        nmse_naive = compute_nmse_complex(naive_true, naive_pred)
        print(f"Naive Baseline NMSE: {nmse_naive:.4f} dB")
        results_dict[f'Naive_H{horizon}'] = nmse_naive

        # 运行所有模型
        models_to_run = ['GRU', 'Transformer']
        
        for name in models_to_run:
            print(f"\n--- Training {name} ---")
            
            lr = CONFIG['learning_rates'].get(name, CONFIG['learning_rate'])
            model = None
            
            if name == 'GRU':
                model = ChannelGRU(INPUT_DIM, CONFIG['hidden_dim'], CONFIG['num_layers'], CONFIG['dropout'], output_dim=INPUT_DIM)
            elif name == 'LSTM':
                model = ChannelLSTM(INPUT_DIM, CONFIG['hidden_dim'], CONFIG['num_layers'], CONFIG['dropout'], output_dim=INPUT_DIM)
            elif name == 'CNN':
                model = ChannelCNN(CONFIG['lookback'], INPUT_DIM)
            elif name == 'MLP':
                model = ChannelMLP(CONFIG['lookback'], INPUT_DIM)
            elif name == 'Transformer':
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
                    max_seq_len=CONFIG['lookback'] + horizon + 50,
                    dec_seq_len=horizon,
                    out_seq_len=horizon
                )
            
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
            criterion = nn.MSELoss()
            
            best_val_loss = float('inf')
            epochs_no_improve = 0
            
            for epoch in range(CONFIG['epochs']):
                # Train Loop
                model.train()
                loss_sum = 0
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    if y.ndim == 2: y = y.unsqueeze(1)
                    optimizer.zero_grad()
                    
                    if isinstance(model, TimeSeriesTransformer):
                        sos = x[:, -1:, :]
                        dec_in = sos if horizon==1 else torch.cat([sos, y[:, :-1, :]], dim=1)
                        tgt_mask = generate_square_subsequent_mask(dec_in.size(1), device)
                        out = model(x, dec_in, tgt_mask=tgt_mask)
                        loss = criterion(out, y)
                    elif isinstance(model, (ChannelLSTM, ChannelGRU)):
                         h = model.init_hidden(x.size(0), device)
                         out, _ = model(x, h)
                         if horizon == 1:
                             loss = criterion(out[:, -1, :].unsqueeze(1), y)
                         else:
                             loss = criterion(out[:, -horizon:, :], y)
                    else:
                        out = model(x)
                        loss = criterion(out.unsqueeze(1), y if horizon==1 else y[:, -1:, :])

                    loss.backward()
                    optimizer.step()
                    loss_sum += loss.item()
                
                avg_train_loss = loss_sum / len(train_loader)

                # Validation Loop
                model.eval()
                val_loss_sum = 0
                with torch.no_grad():
                     for vx, vy in val_loader:
                        vx, vy = vx.to(device), vy.to(device)
                        if vy.ndim == 2: vy = vy.unsqueeze(1)
                        
                        if isinstance(model, TimeSeriesTransformer):
                            sos = vx[:, -1:, :]
                            dec_in = sos if horizon==1 else torch.cat([sos, vy[:, :-1, :]], dim=1)
                            tgt_mask = generate_square_subsequent_mask(dec_in.size(1), device)
                            out = model(vx, dec_in, tgt_mask=tgt_mask)
                            val_loss = criterion(out, vy)
                        elif isinstance(model, (ChannelLSTM, ChannelGRU)):
                             h = model.init_hidden(vx.size(0), device)
                             out, _ = model(vx, h)
                             val_loss = criterion(out[:, -1, :].unsqueeze(1), vy) if horizon==1 else criterion(out[:, -horizon:, :], vy)
                        else:
                            out = model(vx)
                            val_loss = criterion(out.unsqueeze(1), vy) if horizon==1 else criterion(out.unsqueeze(1), vy[:, -1:, :])
                        val_loss_sum += val_loss.item()
                
                avg_val_loss = val_loss_sum / len(val_loader)

                if (epoch+1) % 10 == 0:
                    print(f"Epoch {epoch+1} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                    torch.save(model.state_dict(), os.path.join(results_dir, f'{name}_H{horizon}_best.pth'))
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= 15:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
                scheduler.step(avg_val_loss)
            
            # Predict & Evaluate
            model.load_state_dict(torch.load(os.path.join(results_dir, f'{name}_H{horizon}_best.pth')))
            H_pred_final, idx_matrix_pred = predict_and_rerotate(
                model, test_loader, device, scaler, test_complex_raw, 
                CONFIG['lookback'], horizon, NUM_SC, avg_slope, test_start_idx
            )
            
            valid_mask_pred = idx_matrix_pred < len(test_complex_raw)
            valid_rows = np.all(valid_mask_pred, axis=1)
            H_pred_flat = H_pred_final[valid_rows].flatten()
            H_true_flat = test_complex_raw[idx_matrix_pred[valid_rows]].flatten()
            
            nmse = compute_nmse_complex(H_true_flat, H_pred_flat)
            print(f"{name} Test NMSE: {nmse:.4f} dB")
            results_dict[f'{name}_H{horizon}'] = nmse

    print("\nSummary (Complex NMSE):")
    print(results_dict)
    pd.DataFrame([results_dict]).to_csv(os.path.join(results_dir, 'results_final.csv'))

if __name__ == "__main__":
    main()