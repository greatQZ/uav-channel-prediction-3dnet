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
from typing import Tuple, Optional

# ==============================================================================
# 1. 模型定义 (保持修复版)
# ==============================================================================

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, input_size, dim_val, n_encoder_layers, n_decoder_layers, n_heads, dropout_encoder, dropout_decoder, dropout_pos_enc, dim_feedforward_encoder, dim_feedforward_decoder, max_seq_len, dec_seq_len, out_seq_len):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoder(dim_val, dropout=dropout_pos_enc)
        self.encoder_layer = nn.Linear(input_size, dim_val)
        self.decoder_layer = nn.Linear(input_size, dim_val)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=dim_val, nhead=n_heads, dim_feedforward=dim_feedforward_encoder, dropout=dropout_encoder)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_encoder_layers)
        
        decoder_layers = nn.TransformerDecoderLayer(d_model=dim_val, nhead=n_heads, dim_feedforward=dim_feedforward_decoder, dropout=dropout_decoder)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, n_decoder_layers)
        
        self.output_layer = nn.Linear(dim_val, input_size)
        self.src_mask = None

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt):
        if self.src_mask is None or self.src_mask.size(0) != len(tgt):
            mask = self._generate_square_subsequent_mask(len(tgt)).to(tgt.device)
            self.src_mask = mask

        src = self.encoder_layer(src)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        
        tgt = self.decoder_layer(tgt)
        tgt = self.pos_encoder(tgt)
        output = self.transformer_decoder(tgt, memory, tgt_mask=self.src_mask)
        return self.output_layer(output)

class ChannelGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, output_dim):
        super(ChannelGRU, self).__init__()
        self.model_type = 'RNN'
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim) 

    def forward(self, x):
        out, _ = self.gru(x)
        last_out = out[:, -1, :] 
        return self.fc(last_out)

class ChannelLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, output_dim):
        super(ChannelLSTM, self).__init__()
        self.model_type = 'RNN'
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        return self.fc(last_out)

class ChannelCNN(nn.Module):
    def __init__(self, lookback, input_channels, hidden_dim, output_dim):
        super(ChannelCNN, self).__init__()
        self.model_type = 'CNN'
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = x.permute(0, 2, 1) 
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

class ChannelMLP(nn.Module):
    def __init__(self, lookback, input_channels, output_dim):
        super(ChannelMLP, self).__init__()
        self.model_type = 'MLP'
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_channels * lookback, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# ==============================================================================
# 2. 数据处理与工具函数
# ==============================================================================

def load_amplitude_data(file_path):
    print(f"\n--- Loading Data from {file_path} ---")
    if not os.path.exists(file_path):
        print("Error: File not found.")
        return None
    try:
        mat = scipy.io.loadmat(file_path)
        if 'H_active_matrix' in mat: H_complex = mat['H_active_matrix']
        elif 'H' in mat: H_complex = mat['H']
        else: 
            key = max(mat.keys(), key=lambda k: mat[k].size if isinstance(mat[k], np.ndarray) else 0)
            H_complex = mat[key]
        return np.abs(H_complex)
    except Exception as e:
        print(f"Error loading MAT: {e}")
        return None

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

def compute_magnitude_nmse(y_true, y_pred):
    error_power = np.sum((y_true - y_pred)**2)
    true_power = np.sum(y_true**2)
    if true_power == 0: return 0.0
    return 10 * np.log10(error_power / true_power)

def ask_to_save(save_path):
    print(f"图片已生成。请查看弹出的窗口...")
    try:
        plt.show() 
    except Exception as e:
        print(f"无法显示窗口: {e}")
    
    while True:
        choice = input(f"是否保存图片到 {save_path}? (y/n): ").lower()
        if choice == 'y':
            plt.savefig(save_path, dpi=300)
            print(f"已保存: {save_path}")
            break
        elif choice == 'n':
            print("已跳过保存。")
            break
        else:
            print("请输入 y 或 n")
    plt.close()

def plot_loss_curve(train_losses, val_losses, model_name, horizon, save_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title(f'{model_name} (H={horizon}) Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    path = os.path.join(save_dir, f'loss_{model_name}_H{horizon}.png')
    ask_to_save(path)

def plot_all_models_comparison(target_true, naive_pred, predictions_dict, horizon, save_dir, sc_idx=0):
    plt.figure(figsize=(15, 7))
    limit = 200 
    
    y_true_db = 20 * np.log10(target_true[:limit, sc_idx] + 1e-9)
    plt.plot(y_true_db, 'k-', label='Ground Truth', linewidth=2, alpha=0.8)
    
    y_naive_db = 20 * np.log10(naive_pred[:limit, sc_idx] + 1e-9)
    plt.plot(y_naive_db, 'g:', label='Naive Baseline', linewidth=2, alpha=0.6)
    
    colors = ['r', 'b', 'orange', 'purple', 'cyan']
    for i, (name, pred) in enumerate(predictions_dict.items()):
        y_pred_db = 20 * np.log10(pred[:limit, sc_idx] + 1e-9)
        plt.plot(y_pred_db, linestyle='--', color=colors[i%len(colors)], label=f'{name}', linewidth=1.5)
    
    plt.title(f"Comparison (Horizon={horizon}, SC={sc_idx})")
    plt.ylabel("Magnitude (dB)")
    plt.xlabel("Time Step")
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(save_dir, f"comparison_H{horizon}.png")
    ask_to_save(save_path)

# ==============================================================================
# 3. 训练与评估流程
# ==============================================================================

def train_step(model, loader, criterion, optimizer, device, horizon, feature_dim):
    model.train()
    total_loss = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        
        if hasattr(model, 'model_type') and model.model_type == 'Transformer':
            sos = x[:, -1:, :] 
            if horizon == 1:
                dec_in = sos
            else:
                dec_in = torch.cat([sos, y[:, :-1, :]], dim=1)
            output = model(x, dec_in) 
            loss = criterion(output, y)
        else:
            output = model(x)
            if isinstance(output, tuple): output = output[0]
            output = output.view(-1, horizon, feature_dim)
            loss = criterion(output, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def predict(model, loader, device, scaler, horizon, feature_dim):
    model.eval()
    preds_list = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            
            if hasattr(model, 'model_type') and model.model_type == 'Transformer':
                curr_dec = x[:, -1:, :] 
                for _ in range(horizon):
                    tgt_mask = generate_square_subsequent_mask(curr_dec.size(1), device)
                    output = model(x, curr_dec)
                    step_pred = output[:, -1:, :]
                    curr_dec = torch.cat([curr_dec, step_pred], dim=1)
                final_pred = curr_dec[:, 1:, :] 
                preds_list.append(final_pred.cpu().numpy())
            else:
                output = model(x)
                if isinstance(output, tuple): output = output[0]
                output = output.view(-1, horizon, feature_dim)
                preds_list.append(output.cpu().numpy())
                
    preds_norm = np.concatenate(preds_list, axis=0)
    B, H, F = preds_norm.shape
    preds_denorm = scaler.inverse_transform(preds_norm.reshape(-1, F)).reshape(B, H, F)
    # 返回预测序列的最后一步
    return preds_denorm[:, -1, :]

# ==============================================================================
# 4. 主程序 (无 CONFIG 字典，变量直连)
# ==============================================================================

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    project_root = '.'
    data_dir = os.path.join(project_root, 'data')
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # --- 文件路径 ---
    target_file_path = 'data/ch_est_40m_15mps_autopilot_2d.mat'
    if not os.path.exists(target_file_path):
         files = [f for f in os.listdir(data_dir) if f.endswith('.mat')]
         if files: target_file_path = os.path.join(data_dir, files[0])
    print(f"Data: {target_file_path}")

    # --- 核心参数 (回归原始设置) ---
    LOOKBACK = 10
    HORIZONS = [1, 10]
    BATCH_SIZE = 512
    EPOCHS = 500
    HIDDEN_DIM = 512
    NUM_LAYERS = 2
    DROPOUT = 0.1
    NHEAD = 16
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TS_STEP = 1
    
    # 1. Load Data
    raw_data = load_amplitude_data(target_file_path)
    if raw_data is None: return
    INPUT_DIM = raw_data.shape[1]
    print(f"Features (Magnitude): {INPUT_DIM}")
    
    # 2. Scale
    scaler = StandardScaler()
    data_norm = scaler.fit_transform(raw_data)
    
    total_len = len(data_norm)
    train_end = int(total_len * TRAIN_RATIO)
    val_end = int(total_len * (TRAIN_RATIO + VAL_RATIO))
    train_raw = data_norm[:train_end]
    val_raw = data_norm[train_end:val_end]
    test_raw = data_norm[val_end:]
    
    results_summary = {}

    for horizon in HORIZONS:
        print(f"\n{'='*40}\n Processing Horizon = {horizon} \n{'='*40}")
        
        # Dataset
        X_train, y_train = generate_windows(train_raw, LOOKBACK, horizon, TS_STEP)
        X_val, y_val = generate_windows(val_raw, LOOKBACK, horizon, TS_STEP)
        X_test, y_test = generate_windows(test_raw, LOOKBACK, horizon, TS_STEP)
        
        if len(X_train) == 0: continue
        
        train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)), batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)), batch_size=BATCH_SIZE, shuffle=False)
        
        # Naive Baseline
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, INPUT_DIM)).reshape(-1, horizon, INPUT_DIM)
        X_test_last = scaler.inverse_transform(X_test[:, -1, :])
        naive_pred = np.tile(X_test_last[:, np.newaxis, :], (1, horizon, 1))
        
        target_H_last = y_test_inv[:, -1, :] 
        naive_H_last = naive_pred[:, -1, :]
        
        nmse_naive = compute_magnitude_nmse(target_H_last, naive_H_last)
        print(f"Naive Baseline NMSE: {nmse_naive:.4f} dB")
        results_summary[f'Naive_H{horizon}'] = nmse_naive
        
        # Models
        models_to_run = ['GRU', 'Transformer']
        model_preds = {} 
        
        for name in models_to_run:
            print(f"\n--- Training {name} ---")
            
            flat_out = horizon * INPUT_DIM
            
            # 学习率逻辑
            if name == 'Transformer':
                lr = 0.00001
            else:
                lr = 0.0001
            
            if name == 'Transformer':
                model = TransformerModel(INPUT_DIM, HIDDEN_DIM, NHEAD, NUM_LAYERS, 
                                         NHEAD, DROPOUT, DROPOUT, DROPOUT, 
                                         1024, 1024, LOOKBACK+horizon+50, horizon, horizon).to(device)
            elif name == 'GRU':
                model = ChannelGRU(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, flat_out).to(device)
            elif name == 'LSTM':
                model = ChannelLSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, flat_out).to(device)
            elif name == 'CNN':
                model = ChannelCNN(LOOKBACK, INPUT_DIM, HIDDEN_DIM, flat_out).to(device)
            elif name == 'MLP':
                model = ChannelMLP(LOOKBACK, INPUT_DIM, flat_out).to(device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
            criterion = nn.MSELoss()
            
            best_val_loss = float('inf')
            patience = 15 
            counter = 0
            train_losses, val_losses = [], []
            
            for epoch in range(EPOCHS):
                train_loss = train_step(model, train_loader, criterion, optimizer, device, horizon, INPUT_DIM)
                
                model.eval()
                val_loss_sum = 0
                with torch.no_grad():
                    for vx, vy in val_loader:
                        vx, vy = vx.to(device)
                        vy = vy.to(device)
                        if hasattr(model, 'model_type') and model.model_type == 'Transformer':
                            sos = vx[:, -1:, :]
                            dec_in = sos if horizon==1 else torch.cat([sos, vy[:, :-1, :]], dim=1)
                            out = model(vx, dec_in) 
                            loss = criterion(out, vy)
                        else:
                            out = model(vx)
                            if isinstance(out, tuple): out = out[0]
                            out = out.view(-1, horizon, INPUT_DIM)
                            loss = criterion(out, vy)
                        val_loss_sum += loss.item()
                val_loss = val_loss_sum / len(val_loader)
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                
                curr_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{EPOCHS} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {curr_lr:.8f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    counter = 0
                    torch.save(model.state_dict(), f'best_{name}_H{horizon}.pth')
                else:
                    counter += 1
                    if counter >= patience:
                        print("Early stopping.")
                        break
                scheduler.step(val_loss)
            
            plot_loss_curve(train_losses, val_losses, name, horizon, results_dir)
            
            model.load_state_dict(torch.load(f'best_{name}_H{horizon}.pth'))
            preds_inv = predict(model, test_loader, device, scaler, horizon, INPUT_DIM)
            
            pred_H_last = preds_inv[:, -1, :]
            model_preds[name] = pred_H_last
            
            nmse = compute_magnitude_nmse(target_H_last, pred_H_last)
            print(f"{name} Test NMSE: {nmse:.4f} dB (Gain: {nmse_naive - nmse:.4f} dB)")
            results_summary[f'{name}_H{horizon}'] = nmse

        plot_all_models_comparison(target_H_last, naive_H_last, model_preds, horizon, os.path.join(results_dir, f"comparison_H{horizon}.png"))

    print("\nFinal Results Summary:")
    print(results_summary)
    pd.DataFrame([results_summary]).to_csv(os.path.join(results_dir, 'results_final.csv'))

if __name__ == "__main__":
    main()