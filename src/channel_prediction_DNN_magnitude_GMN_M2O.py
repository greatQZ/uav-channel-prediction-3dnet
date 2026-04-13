# -*- coding: utf-8 -*-
"""
DNN Channel Prediction (Magnitude Version): A comprehensive script for training
and evaluating various deep learning models for predicting wireless channel magnitude.

This version has been refactored to use a unified Many-to-One (M2O) framework
for all models to ensure fair comparison. The Transformer model uses a dedicated
Encoder-Only architecture suitable for M2O tasks.
"""

import os
import time
import math
import scipy.io
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
import random
from typing import Tuple

# ==============================================================================
# 1. 模型定义 (Model Definitions)
# ==============================================================================

class PositionalEncoder(nn.Module):
    def __init__(self, dropout: float = 0.1, max_seq_len: int = 5000, d_model: int = 512):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)

class TransformerEncoderNet(nn.Module):
    """ Transformer Encoder-Only model for Many-to-One tasks. """
    def __init__(self, input_size: int, d_model: int, n_layers: int, n_heads: int, 
                 dim_feedforward: int, dropout: float, lookback: int):
        super().__init__()
        self.encoder_input_layer = nn.Linear(input_size, d_model)
        self.positional_encoding_layer = PositionalEncoder(d_model=d_model, dropout=dropout, max_seq_len=lookback)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.linear_mapping = nn.Linear(d_model, input_size)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # src shape: (batch, lookback, input_size)
        src = self.encoder_input_layer(src) # -> (batch, lookback, d_model)
        src = self.positional_encoding_layer(src)
        
        encoder_output = self.encoder(src) # -> (batch, lookback, d_model)
        
        # We only use the output of the last timestep for prediction
        last_step_output = encoder_output[:, -1, :] # -> (batch, d_model)
        
        prediction = self.linear_mapping(last_step_output) # -> (batch, input_size)
        
        return prediction

class GRUNet_Seq2Seq(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int, drop_prob: float = 0.2):
        super(GRUNet_Seq2Seq, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out, h = self.gru(x, h)
        # For M2O, we apply the FC layer to the entire sequence, 
        # but will only use the last step's output in the training loop.
        out = self.fc(out)
        return out, h

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden

class LSTMNet_Seq2Seq(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int, drop_prob: float = 0.2):
        super(LSTMNet_Seq2Seq, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, h: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        out, h = self.lstm(x, h)
        # For M2O, we apply the FC layer to the entire sequence, 
        # but will only use the last step's output in the training loop.
        out = self.fc(out)
        return out, h

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden

class CNNNet(nn.Module):
    def __init__(self, lookback: int, input_channels: int):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=128, kernel_size=2)
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # Dynamic calculation of the flattened size
        with torch.no_grad():
            dummy_input = torch.randn(1, input_channels, lookback)
            dummy_output = self.pool(F.relu(self.conv1(dummy_input)))
            conv_output_size = dummy_output.numel()
            
        self.fc1 = nn.Linear(conv_output_size, 128)
        self.fc2 = nn.Linear(128, input_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1) # (batch, features, lookback)
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MLPNet(nn.Module):
    def __init__(self, lookback: int, input_channels: int):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(lookback * input_channels, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, input_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ==============================================================================
# 2. 数据处理函数 (Data Processing Functions)
# ==============================================================================

def generate_samples_from_block_m2o(block_2d: np.ndarray, lookback: int, horizon: int, step: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generates Many-to-One samples from a continuous data block."""
    num_timesteps, num_features = block_2d.shape
    X, y = [], []
    
    required_len = (lookback - 1) * step + horizon
    num_samples = num_timesteps - required_len
    
    if num_samples <= 0:
        return np.array([]), np.array([])
        
    for i in range(num_samples):
        input_end_idx = i + (lookback - 1) * step
        label_idx = input_end_idx + horizon
        indices = np.arange(i, input_end_idx + 1, step)
        
        X.append(block_2d[indices])
        y.append(block_2d[label_idx])
        
    return np.array(X), np.array(y)

# ==============================================================================
# 3. 训练与评估函数 (Training & Evaluation Functions)
# ==============================================================================

def train(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, 
          optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    running_loss = 0.0
    for x, y, _ in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        if isinstance(model, (GRUNet_Seq2Seq, LSTMNet_Seq2Seq)):
            h = model.init_hidden(x.size(0), device)
            out, h = model(x.float(), h)
            # Take the last time step's output for M2O prediction
            out = out[:, -1, :]
        else:
            out = model(x.float())

        loss = criterion(out, y.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item()
        
    return running_loss / len(train_loader)

def evaluate(model: nn.Module, test_loader: DataLoader, device: torch.device, 
             scaler: StandardScaler) -> float:
    model.eval()
    total_loss_real = 0
    with torch.no_grad():
        for i, (x, y, y_clean) in enumerate(test_loader):
            x, y, y_clean = x.to(device), y.to(device), y_clean.to(device)

            if isinstance(model, (GRUNet_Seq2Seq, LSTMNet_Seq2Seq)):
                h = model.init_hidden(x.size(0), device)
                out, h = model(x.float(), h)
                out = out[:, -1, :]
            else:
                out = model(x.float())

            # De-normalize for real loss calculation
            out_denorm = scaler.inverse_transform(out.cpu().numpy())
            y_clean_denorm = y_clean.cpu().numpy()

            loss_real = np.mean((out_denorm - y_clean_denorm)**2)
            total_loss_real += loss_real * x.size(0)
            
    return total_loss_real / len(test_loader.dataset)

def predict(model: nn.Module, test_loader: DataLoader, device: torch.device, 
            scaler: StandardScaler) -> Tuple[np.ndarray, float]:
    model.eval()
    all_predictions = []
    start_time = time.time()
    
    with torch.no_grad():
        for x, y, _ in test_loader:
            x = x.to(device)

            if isinstance(model, (GRUNet_Seq2Seq, LSTMNet_Seq2Seq)):
                h = model.init_hidden(x.size(0), device)
                out, h = model(x.float(), h)
                out = out[:, -1, :]
            else:
                out = model(x.float())

            out_denorm = scaler.inverse_transform(out.cpu().numpy())
            all_predictions.append(out_denorm)

    end_time = time.time()
    total_time = end_time - start_time

    return np.concatenate(all_predictions, axis=0), total_time

def epoch_loop(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, 
               epochs: int, lr_scheduler, criterion: nn.Module, optimizer: torch.optim.Optimizer, 
               device: torch.device, scaler: StandardScaler, model_name: str, results_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    training_loss, validation_loss = [], []
    best_loss = float('inf')
    epochs_no_improve = 0
    early_stop_patience = 20

    for epoch in range(epochs):
        start_time = time.time()
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_loss = evaluate(model, test_loader, device, scaler)

        training_loss.append(train_loss)
        validation_loss.append(test_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} | Train Loss (Norm.): {train_loss:.8f} | "
              f"Test Loss (Real): {test_loss:.6f} | LR: {current_lr:.6f} | "
              f"Time: {time.time()-start_time:.2f}s")

        if test_loss < best_loss:
            best_loss = test_loss
            epochs_no_improve = 0
            
            model_save_path = os.path.join(results_dir, f'{model_name}_best_model_magnitude.pth')
            torch.save(model.state_dict(), model_save_path)
            
            print(f" -> Validation loss improved to {best_loss:.6f}. Saving {model_name} model to {model_save_path}")
        else:
            epochs_no_improve += 1
            
        if current_lr < 1e-6:
            print("Learning rate too small, stopping training")
            break
            
        lr_scheduler.step(test_loss)
        
        if epochs_no_improve >= early_stop_patience:
            print(f"Early stopping triggered after {epoch+1} epochs!")
            break

    # Save the scaler along with the best model
    scaler_save_path = os.path.join(results_dir, f'{model_name}_scaler_magnitude.gz')
    joblib.dump(scaler, scaler_save_path)

    return np.array(training_loss), np.array(validation_loss)

def plot_losses(training_loss: np.ndarray, validation_loss: np.ndarray, model_name: str):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(training_loss, label='Training Loss (Normalized)', color='blue')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Normalized MSE Loss', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    ax2 = ax1.twinx()
    ax2.plot(validation_loss, label='Validation Loss (Real Scale)', color='orange')
    ax2.set_ylabel('Real Scale MSE Loss', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    
    plt.title(f'Loss Curve for {model_name} (Magnitude Prediction)')
    fig.tight_layout()
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    
    plt.grid(True)
    plt.show()

def plot_predictions(true_vals: np.ndarray, preds: np.ndarray, model_name: str, 
                     naive_preds: np.ndarray, num_timesteps: int = 150, subcarrier_idx: int = 0):
    plt.figure(figsize=(15, 7))
    
    min_length = min(len(true_vals), len(preds), len(naive_preds))
    true_vals = true_vals[:min_length]
    preds = preds[:min_length]
    naive_preds = naive_preds[:min_length]
    
    time_steps = np.arange(min(min_length, num_timesteps))
    
    if subcarrier_idx >= true_vals.shape[1]:
        print(f"Warning: Subcarrier index {subcarrier_idx} is out of bounds. Using subcarrier 0.")
        subcarrier_idx = 0
    
    plt.plot(time_steps, true_vals[:num_timesteps, subcarrier_idx], 
             label='True Magnitude', color='blue', linewidth=2)
    plt.plot(time_steps, preds[:num_timesteps, subcarrier_idx], 
             label=f'{model_name} Prediction', color='red', linestyle='--')
    
    plt.plot(time_steps, naive_preds[:num_timesteps, subcarrier_idx], 
             label='Naive Prediction', color='green', linestyle=':')
    
    plt.title(f'Subcarrier #{subcarrier_idx} Magnitude Prediction (First {min(min_length, num_timesteps)} timesteps)')
    plt.xlabel('Time Step')
    plt.ylabel('Channel Magnitude')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ==============================================================================
# 4. 主执行流程 (Main Execution Flow)
# ==============================================================================

def main():
    # --- 4.1. Parameter Configuration ---
    print("--- 1. Configuring Experiment Parameters ---")
    project_root = '.'
    data_dir = os.path.join(project_root, 'data')
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    combined_data_file = os.path.join(data_dir, 'my_H_clean_combined_padded_optimized_test_autopilot_40m_for_training.mat')
    results_file = os.path.join(results_dir, 'performance_results_magnitude_m2o.csv')
    
    if not os.path.exists(combined_data_file):
        raise FileNotFoundError(f"Data file not found: {combined_data_file}")

    print(f"Data file path: {combined_data_file}")
    print(f"Results file path: {results_file}")

    NUM_ACTIVE_SUBCARRIERS = 576
    horizons = [1]
    lookback = 5
    ts_step = 1
    epochs = 200 # Reduced for faster runs, can be increased
    batch_size = 512
    learning_rates = {'GRU': 0.0001, 'LSTM': 0.0005, 'CNN': 0.0005, 'MLP': 0.0005, 'Transformer': 0.0001}

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device set to: {device}")

    # --- 4.2. Data Loading and Processing ---
    print("\n--- 2. Loading and Processing Data ---")
    mat = scipy.io.loadmat(combined_data_file)
    complex_data = mat['H'].flatten().reshape(-1, NUM_ACTIVE_SUBCARRIERS, 2).astype(np.float32)
    H_2d_matrix = np.sqrt(complex_data[:, :, 0]**2 + complex_data[:, :, 1]**2)

    is_zero_row = ~H_2d_matrix.any(axis=1)
    block_change_indices = np.where(np.diff(is_zero_row))[0] + 1
    data_blocks_2d = np.split(H_2d_matrix, block_change_indices)
    continuous_blocks = [block for block in data_blocks_2d if not np.all(block == 0)]
    print(f"Successfully recovered {len(continuous_blocks)} continuous data blocks.")

    # --- 4.3. Experiment Loop ---
    print("\n--- 3. Starting Experiment Loop ---")
    performance_df = pd.DataFrame()
    if os.path.exists(results_file):
        performance_df = pd.read_csv(results_file, index_col=0)

    #models_to_run = ['Naive Outdated', 'GRU', 'LSTM', 'CNN', 'MLP', 'Transformer']
    models_to_run = ['Naive Outdated', 'GRU', 'Transformer']

    for horizon in horizons:
        print(f"\n{'='*25} Processing Horizon = {horizon} {'='*25}")

        split_idx = int(len(continuous_blocks) * 0.8)
        train_blocks, test_blocks = continuous_blocks[:split_idx], continuous_blocks[split_idx:]

        print("Preparing data for the unified Many-to-One framework...")
        train_x_list, train_y_list = [], []
        for block in train_blocks:
            x, y = generate_samples_from_block_m2o(block, lookback, horizon, ts_step)
            if x.size > 0:
                train_x_list.append(x)
                train_y_list.append(y)
        
        test_x_list, test_y_list = [], []
        for block in test_blocks:
            x, y = generate_samples_from_block_m2o(block, lookback, horizon, ts_step)
            if x.size > 0:
                test_x_list.append(x)
                test_y_list.append(y)

        train_x, train_y = np.concatenate(train_x_list), np.concatenate(train_y_list)
        test_x, test_y = np.concatenate(test_x_list), np.concatenate(test_y_list)
        
        # Fit scaler ONLY on training data
        scaler = StandardScaler().fit(train_x.reshape(-1, NUM_ACTIVE_SUBCARRIERS))

        # Normalize data
        train_x_norm = scaler.transform(train_x.reshape(-1, NUM_ACTIVE_SUBCARRIERS)).reshape(train_x.shape)
        train_y_norm = scaler.transform(train_y)
        test_x_norm = scaler.transform(test_x.reshape(-1, NUM_ACTIVE_SUBCARRIERS)).reshape(test_x.shape)
        test_y_norm = scaler.transform(test_y)

        # Create DataLoaders once for all models
        train_dataset = TensorDataset(torch.from_numpy(train_x_norm), torch.from_numpy(train_y_norm), torch.from_numpy(train_y))
        test_dataset = TensorDataset(torch.from_numpy(test_x_norm), torch.from_numpy(test_y_norm), torch.from_numpy(test_y))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        print("Data preparation complete.")

        # --- Naive Baseline ---
        naive_predictions = test_x[:, -1, :]
        naive_mse = np.mean((naive_predictions - test_y)**2)
        print(f"\n--- Naive Outdated Baseline MSE: {naive_mse:.6f} ---")
        performance_df.loc[horizon, 'Naive Outdated'] = naive_mse
        performance_df.to_csv(results_file)
        
        # --- DNN Models ---
        for model_name in models_to_run:
            if model_name == 'Naive Outdated':
                continue

            print(f"\n--- Processing model: {model_name} ---")
            
            if model_name == 'GRU':
                model = GRUNet_Seq2Seq(input_dim=NUM_ACTIVE_SUBCARRIERS, hidden_dim=256, output_dim=NUM_ACTIVE_SUBCARRIERS, n_layers=2)
            elif model_name == 'LSTM':
                model = LSTMNet_Seq2Seq(input_dim=NUM_ACTIVE_SUBCARRIERS, hidden_dim=256, output_dim=NUM_ACTIVE_SUBCARRIERS, n_layers=2)
            elif model_name == 'CNN':
                model = CNNNet(lookback, input_channels=NUM_ACTIVE_SUBCARRIERS)
            elif model_name == 'MLP':
                model = MLPNet(lookback, input_channels=NUM_ACTIVE_SUBCARRIERS)
            elif model_name == 'Transformer':
                model = TransformerEncoderNet(
                    input_size=NUM_ACTIVE_SUBCARRIERS,
                    d_model=256,
                    n_layers=2,
                    n_heads=8,
                    dim_feedforward=512,
                    dropout=0.1,
                    lookback=lookback
                )

            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rates[model_name])
            criterion = nn.MSELoss()
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

            training_loss, validation_loss = epoch_loop(
                model, train_loader, test_loader, epochs, lr_scheduler, 
                criterion, optimizer, device, scaler, model_name, results_dir
            )

            if len(validation_loss) > 0:
                final_loss = min(validation_loss)
                performance_df.loc[horizon, model_name] = final_loss
                print(f" -> {model_name} training complete. Best Validation MSE: {final_loss:.6f}")

            print(f" -> Evaluating inference latency for {model_name}...")
            # Load the best performing model for final prediction and plotting
            model.load_state_dict(torch.load(os.path.join(results_dir, f'{model_name}_best_model_magnitude.pth')))
            
            preds, total_pred_time = predict(model, test_loader, device, scaler)
            
            num_samples = len(test_loader.dataset)
            avg_time = total_pred_time / num_samples
            print(f" -> Total inference time for {num_samples} samples: {total_pred_time:.4f} seconds")
            print(f" -> Average single-step prediction time: {avg_time * 1000:.4f} ms / {avg_time * 1e6:.2f} µs")

            plot_losses(training_loss, validation_loss, model_name)
            plot_predictions(test_y, preds, model_name, naive_predictions, subcarrier_idx=0)
            
            performance_df.to_csv(results_file)

    print("\n--- 5. Experiment Complete ---")
    print("Final Performance Comparison (Magnitude MSE):")
    print(performance_df)

if __name__ == '__main__':
    main()