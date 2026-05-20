"""
train_pi_stgcn.py

Physics-Informed Spatio-Temporal Graph Convolutional Network
for multi-step river discharge forecasting.

Model name:
    PI-STGCN

Purpose:
    - Forecast river discharge at multiple cascade stations.
    - Use graph convolution to represent spatial station connectivity.
    - Use temporal convolution to learn time-dependent flow patterns.
    - Add a physics-informed cascade loss based on upstream-downstream routing.

Expected input CSV:
    ../../Data/Aligned_River_Cascade_Data.csv

Required columns:
    Date, Lepoglava, Zeljeznica, Kljuc, Tuhovec, Ludbreg

Outputs:
    - Best model checkpoint
    - Training history CSV
    - Test-set predictions CSV
    - Full-dataset lead-1 predictions CSV
"""

import os
import random
import argparse
from dataclasses import dataclass
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib


# ============================================================
# 1. Configuration
# ============================================================

@dataclass
class Config:
    # Paths
    data_path: str = "../../Data/Aligned_River_Cascade_Data.csv"
    results_dir: str = "../../Results/PI_STGCN/"
    models_dir: str = "../../Models/PI_STGCN/"

    # River stations in upstream-to-downstream order
    stations: Tuple[str, ...] = (
        "Lepoglava",
        "Zeljeznica",
        "Kljuc",
        "Tuhovec",
        "Ludbreg",
    )

    # Forecast setup
    lookback_days: int = 14
    forecast_horizon: int = 14

    # Dataset split ratios
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Training setup
    batch_size: int = 256
    epochs: int = 800
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 80
    seed: int = 42

    # Model size
    hidden_channels: int = 64
    temporal_kernel_size: int = 3
    dropout: float = 0.20

    # Loss weights
    data_loss_weight: float = 1.0
    physics_loss_weight: float = 0.10

    # Hardware
    num_workers: int = 0


# ============================================================
# 2. Utility Functions
# ============================================================

def set_seed(seed: int) -> None:
    """
    Make training as reproducible as possible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic behavior can be slower but is cleaner for experiments.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_directories(cfg: Config) -> None:
    """
    Create output directories.
    """
    os.makedirs(cfg.results_dir, exist_ok=True)
    os.makedirs(cfg.models_dir, exist_ok=True)


def get_device() -> torch.device:
    """
    Select CUDA if available, otherwise CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_cascade_adjacency(num_nodes: int) -> torch.Tensor:
    """
    Build an adjacency matrix for a river cascade.

    The graph is treated as undirected for neural message passing:
        station_i <-> station_{i+1}

    The physical direction is still handled separately in the
    physics-informed loss.
    """
    adjacency = torch.eye(num_nodes)

    for i in range(num_nodes - 1):
        adjacency[i, i + 1] = 1.0
        adjacency[i + 1, i] = 1.0

    return adjacency


def normalize_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    """
    Symmetric adjacency normalization:

        A_hat = D^(-1/2) A D^(-1/2)

    This keeps graph convolution numerically stable.
    """
    degree = adjacency.sum(dim=1)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0

    degree_matrix_inv_sqrt = torch.diag(degree_inv_sqrt)
    normalized_adjacency = degree_matrix_inv_sqrt @ adjacency @ degree_matrix_inv_sqrt

    return normalized_adjacency


def load_river_data(cfg: Config) -> pd.DataFrame:
    """
    Load and clean river discharge data.
    """
    df = pd.read_csv(cfg.data_path, parse_dates=["Date"], index_col="Date")

    missing_cols = [col for col in cfg.stations if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required station columns: {missing_cols}")

    df = df[list(cfg.stations)].copy()

    # Time interpolation is suitable because the index is datetime.
    df = df.interpolate(method="time").bfill().ffill()

    return df


# ============================================================
# 3. Dataset Preparation
# ============================================================

class RiverForecastDataset(Dataset):
    """
    Converts a continuous time series into supervised samples.

    Input:
        x = past lookback_days values
        shape: [lookback_days, num_stations]

    Target:
        y = next forecast_horizon values
        shape: [forecast_horizon, num_stations]
    """

    def __init__(
        self,
        data_array: np.ndarray,
        lookback_days: int,
        forecast_horizon: int,
    ):
        self.x, self.y = self._create_sequences(
            data_array,
            lookback_days,
            forecast_horizon,
        )

    @staticmethod
    def _create_sequences(
        data_array: np.ndarray,
        lookback_days: int,
        forecast_horizon: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        xs = []
        ys = []

        max_start = len(data_array) - lookback_days - forecast_horizon + 1

        for i in range(max_start):
            x_i = data_array[i : i + lookback_days]
            y_i = data_array[i + lookback_days : i + lookback_days + forecast_horizon]

            xs.append(x_i)
            ys.append(y_i)

        xs = torch.tensor(np.array(xs), dtype=torch.float32)
        ys = torch.tensor(np.array(ys), dtype=torch.float32)

        return xs, ys

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def chronological_split(
    data: np.ndarray,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split time series chronologically.

    No shuffling is used because this is time-series forecasting.
    """
    n = len(data)

    train_end = int(n * cfg.train_ratio)
    val_end = int(n * (cfg.train_ratio + cfg.val_ratio))

    train_data = data[:train_end]
    val_data = data[train_end - cfg.lookback_days - cfg.forecast_horizon + 1 : val_end]
    test_data = data[val_end - cfg.lookback_days - cfg.forecast_horizon + 1 :]

    return train_data, val_data, test_data


def prepare_dataloaders(
    df: pd.DataFrame,
    cfg: Config,
) -> Tuple[DataLoader, DataLoader, DataLoader, MinMaxScaler]:
    """
    Scale data using only the training split, then create dataloaders.
    """
    raw_data = df.values.astype(np.float32)

    train_raw, val_raw, test_raw = chronological_split(raw_data, cfg)

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_raw)
    val_scaled = scaler.transform(val_raw)
    test_scaled = scaler.transform(test_raw)

    train_dataset = RiverForecastDataset(
        train_scaled,
        cfg.lookback_days,
        cfg.forecast_horizon,
    )

    val_dataset = RiverForecastDataset(
        val_scaled,
        cfg.lookback_days,
        cfg.forecast_horizon,
    )

    test_dataset = RiverForecastDataset(
        test_scaled,
        cfg.lookback_days,
        cfg.forecast_horizon,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, scaler


# ============================================================
# 4. PI-STGCN Model
# ============================================================

class GraphConvolution(nn.Module):
    """
    Basic graph convolution layer.

    Input shape:
        [batch, time, nodes, in_channels]

    Output shape:
        [batch, time, nodes, out_channels]
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # Message passing over nodes.
        # adjacency: [nodes, nodes]
        # x: [batch, time, nodes, channels]
        x = torch.einsum("ij,btjc->btic", adjacency, x)

        # Feature transformation.
        x = self.linear(x)

        return x


class STGCNBlock(nn.Module):
    """
    Spatio-temporal graph convolution block.

    Sequence:
        1. Graph convolution
        2. Temporal convolution
        3. Activation
        4. Dropout
        5. Residual connection
    """

    def __init__(
        self,
        channels: int,
        temporal_kernel_size: int,
        dropout: float,
    ):
        super().__init__()

        self.graph_conv = GraphConvolution(channels, channels)

        padding = temporal_kernel_size // 2

        self.temporal_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(temporal_kernel_size, 1),
            padding=(padding, 0),
        )

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        residual = x

        # Graph convolution.
        x = self.graph_conv(x, adjacency)

        # Conv2D expects [batch, channels, time, nodes].
        x = x.permute(0, 3, 1, 2)
        x = self.temporal_conv(x)

        # Return to [batch, time, nodes, channels].
        x = x.permute(0, 2, 3, 1)

        x = self.activation(x)
        x = self.dropout(x)

        # Residual connection.
        x = x + residual

        # Normalize over feature channels.
        x = self.norm(x)

        return x


class PISTGCN(nn.Module):
    """
    Physics-Informed Spatio-Temporal GCN.

    Input:
        [batch, lookback_days, num_stations]

    Output:
        predictions:
            [batch, forecast_horizon, num_stations]

        routing coefficients:
            c1, c2, c3 for each river reach

        dynamic baseflow:
            [batch, num_reaches]
    """

    def __init__(
        self,
        num_nodes: int,
        lookback_days: int,
        forecast_horizon: int,
        hidden_channels: int,
        temporal_kernel_size: int,
        dropout: float,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.num_reaches = num_nodes - 1
        self.lookback_days = lookback_days
        self.forecast_horizon = forecast_horizon

        # Input discharge value becomes one feature channel.
        self.input_projection = nn.Linear(1, hidden_channels)

        self.block_1 = STGCNBlock(
            channels=hidden_channels,
            temporal_kernel_size=temporal_kernel_size,
            dropout=dropout,
        )

        self.block_2 = STGCNBlock(
            channels=hidden_channels,
            temporal_kernel_size=temporal_kernel_size,
            dropout=dropout,
        )

        self.block_3 = STGCNBlock(
            channels=hidden_channels,
            temporal_kernel_size=temporal_kernel_size,
            dropout=dropout,
        )

        # Forecast head maps the final temporal representation to horizon values.
        self.forecast_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, forecast_horizon),
        )

        # Learnable Muskingum-style routing coefficients.
        # c1 + c2 + c3 = 1 for each reach.
        self.raw_c3 = nn.Parameter(torch.zeros(self.num_reaches))
        self.raw_c12_split = nn.Parameter(torch.zeros(self.num_reaches))

        # Dynamic baseflow is estimated from graph-level hidden features.
        self.baseflow_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, self.num_reaches),
            nn.ReLU(),
        )

    def get_routing_coefficients(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Produce physically constrained routing coefficients.

        c3 is restricted to a smaller range to improve stability.
        c1 and c2 split the remaining mass.
        """
        c3 = 0.5 * torch.sigmoid(self.raw_c3)
        remaining = 1.0 - c3

        split = torch.sigmoid(self.raw_c12_split)

        c1 = remaining * split
        c2 = remaining * (1.0 - split)

        return c1, c2, c3

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:

        # x: [batch, time, nodes]
        x = x.unsqueeze(-1)  # [batch, time, nodes, 1]

        x = self.input_projection(x)

        x = self.block_1(x, adjacency)
        x = self.block_2(x, adjacency)
        x = self.block_3(x, adjacency)

        # Use the last hidden state for each station.
        last_hidden = x[:, -1, :, :]  # [batch, nodes, hidden_channels]

        # Forecast per node.
        pred = self.forecast_head(last_hidden)  # [batch, nodes, horizon]
        pred = pred.permute(0, 2, 1)            # [batch, horizon, nodes]

        # Graph-level feature for dynamic baseflow.
        graph_hidden = last_hidden.mean(dim=1)  # [batch, hidden_channels]
        baseflow = self.baseflow_head(graph_hidden)

        coeffs = self.get_routing_coefficients()

        return pred, coeffs, baseflow


# ============================================================
# 5. Physics-Informed Loss
# ============================================================

def physics_cascade_loss(
    predictions: torch.Tensor,
    input_sequence: torch.Tensor,
    coeffs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    baseflow: torch.Tensor,
) -> torch.Tensor:
    """
    Physics-informed routing consistency loss.

    For each river reach:
        upstream station i
        downstream station i + 1

    The downstream prediction should be consistent with routed upstream flow,
    previous upstream flow, previous downstream flow, and dynamic baseflow.

    All values are in normalized space.
    """
    c1, c2, c3 = coeffs

    batch_size, horizon, num_nodes = predictions.shape
    num_reaches = num_nodes - 1

    losses = []

    for reach in range(num_reaches):
        upstream = reach
        downstream = reach + 1

        for h in range(horizon):
            q_up_current = predictions[:, h, upstream]

            if h == 0:
                q_up_previous = input_sequence[:, -1, upstream]
                q_down_previous = input_sequence[:, -1, downstream]
            else:
                q_up_previous = predictions[:, h - 1, upstream]
                q_down_previous = predictions[:, h - 1, downstream]

            routed_downstream = (
                c1[reach] * q_up_current
                + c2[reach] * q_up_previous
                + c3[reach] * q_down_previous
                + baseflow[:, reach]
            )

            q_down_predicted = predictions[:, h, downstream]

            losses.append(torch.mean((q_down_predicted - routed_downstream) ** 2))

    return torch.stack(losses).mean()


def combined_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    input_sequence: torch.Tensor,
    coeffs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    baseflow: torch.Tensor,
    cfg: Config,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Total loss = data loss + physics-informed loss.
    """
    data_loss = nn.MSELoss()(predictions, targets)

    phys_loss = physics_cascade_loss(
        predictions=predictions,
        input_sequence=input_sequence,
        coeffs=coeffs,
        baseflow=baseflow,
    )

    total_loss = (
        cfg.data_loss_weight * data_loss
        + cfg.physics_loss_weight * phys_loss
    )

    return total_loss, data_loss, phys_loss


# ============================================================
# 6. Evaluation Metrics
# ============================================================

def nse_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Nash-Sutcliffe Efficiency.
    """
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)

    if denominator == 0:
        return np.nan

    return 1.0 - numerator / denominator


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    station_names: Tuple[str, ...],
) -> pd.DataFrame:
    """
    Compute metrics for each station using rescaled discharge values.
    """
    rows = []

    for i, station in enumerate(station_names):
        true_i = y_true[:, :, i].reshape(-1)
        pred_i = y_pred[:, :, i].reshape(-1)

        rmse = np.sqrt(mean_squared_error(true_i, pred_i))
        mae = mean_absolute_error(true_i, pred_i)
        r2 = r2_score(true_i, pred_i)
        nse = nse_score(true_i, pred_i)

        rows.append(
            {
                "Station": station,
                "RMSE": rmse,
                "MAE": mae,
                "R2": r2,
                "NSE": nse,
            }
        )

    return pd.DataFrame(rows)


def inverse_transform_3d(
    data_3d: np.ndarray,
    scaler: MinMaxScaler,
) -> np.ndarray:
    """
    Inverse transform data with shape:
        [samples, horizon, stations]
    """
    samples, horizon, stations = data_3d.shape

    data_2d = data_3d.reshape(-1, stations)
    inv_2d = scaler.inverse_transform(data_2d)

    return inv_2d.reshape(samples, horizon, stations)


# ============================================================
# 7. Training and Validation
# ============================================================

def run_one_epoch(
    model: PISTGCN,
    loader: DataLoader,
    adjacency: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    cfg: Config,
    device: torch.device,
    train: bool,
) -> Dict[str, float]:
    """
    Run one training or validation epoch.
    """
    if train:
        model.train()
    else:
        model.eval()

    total_losses = []
    data_losses = []
    phys_losses = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            predictions, coeffs, baseflow = model(x_batch, adjacency)

            loss, data_loss, phys_loss = combined_loss(
                predictions=predictions,
                targets=y_batch,
                input_sequence=x_batch,
                coeffs=coeffs,
                baseflow=baseflow,
                cfg=cfg,
            )

            if train:
                loss.backward()

                # Gradient clipping improves stability in long training.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

        total_losses.append(loss.item())
        data_losses.append(data_loss.item())
        phys_losses.append(phys_loss.item())

    return {
        "total_loss": float(np.mean(total_losses)),
        "data_loss": float(np.mean(data_losses)),
        "physics_loss": float(np.mean(phys_losses)),
    }


def train_model(
    model: PISTGCN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    adjacency: torch.Tensor,
    cfg: Config,
    device: torch.device,
) -> pd.DataFrame:
    """
    Main training loop with early stopping.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=25,
    )

    best_val_loss = np.inf
    patience_counter = 0

    history = []

    best_model_path = os.path.join(cfg.models_dir, "best_pi_stgcn.pt")

    for epoch in range(1, cfg.epochs + 1):
        train_stats = run_one_epoch(
            model=model,
            loader=train_loader,
            adjacency=adjacency,
            optimizer=optimizer,
            cfg=cfg,
            device=device,
            train=True,
        )

        val_stats = run_one_epoch(
            model=model,
            loader=val_loader,
            adjacency=adjacency,
            optimizer=optimizer,
            cfg=cfg,
            device=device,
            train=False,
        )

        scheduler.step(val_stats["total_loss"])

        row = {
            "epoch": epoch,
            "train_total_loss": train_stats["total_loss"],
            "train_data_loss": train_stats["data_loss"],
            "train_physics_loss": train_stats["physics_loss"],
            "val_total_loss": val_stats["total_loss"],
            "val_data_loss": val_stats["data_loss"],
            "val_physics_loss": val_stats["physics_loss"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }

        history.append(row)

        improved = val_stats["total_loss"] < best_val_loss

        if improved:
            best_val_loss = val_stats["total_loss"]
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg.__dict__,
                    "best_val_loss": best_val_loss,
                },
                best_model_path,
            )
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 10 == 0 or improved:
            print(
                f"Epoch {epoch:04d} | "
                f"Train Loss: {train_stats['total_loss']:.6f} | "
                f"Val Loss: {val_stats['total_loss']:.6f} | "
                f"Data: {val_stats['data_loss']:.6f} | "
                f"Physics: {val_stats['physics_loss']:.6f}"
            )

        if patience_counter >= cfg.patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    history_df = pd.DataFrame(history)

    history_path = os.path.join(cfg.results_dir, "training_history_pi_stgcn.csv")
    history_df.to_csv(history_path, index=False)

    print(f"Training history saved to: {history_path}")
    print(f"Best model saved to: {best_model_path}")

    return history_df


# ============================================================
# 8. Prediction and Saving Results
# ============================================================

def collect_predictions(
    model: PISTGCN,
    loader: DataLoader,
    adjacency: torch.Tensor,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect model predictions and targets from a dataloader.
    """
    model.eval()

    preds = []
    targets = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)

            pred_batch, _, _ = model(x_batch, adjacency)

            preds.append(pred_batch.cpu().numpy())
            targets.append(y_batch.numpy())

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    return preds, targets


def save_test_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cfg: Config,
) -> None:
    """
    Save test predictions in long format.
    """
    rows = []

    samples, horizon, stations = y_true.shape

    for sample_idx in range(samples):
        for lead in range(horizon):
            for station_idx, station in enumerate(cfg.stations):
                rows.append(
                    {
                        "Sample": sample_idx,
                        "Lead_Day": lead + 1,
                        "Station": station,
                        "Actual": y_true[sample_idx, lead, station_idx],
                        "PI_STGCN_Prediction": y_pred[sample_idx, lead, station_idx],
                    }
                )

    out_df = pd.DataFrame(rows)

    out_path = os.path.join(cfg.results_dir, "test_predictions_pi_stgcn.csv")
    out_df.to_csv(out_path, index=False)

    print(f"Test predictions saved to: {out_path}")


def save_full_dataset_lead1_predictions(
    model: PISTGCN,
    df: pd.DataFrame,
    scaler: MinMaxScaler,
    adjacency: torch.Tensor,
    cfg: Config,
    device: torch.device,
) -> None:
    """
    Generate continuous lead-1 predictions for the full dataset.

    This is useful for plotting a continuous predicted hydrograph.
    """
    model.eval()

    raw_data = df.values.astype(np.float32)
    scaled_data = scaler.transform(raw_data)

    xs = []
    dates = []

    for i in range(len(scaled_data) - cfg.lookback_days):
        xs.append(scaled_data[i : i + cfg.lookback_days])
        dates.append(df.index[i + cfg.lookback_days])

    x_tensor = torch.tensor(np.array(xs), dtype=torch.float32).to(device)

    lead1_preds = []

    with torch.no_grad():
        for i in range(0, len(x_tensor), cfg.batch_size):
            batch = x_tensor[i : i + cfg.batch_size]
            pred_batch, _, _ = model(batch, adjacency)

            # Lead-1 prediction.
            lead1 = pred_batch[:, 0, :]
            lead1_preds.append(lead1.cpu().numpy())

    lead1_preds = np.concatenate(lead1_preds, axis=0)
    lead1_preds_rescaled = scaler.inverse_transform(lead1_preds)

    actuals = raw_data[cfg.lookback_days:]

    results_df = pd.DataFrame({"Date": dates})

    for i, station in enumerate(cfg.stations):
        results_df[f"{station}_Actual"] = actuals[:, i]
        results_df[f"{station}_PI_STGCN_Pred"] = lead1_preds_rescaled[:, i]

    out_path = os.path.join(cfg.results_dir, "full_dataset_lead1_pi_stgcn.csv")
    results_df.to_csv(out_path, index=False)

    print(f"Full-dataset lead-1 predictions saved to: {out_path}")


# ============================================================
# 9. Main Script
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PI-STGCN for river discharge forecasting.")

    parser.add_argument("--data_path", type=str, default=Config.data_path)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch_size", type=int, default=Config.batch_size)
    parser.add_argument("--lr", type=float, default=Config.learning_rate)
    parser.add_argument("--physics_weight", type=float, default=Config.physics_loss_weight)
    parser.add_argument("--hidden_channels", type=int, default=Config.hidden_channels)
    parser.add_argument("--lookback", type=int, default=Config.lookback_days)
    parser.add_argument("--horizon", type=int, default=Config.forecast_horizon)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = Config(
        data_path=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        physics_loss_weight=args.physics_weight,
        hidden_channels=args.hidden_channels,
        lookback_days=args.lookback,
        forecast_horizon=args.horizon,
    )

    set_seed(cfg.seed)
    create_directories(cfg)

    device = get_device()
    print(f"Using device: {device}")

    # -------------------------------
    # Load data
    # -------------------------------
    df = load_river_data(cfg)

    train_loader, val_loader, test_loader, scaler = prepare_dataloaders(df, cfg)

    scaler_path = os.path.join(cfg.models_dir, "scaler_pi_stgcn.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")

    # -------------------------------
    # Build graph
    # -------------------------------
    num_nodes = len(cfg.stations)

    adjacency = build_cascade_adjacency(num_nodes)
    adjacency = normalize_adjacency(adjacency)
    adjacency = adjacency.to(device)

    # -------------------------------
    # Build model
    # -------------------------------
    model = PISTGCN(
        num_nodes=num_nodes,
        lookback_days=cfg.lookback_days,
        forecast_horizon=cfg.forecast_horizon,
        hidden_channels=cfg.hidden_channels,
        temporal_kernel_size=cfg.temporal_kernel_size,
        dropout=cfg.dropout,
    ).to(device)

    print(model)

    # -------------------------------
    # Train
    # -------------------------------
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        adjacency=adjacency,
        cfg=cfg,
        device=device,
    )

    # -------------------------------
    # Load best model
    # -------------------------------
    best_model_path = os.path.join(cfg.models_dir, "best_pi_stgcn.pt")

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Loaded best model from: {best_model_path}")

    # -------------------------------
    # Test evaluation
    # -------------------------------
    test_preds_scaled, test_targets_scaled = collect_predictions(
        model=model,
        loader=test_loader,
        adjacency=adjacency,
        device=device,
    )

    test_preds = inverse_transform_3d(test_preds_scaled, scaler)
    test_targets = inverse_transform_3d(test_targets_scaled, scaler)

    metrics_df = compute_metrics(
        y_true=test_targets,
        y_pred=test_preds,
        station_names=cfg.stations,
    )

    metrics_path = os.path.join(cfg.results_dir, "test_metrics_pi_stgcn.csv")
    metrics_df.to_csv(metrics_path, index=False)

    print("\nTest Metrics:")
    print(metrics_df)
    print(f"Test metrics saved to: {metrics_path}")

    save_test_predictions(
        y_true=test_targets,
        y_pred=test_preds,
        cfg=cfg,
    )

    save_full_dataset_lead1_predictions(
        model=model,
        df=df,
        scaler=scaler,
        adjacency=adjacency,
        cfg=cfg,
        device=device,
    )

    # -------------------------------
    # Print learned routing coefficients
    # -------------------------------
    c1, c2, c3 = model.get_routing_coefficients()

    coeff_df = pd.DataFrame(
        {
            "Reach": [
                f"{cfg.stations[i]} -> {cfg.stations[i + 1]}"
                for i in range(len(cfg.stations) - 1)
            ],
            "c1": c1.detach().cpu().numpy(),
            "c2": c2.detach().cpu().numpy(),
            "c3": c3.detach().cpu().numpy(),
            "sum": (
                c1.detach().cpu().numpy()
                + c2.detach().cpu().numpy()
                + c3.detach().cpu().numpy()
            ),
        }
    )

    coeff_path = os.path.join(cfg.results_dir, "learned_routing_coefficients_pi_stgcn.csv")
    coeff_df.to_csv(coeff_path, index=False)

    print("\nLearned Routing Coefficients:")
    print(coeff_df)
    print(f"Routing coefficients saved to: {coeff_path}")


if __name__ == "__main__":
    main()
