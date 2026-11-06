"""Core models and evaluation utilities for reviewer experiment E7."""

from __future__ import annotations

import json
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import matthews_corrcoef, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


SPLIT_MODES = ("stratified", "disjoint_molecule", "disjoint_aptamer")
# LightGBM is excluded from this equal-budget DL rerun; use existing
# OneHot+MorganFP screening numbers if a classical baseline row is needed.
MODEL_NAMES = ("TwoTowerMLP", "CNNDualEncoder", "CrossAttention")
SEARCH_FOLD_COUNT = 3
FINAL_FOLD_COUNT = 5


def set_global_seed(seed: int) -> None:
    """Seed all libraries used by E7 and request deterministic torch kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:  # pragma: no cover - older torch
        torch.use_deterministic_algorithms(True)


def load_released_splits(splits_dir: str | Path, split_mode: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load and validate the five released folds in their published order."""
    if split_mode not in SPLIT_MODES:
        raise ValueError(f"Unknown split mode: {split_mode}")
    payload = json.loads((Path(splits_dir) / f"{split_mode}.json").read_text(encoding="utf-8"))
    if len(payload) != FINAL_FOLD_COUNT:
        raise ValueError(f"{split_mode} must contain exactly {FINAL_FOLD_COUNT} folds")
    folds = []
    for expected_fold, record in enumerate(payload):
        if int(record.get("fold", expected_fold)) != expected_fold:
            raise ValueError(f"Unexpected fold order in {split_mode}")
        train_idx = np.asarray(record["train_idx"], dtype=int)
        val_idx = np.asarray(record["val_idx"], dtype=int)
        if np.intersect1d(train_idx, val_idx).size:
            raise ValueError(f"Index leakage in {split_mode} fold {expected_fold}")
        folds.append((train_idx, val_idx))
    return folds


def select_top_regression_combinations(results: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    """Select the lowest-RMSE baseline combinations with stable tie-breaking."""
    required = {
        "split",
        "aptamer_encoder",
        "molecule_encoder",
        "RMSE mean",
        "R2 mean",
    }
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"Regression baseline is missing columns: {sorted(missing)}")
    rows = []
    for split_mode in SPLIT_MODES:
        candidates = results.loc[results["split"] == split_mode].copy()
        if len(candidates) < top_k:
            raise ValueError(f"Need at least {top_k} candidates for {split_mode}")
        candidates = candidates.sort_values(
            ["RMSE mean", "aptamer_encoder", "molecule_encoder"],
            kind="mergesort",
        ).head(top_k)
        candidates["baseline_rank"] = np.arange(1, len(candidates) + 1)
        rows.append(candidates)
    return pd.concat(rows, ignore_index=True)


def _mlp_layers(input_dim: int, hidden_dim: int, depth: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_dim
    for index in range(depth):
        output = max(hidden_dim // (2**index), 32)
        layers.extend((nn.Linear(current, output), nn.BatchNorm1d(output), nn.ReLU(), nn.Dropout(dropout)))
        current = output
    return nn.Sequential(*layers)


class TwoTowerMLP(nn.Module):
    """Feature-based two-tower MLP with a configurable fusion head."""

    def __init__(self, apt_dim: int, mol_dim: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.apt_tower = _mlp_layers(apt_dim, hidden_dim, n_layers, dropout)
        self.mol_tower = _mlp_layers(mol_dim, hidden_dim, n_layers, dropout)
        tower_out = max(hidden_dim // (2 ** (n_layers - 1)), 32)
        self.fusion = nn.Sequential(
            _mlp_layers(tower_out * 2, hidden_dim, n_layers, dropout),
            nn.Linear(tower_out, 1),
        )

    def forward(self, aptamer: torch.Tensor, molecule: torch.Tensor) -> torch.Tensor:
        return self.fusion(torch.cat((self.apt_tower(aptamer), self.mol_tower(molecule)), dim=1)).squeeze(-1)


class CNNTop(nn.Module):
    """One-dimensional convolutional top with global max pooling."""

    def __init__(self, hidden_dim: int, output_dim: int, kernel_size: int, dropout: float):
        super().__init__()
        kernels = tuple(k for k in (3, kernel_size, 7) if k <= kernel_size or k == 3)
        kernels = tuple(dict.fromkeys(kernels))
        widths = [output_dim // len(kernels)] * len(kernels)
        widths[-1] += output_dim - sum(widths)
        self.convs = nn.ModuleList(
            nn.Conv1d(hidden_dim, width, kernel, padding=kernel // 2)
            for kernel, width in zip(kernels, widths)
        )
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        channels_first = sequence.transpose(1, 2)
        pooled = [F.relu(conv(channels_first)).amax(dim=2) for conv in self.convs]
        return self.dropout(self.norm(torch.cat(pooled, dim=1)))


class CNNDualEncoder(nn.Module):
    """Dual pseudo-sequence CNN encoder corresponding to TwoTower_CNN_CNN."""

    def __init__(
        self,
        apt_dim: int,
        mol_dim: int,
        hidden_dim: int,
        n_layers: int,
        dropout: float,
        seq_len: int,
        kernel_size: int,
    ):
        super().__init__()
        projection_dim = seq_len * hidden_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.apt_projection = nn.Sequential(nn.Linear(apt_dim, projection_dim), nn.LayerNorm(projection_dim))
        self.mol_projection = nn.Sequential(nn.Linear(mol_dim, projection_dim), nn.LayerNorm(projection_dim))
        self.apt_top = CNNTop(hidden_dim, hidden_dim, kernel_size, dropout)
        self.mol_top = CNNTop(hidden_dim, hidden_dim, kernel_size, dropout)
        self.fusion = nn.Sequential(_mlp_layers(hidden_dim * 2, hidden_dim, n_layers, dropout), nn.Linear(max(hidden_dim // (2 ** (n_layers - 1)), 32), 1))

    def forward(self, aptamer: torch.Tensor, molecule: torch.Tensor) -> torch.Tensor:
        apt_seq = self.apt_projection(aptamer).reshape(-1, self.seq_len, self.hidden_dim)
        mol_seq = self.mol_projection(molecule).reshape(-1, self.seq_len, self.hidden_dim)
        return self.fusion(torch.cat((self.apt_top(apt_seq), self.mol_top(mol_seq)), dim=1)).squeeze(-1)


class CrossAttentionBlock(nn.Module):
    """Residual multi-head cross-attention block."""

    def __init__(self, hidden_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(query, context, context, need_weights=False)
        return self.norm(query + self.dropout(attended))


class CrossAttentionModel(nn.Module):
    """Bidirectional cross-attention fusion over modality tokens."""

    def __init__(
        self,
        apt_dim: int,
        mol_dim: int,
        hidden_dim: int,
        n_layers: int,
        dropout: float,
        n_heads: int,
    ):
        super().__init__()
        self.apt_projection = nn.Sequential(nn.Linear(apt_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())
        self.mol_projection = nn.Sequential(nn.Linear(mol_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())
        self.apt_attention = nn.ModuleList(CrossAttentionBlock(hidden_dim, n_heads, dropout) for _ in range(n_layers))
        self.mol_attention = nn.ModuleList(CrossAttentionBlock(hidden_dim, n_heads, dropout) for _ in range(n_layers))
        self.classifier = nn.Sequential(_mlp_layers(hidden_dim * 2, hidden_dim, n_layers, dropout), nn.Linear(max(hidden_dim // (2 ** (n_layers - 1)), 32), 1))

    def forward(self, aptamer: torch.Tensor, molecule: torch.Tensor) -> torch.Tensor:
        apt = self.apt_projection(aptamer).unsqueeze(1)
        mol = self.mol_projection(molecule).unsqueeze(1)
        apt_context, mol_context = apt, mol
        for apt_block, mol_block in zip(self.apt_attention, self.mol_attention):
            apt_context = apt_block(apt_context, mol)
            mol_context = mol_block(mol_context, apt)
        fused = torch.cat((apt_context.squeeze(1), mol_context.squeeze(1)), dim=1)
        return self.classifier(fused).squeeze(-1)


def suggest_classification_params(trial: Any, model_name: str) -> dict[str, Any]:
    """Return a bounded model-specific E7 search configuration."""
    if model_name == "LightGBM":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        }
    params: dict[str, Any] = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
        "n_layers": trial.suggest_int("n_layers", 1, 3),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
    }
    if model_name == "CNNDualEncoder":
        params["seq_len"] = trial.suggest_categorical("seq_len", [8, 16, 32])
        params["kernel_size"] = trial.suggest_categorical("kernel_size", [3, 5, 7])
    elif model_name == "CrossAttention":
        params["n_heads"] = trial.suggest_categorical("n_heads", [2, 4, 8])
    elif model_name != "TwoTowerMLP":
        raise ValueError(f"Unknown model: {model_name}")
    return params


def suggest_regression_params(trial: Any) -> dict[str, Any]:
    """Regression-specific LightGBM search space."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 160),
        "max_depth": trial.suggest_int("max_depth", 3, 16),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 80),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 10.0, log=True),
    }


def build_dl_model(model_name: str, apt_dim: int, mol_dim: int, params: dict[str, Any]) -> nn.Module:
    """Instantiate one of the three deep E7 architectures."""
    architecture = {key: params[key] for key in ("hidden_dim", "n_layers", "dropout")}
    if model_name == "TwoTowerMLP":
        return TwoTowerMLP(apt_dim, mol_dim, **architecture)
    if model_name == "CNNDualEncoder":
        return CNNDualEncoder(
            apt_dim,
            mol_dim,
            **architecture,
            seq_len=params["seq_len"],
            kernel_size=params["kernel_size"],
        )
    if model_name == "CrossAttention":
        return CrossAttentionModel(apt_dim, mol_dim, **architecture, n_heads=params["n_heads"])
    raise ValueError(f"Not a DL model: {model_name}")


def _classification_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    predictions = (scores >= 0.5).astype(int)
    roc_auc = roc_auc_score(y_true, scores) if np.unique(y_true).size > 1 else math.nan
    mcc = matthews_corrcoef(y_true, predictions) if np.unique(y_true).size > 1 else math.nan
    return {"roc_auc": float(roc_auc), "mcc": float(mcc)}


def _predict_dl(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    scores = []
    with torch.no_grad():
        for aptamer, molecule, _ in loader:
            logits = model(aptamer.to(device), molecule.to(device))
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores)


def evaluate_dl_fold(
    model_name: str,
    params: dict[str, Any],
    apt_features: np.ndarray,
    mol_features: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    seed: int,
    device: str | torch.device,
    max_epochs: int = 80,
    patience: int = 10,
) -> dict[str, float | int]:
    """Fit one DL fold with ROC-AUC early stopping and return its metrics."""
    started = time.perf_counter()
    set_global_seed(seed)
    target_device = torch.device(device)
    apt_scaler, mol_scaler = StandardScaler(), StandardScaler()
    apt_train = apt_scaler.fit_transform(apt_features[train_idx]).astype(np.float32)
    apt_val = apt_scaler.transform(apt_features[val_idx]).astype(np.float32)
    mol_train = mol_scaler.fit_transform(mol_features[train_idx]).astype(np.float32)
    mol_val = mol_scaler.transform(mol_features[val_idx]).astype(np.float32)
    y_train = labels[train_idx].astype(np.float32)
    y_val = labels[val_idx].astype(np.float32)
    train_dataset = TensorDataset(torch.from_numpy(apt_train), torch.from_numpy(mol_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(apt_val), torch.from_numpy(mol_val), torch.from_numpy(y_val))
    generator = torch.Generator().manual_seed(seed)
    batch_size = min(int(params["batch_size"]), len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=len(train_dataset) % batch_size == 1, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model = build_dl_model(model_name, apt_train.shape[1], mol_train.shape[1], params).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
    criterion = nn.BCEWithLogitsLoss()
    best_score, best_state, stale, best_epoch = -math.inf, None, 0, 0
    for epoch in range(max_epochs):
        model.train()
        for aptamer, molecule, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(aptamer.to(target_device), molecule.to(target_device))
            loss = criterion(logits, target.to(target_device))
            loss.backward()
            optimizer.step()
        scores = _predict_dl(model, val_loader, target_device)
        score = _classification_metrics(y_val.astype(int), scores)["roc_auc"]
        if score > best_score:
            best_score = score
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("No valid DL checkpoint was produced")
    model.load_state_dict(best_state)
    scores = _predict_dl(model, val_loader, target_device)
    metrics = _classification_metrics(y_val.astype(int), scores)
    metrics.update(
        {
            "epochs": epoch + 1,
            "best_epoch": best_epoch,
            "seconds": time.perf_counter() - started,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
        }
    )
    del model
    if target_device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def evaluate_lgbm_classification_fold(
    params: dict[str, Any],
    features: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    seed: int,
) -> dict[str, float | int]:
    """Fit and score one LightGBM classification fold."""
    started = time.perf_counter()
    scaler = StandardScaler()
    train_features = scaler.fit_transform(features[train_idx])
    val_features = scaler.transform(features[val_idx])
    model = LGBMClassifier(**params, random_state=seed, n_jobs=2, verbosity=-1)
    model.fit(
        train_features,
        labels[train_idx],
        eval_set=[(val_features, labels[val_idx])],
        eval_metric="auc",
        callbacks=[early_stopping(10, verbose=False), log_evaluation(0)],
    )
    scores = model.predict_proba(val_features)[:, 1]
    metrics = _classification_metrics(labels[val_idx], scores)
    metrics.update(
        {
            "epochs": int(model.best_iteration_ or params["n_estimators"]),
            "best_epoch": int(model.best_iteration_ or params["n_estimators"]),
            "seconds": time.perf_counter() - started,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
        }
    )
    return metrics


def evaluate_regression_fold(
    params: dict[str, Any],
    features: np.ndarray,
    targets: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    seed: int,
) -> dict[str, float | int]:
    """Fit one regression fold after excluding missing pKd targets."""
    started = time.perf_counter()
    train_idx = train_idx[np.isfinite(targets[train_idx])]
    val_idx = val_idx[np.isfinite(targets[val_idx])]
    if not len(train_idx) or not len(val_idx):
        raise ValueError("Regression fold contains no finite targets")
    scaler = StandardScaler()
    train_features = scaler.fit_transform(features[train_idx])
    val_features = scaler.transform(features[val_idx])
    model = LGBMRegressor(**params, random_state=seed, n_jobs=2, verbosity=-1)
    model.fit(
        train_features,
        targets[train_idx],
        eval_set=[(val_features, targets[val_idx])],
        eval_metric="rmse",
        callbacks=[early_stopping(20, verbose=False), log_evaluation(0)],
    )
    predictions = model.predict(val_features, num_iteration=model.best_iteration_)
    return {
        "rmse": float(np.sqrt(mean_squared_error(targets[val_idx], predictions))),
        "r2": float(r2_score(targets[val_idx], predictions)),
        "mae": float(mean_absolute_error(targets[val_idx], predictions)),
        "best_iteration": int(model.best_iteration_ or params["n_estimators"]),
        "seconds": time.perf_counter() - started,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
    }


def aggregate_folds(rows: Iterable[dict[str, Any]], metrics: Iterable[str]) -> dict[str, float]:
    """Calculate equal-weight fold means and population standard deviations."""
    frame = pd.DataFrame(rows)
    summary: dict[str, float] = {}
    for metric in metrics:
        summary[f"{metric}_mean"] = float(frame[metric].mean())
        summary[f"{metric}_std"] = float(frame[metric].std(ddof=0))
    return summary
