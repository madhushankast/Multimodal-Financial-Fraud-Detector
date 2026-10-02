"""
Training and Validation Pipeline Module for Financial Fraud Detection.
Phase 5: PyTorch MLP training loop, validation tracking, early stopping,
checkpoint saving, and metric evaluation.
"""

import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluate import compute_fraud_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping handler that monitors a validation metric and saves the best model checkpoint.
    """
    def __init__(
        self,
        patience: int = 5,
        mode: str = "max",
        delta: float = 1e-4,
        checkpoint_path: Optional[Union[str, Path]] = None,
        verbose: bool = True
    ):
        """
        Args:
            patience: Number of epochs to wait without improvement before stopping.
            mode: 'max' if higher is better (e.g., PR-AUC, F1), 'min' if lower is better (e.g., loss).
            delta: Minimum change in monitored metric to qualify as improvement.
            checkpoint_path: Optional path to save the best model weights.
            verbose: Verbosity flag.
        """
        self.patience = patience
        self.mode = mode.lower()
        self.delta = delta
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.verbose = verbose
        
        self.best_score: Optional[float] = None
        self.best_epoch: int = 0
        self.counter: int = 0
        self.early_stop: bool = False
        self.best_state_dict: Optional[Dict[str, Any]] = None

    def __call__(self, score: float, model: nn.Module, epoch: int) -> bool:
        if self.mode == "max":
            improved = (self.best_score is None) or (score > self.best_score + self.delta)
        else:
            improved = (self.best_score is None) or (score < self.best_score - self.delta)

        if improved:
            if self.verbose:
                prev = "None" if self.best_score is None else f"{self.best_score:.4f}"
                logger.info(f"Validation metric improved ({prev} -> {score:.4f}). Saving checkpoint...")
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            self.best_state_dict = copy.deepcopy(model.state_dict())
            
            if self.checkpoint_path:
                self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.best_state_dict, self.checkpoint_path)
        else:
            self.counter += 1
            if self.verbose:
                logger.info(f"EarlyStopping counter: {self.counter}/{self.patience} (Best: {self.best_score:.4f} at Epoch {self.best_epoch})")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info(f"Early stopping triggered. Restoring best weights from epoch {self.best_epoch}.")
                    
        return self.early_stop


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    clip_grad_norm: Optional[float] = 5.0
) -> float:
    """
    Run one full training epoch.
    
    Args:
        model: PyTorch model.
        dataloader: Training DataLoader.
        criterion: Loss function (e.g. BCEWithLogitsLoss).
        optimizer: Optimizer (e.g. AdamW).
        device: Computation device ('cpu' or 'cuda').
        clip_grad_norm: Max gradient norm for clipping.
        
    Returns:
        Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0
    
    for batch in dataloader:
        x_num, x_cat, y = batch
        x_num = x_num.to(device)
        x_cat = x_cat.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        logits = model(x_num, x_cat)
        loss = criterion(logits, y)
        loss.backward()
        
        if clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
            
        optimizer.step()
        
        batch_size = x_num.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size
        
    return running_loss / total_samples if total_samples > 0 else 0.0


@torch.no_grad()
def evaluate_dataloader(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5
) -> Tuple[float, Dict[str, Any], np.ndarray, np.ndarray]:
    """
    Evaluate model on a DataLoader without computing gradients.
    
    Returns:
        val_loss: Average loss over dataloader.
        metrics: Dictionary of fraud metrics at specified threshold.
        y_true: Numpy array of ground truth labels.
        y_prob: Numpy array of predicted fraud probabilities.
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0
    
    all_probs = []
    all_targets = []
    
    for batch in dataloader:
        x_num, x_cat, y = batch
        x_num = x_num.to(device)
        x_cat = x_cat.to(device)
        y = y.to(device)
        
        logits = model(x_num, x_cat)
        loss = criterion(logits, y)
        probs = torch.sigmoid(logits)
        
        batch_size = x_num.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size
        
        all_probs.append(probs.cpu().numpy())
        all_targets.append(y.cpu().numpy())
        
    val_loss = running_loss / total_samples if total_samples > 0 else 0.0
    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_targets)
    
    metrics = compute_fraud_metrics(y_true, y_prob, threshold=threshold)
    return val_loss, metrics, y_true, y_prob


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
    num_epochs: int = 15,
    patience: int = 5,
    device: Optional[torch.device] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    monitor_metric: str = "PR-AUC",
    verbose: bool = True
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Complete training loop with validation tracking, early stopping, and history recording.
    
    Args:
        model: PyTorch model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        scheduler: Optional learning rate scheduler.
        num_epochs: Maximum epochs.
        patience: Early stopping patience.
        device: 'cpu' or 'cuda'.
        checkpoint_path: Filepath to save best checkpoint.
        monitor_metric: Metric to monitor for early stopping ('PR-AUC', 'F1', or 'val_loss').
        verbose: Verbosity flag.
        
    Returns:
        best_model: Model with restored best weights.
        history: Dictionary containing training and validation curves.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model = model.to(device)
    mode = "min" if monitor_metric.lower() in ("val_loss", "loss") else "max"
    
    early_stopping = EarlyStopping(
        patience=patience,
        mode=mode,
        checkpoint_path=checkpoint_path,
        verbose=verbose
    )
    
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_pr_auc": [],
        "val_roc_auc": [],
        "val_f1": []
    }
    
    logger.info(f"Starting training for up to {num_epochs} epochs on {device} (Monitoring: {monitor_metric})...")
    
    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics, _, _ = evaluate_dataloader(model, val_loader, criterion, device)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_pr_auc"].append(val_metrics["PR-AUC"])
        history["val_roc_auc"].append(val_metrics["ROC-AUC"])
        history["val_f1"].append(val_metrics["F1"])
        
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss if mode == "min" else val_metrics["PR-AUC"])
            else:
                scheduler.step()
                
        lr_current = optimizer.param_groups[0]["lr"]
        if verbose:
            logger.info(
                f"Epoch {epoch:2d}/{num_epochs:2d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val PR-AUC: {val_metrics['PR-AUC']:.4f} | "
                f"Val ROC-AUC: {val_metrics['ROC-AUC']:.4f} | "
                f"Val F1: {val_metrics['F1']:.4f} | "
                f"LR: {lr_current:.2e}"
            )
            
        score_to_monitor = val_loss if mode == "min" else val_metrics[monitor_metric]
        if early_stopping(score_to_monitor, model, epoch):
            break
            
    # Load best weights
    if early_stopping.best_state_dict is not None:
        model.load_state_dict(early_stopping.best_state_dict)
        logger.info(f"Restored best model weights from epoch {early_stopping.best_epoch} with {monitor_metric} = {early_stopping.best_score:.4f}")
        
    return model, history


if __name__ == "__main__":
    print("Training module loaded successfully.")
