"""
Loss Functions Module for Imbalanced Financial Fraud Detection.
Phase 7: Custom Focal Loss and Weighted BCE implementations.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Numerically stable Binary Focal Loss for severe class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    where:
        p_t = p if y=1 else (1 - p), with p = sigmoid(logits)
        alpha_t = alpha if y=1 else (1 - alpha) (if alpha is specified)
        gamma = focusing parameter that down-weights the loss contributed by easy examples.
        
    Args:
        gamma: Focusing parameter (gamma >= 0). When gamma=0, Focal Loss is equivalent to BCE.
               Higher values (e.g., 1.0, 2.0, 3.0) reduce the relative loss for easy,
               well-classified non-fraud transactions.
        alpha: Balancing factor for positive class vs negative class.
               If None, no static class weighting is applied (pure focal modulation).
               If float in (0, 1), scales positive class by alpha and negative class by (1 - alpha).
        reduction: 'mean', 'sum', or 'none'.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[float] = None,
        reduction: str = "mean"
    ):
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")
        if alpha is not None and not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction}")

        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss on raw logits.
        
        Args:
            logits: Predicted raw logits of shape (N, 1) or (N,).
            targets: Binary ground truth labels of shape (N, 1) or (N,), values in {0, 1}.
            
        Returns:
            Computed focal loss scalar or tensor depending on reduction.
        """
        targets = targets.view_as(logits).float()

        # Numerically stable unreduced cross entropy: -log(p_t)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Probabilities
        p = torch.sigmoid(logits)
        # p_t: probability of true class
        p_t = p * targets + (1.0 - p) * (1.0 - targets)

        # Modulating factor (1 - p_t)^gamma
        focal_weight = torch.pow(1.0 - p_t, self.gamma)

        # Optional alpha balancing
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * focal_weight * bce_loss
        else:
            loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy with Logits Loss.
    Wraps PyTorch's nn.BCEWithLogitsLoss with pos_weight handling.
    """

    def __init__(self, pos_weight: float, reduction: str = "mean"):
        super().__init__()
        self.pos_weight_val = pos_weight
        self.reduction = reduction
        self.loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]),
            reduction=reduction
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.view_as(logits).float()
        if self.loss_fn.pos_weight.device != logits.device:
            self.loss_fn.pos_weight = self.loss_fn.pos_weight.to(logits.device)
        return self.loss_fn(logits, targets)


def get_loss_function(
    loss_name: str,
    pos_weight: Optional[float] = None,
    gamma: float = 2.0,
    alpha: Optional[float] = None,
    reduction: str = "mean"
) -> nn.Module:
    """
    Factory function to retrieve loss functions by name.
    
    Args:
        loss_name: 'bce', 'weighted_bce', or 'focal'.
        pos_weight: Positive class weight for weighted BCE.
        gamma: Focusing parameter for Focal Loss.
        alpha: Alpha class weight for Focal Loss.
        reduction: 'mean', 'sum', or 'none'.
        
    Returns:
        Configured nn.Module loss instance.
    """
    name = loss_name.lower().strip()
    if name == "bce":
        return nn.BCEWithLogitsLoss(reduction=reduction)
    elif name in ("weighted_bce", "wbce"):
        if pos_weight is None:
            raise ValueError("pos_weight must be specified for weighted BCE")
        return WeightedBCELoss(pos_weight=pos_weight, reduction=reduction)
    elif name == "focal":
        return FocalLoss(gamma=gamma, alpha=alpha, reduction=reduction)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}. Choose from 'bce', 'weighted_bce', 'focal'.")
