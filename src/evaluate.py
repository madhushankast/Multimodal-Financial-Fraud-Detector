"""
Evaluation Module for Financial Fraud Detection.
Computes PR-AUC, ROC-AUC, Precision, Recall, F1, Confusion Matrix,
and visualizes Precision-Recall and ROC Curves across models.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_fraud_metrics(
    y_true: Union[np.ndarray, List[float]],
    y_prob: Union[np.ndarray, List[float]],
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Compute comprehensive fraud classification metrics.
    
    Args:
        y_true: Ground truth binary targets (0 or 1).
        y_prob: Predicted fraud probabilities in [0.0, 1.0].
        threshold: Operating decision threshold for binary classification.
        
    Returns:
        Dictionary containing Precision, Recall, F1, ROC-AUC, PR-AUC, and Confusion Matrix.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    
    # Calculate confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    # Probabilistic threshold-independent metrics
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        roc_auc = float("nan")
        
    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except ValueError:
        pr_auc = float("nan")
        
    return {
        "Threshold": round(threshold, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "ROC-AUC": round(roc_auc, 4),
        "PR-AUC": round(pr_auc, 4),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Total_Transactions": len(y_true),
        "Actual_Frauds": int(tp + fn),
        "Predicted_Frauds": int(tp + fp)
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """Plot and optionally save a clean confusion matrix heatmap with fraud-specific annotations."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    labels = [
        [f"True Negative (TN)\n{tn:,}\n(Legit Approved)", f"False Positive (FP)\n{fp:,}\n(Legit Blocked)"],
        [f"False Negative (FN)\n{fn:,}\n(Fraud Missed)", f"True Positive (TP)\n{tp:,}\n(Fraud Caught)"]
    ]
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted Legit (0)", "Predicted Fraud (1)"],
        yticklabels=["Actual Legit (0)", "Actual Fraud (1)"],
        annot_kws={"fontsize": 10, "weight": "bold"}
    )
    plt.title(f"Confusion Matrix: {model_name}", fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        
    return fig


def plot_pr_curves(
    models_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Plot comparative Precision-Recall curves across multiple models.
    
    Args:
        models_predictions: Dict mapping model_name -> (y_true, y_prob).
        save_path: Optional output filepath.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for name, (y_true, y_prob) in models_predictions.items():
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        ax.plot(recall, precision, lw=2, label=f"{name} (PR-AUC = {pr_auc:.4f})")
        
    # Baseline random guessing line is positive prevalence
    first_y_true = next(iter(models_predictions.values()))[0]
    base_rate = np.mean(first_y_true)
    ax.axhline(y=base_rate, color="gray", linestyle="--", alpha=0.7, label=f"Random Chance (Prevalence = {base_rate*100:.2f}%)")
    
    ax.set_title("Precision-Recall Curves (Primary Imbalanced Metric)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Recall (Fraud Coverage)", fontsize=10)
    ax.set_ylabel("Precision (Detection Accuracy)", fontsize=10)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        
    return fig


def plot_roc_curves(
    models_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Plot comparative ROC curves across multiple models.
    
    Args:
        models_predictions: Dict mapping model_name -> (y_true, y_prob).
        save_path: Optional output filepath.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for name, (y_true, y_prob) in models_predictions.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (ROC-AUC = {roc_auc:.4f})")
        
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5, label="Random Guess (AUC = 0.50)")
    ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=11, fontweight="bold")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=10)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        
    return fig


def evaluate_thresholds(
    y_true: Union[np.ndarray, List[float]],
    y_prob: Union[np.ndarray, List[float]],
    thresholds: Optional[List[float]] = None
) -> pd.DataFrame:
    """
    Evaluate binary classification metrics across candidate probability thresholds.
    
    Args:
        y_true: Ground truth binary targets.
        y_prob: Predicted fraud probabilities.
        thresholds: List of thresholds to test (defaults to 0.05 to 0.95 in steps of 0.05).
        
    Returns:
        DataFrame containing metrics across all evaluated thresholds.
    """
    if thresholds is None:
        thresholds = [round(float(t), 2) for t in np.arange(0.05, 1.00, 0.05)]
        
    records = []
    for t in thresholds:
        m = compute_fraud_metrics(y_true, y_prob, threshold=t)
        records.append(m)
        
    return pd.DataFrame(records)


def find_optimal_threshold(
    y_true: Union[np.ndarray, List[float]],
    y_prob: Union[np.ndarray, List[float]],
    metric: str = "F1",
    thresholds: Optional[List[float]] = None
) -> Tuple[float, Dict[str, Any]]:
    """
    Identify optimal decision threshold on validation data maximizing a target metric.
    
    Args:
        y_true: Validation ground truth.
        y_prob: Validation predicted probabilities.
        metric: Target optimization metric (e.g. 'F1', 'Recall', 'Precision').
        thresholds: Candidate thresholds.
        
    Returns:
        Tuple of (optimal_threshold, metrics_dict_at_optimal_threshold).
    """
    df_thresh = evaluate_thresholds(y_true, y_prob, thresholds=thresholds)
    best_idx = df_thresh[metric].idxmax()
    best_row = df_thresh.loc[best_idx]
    best_threshold = float(best_row["Threshold"])
    return best_threshold, best_row.to_dict()


def compute_cost_metrics(
    y_true: Union[np.ndarray, List[float]],
    y_prob: Union[np.ndarray, List[float]],
    amounts: Union[np.ndarray, List[float], pd.Series],
    c_fp: float = 15.0,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Compute business risk and operational financial dollar loss at a specific threshold.
    
    Financial Cost Function:
      Total Cost(theta) = C_FP * FP(theta) + sum_{i in FN(theta)} TransactionAmt_i
      
    Args:
        y_true: Ground truth binary targets (0 or 1).
        y_prob: Predicted fraud probabilities in [0.0, 1.0].
        amounts: Transaction dollar amounts (TransactionAmt).
        c_fp: Cost per false positive in dollars (investigation cost + customer friction).
        threshold: Operating decision threshold.
        
    Returns:
        Dictionary containing statistical counts, classification metrics, and financial costs.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    amounts = np.asarray(amounts).astype(float)
    
    y_pred = (y_prob >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    # Financial dollar calculations
    fn_mask = (y_true == 1) & (y_pred == 0)
    tp_mask = (y_true == 1) & (y_pred == 1)
    
    missed_fraud_amount = float(np.sum(amounts[fn_mask]))
    prevented_fraud_amount = float(np.sum(amounts[tp_mask]))
    total_fraud_amount = missed_fraud_amount + prevented_fraud_amount
    
    false_positive_cost = float(c_fp * fp)
    total_cost = false_positive_cost + missed_fraud_amount
    
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    return {
        "Threshold": round(float(threshold), 4),
        "C_FP": round(float(c_fp), 2),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "False_Positive_Cost": round(false_positive_cost, 2),
        "Missed_Fraud_Cost": round(missed_fraud_amount, 2),
        "Prevented_Fraud_Amount": round(prevented_fraud_amount, 2),
        "Total_Fraud_Amount": round(total_fraud_amount, 2),
        "Total_Cost": round(total_cost, 2),
        "Cost_Reduction_Pct": round(
            float(1.0 - (total_cost / total_fraud_amount)) * 100.0, 2
        ) if total_fraud_amount > 0 else 0.0
    }


def evaluate_cost_thresholds(
    y_true: Union[np.ndarray, List[float]],
    y_prob: Union[np.ndarray, List[float]],
    amounts: Union[np.ndarray, List[float], pd.Series],
    c_fp_list: Optional[List[float]] = None,
    thresholds: Optional[List[float]] = None
) -> pd.DataFrame:
    """
    Evaluate business financial costs across candidate thresholds and multiple C_FP values.
    """
    if thresholds is None:
        thresholds = [round(float(t), 2) for t in np.arange(0.05, 1.00, 0.05)]
    if c_fp_list is None:
        c_fp_list = [5.0, 10.0, 15.0, 25.0, 50.0]
        
    records = []
    for c_fp in c_fp_list:
        for t in thresholds:
            rec = compute_cost_metrics(y_true, y_prob, amounts, c_fp=c_fp, threshold=t)
            records.append(rec)
            
    return pd.DataFrame(records)


def find_cost_optimal_threshold(
    y_true: Union[np.ndarray, List[float]],
    y_prob: Union[np.ndarray, List[float]],
    amounts: Union[np.ndarray, List[float], pd.Series],
    c_fp: float = 15.0,
    thresholds: Optional[List[float]] = None
) -> Tuple[float, Dict[str, Any]]:
    """
    Identify optimal decision threshold minimizing Total Financial Cost on validation data.
    """
    if thresholds is None:
        thresholds = [round(float(t), 2) for t in np.arange(0.05, 1.00, 0.05)]
        
    records = [
        compute_cost_metrics(y_true, y_prob, amounts, c_fp=c_fp, threshold=t)
        for t in thresholds
    ]
    df = pd.DataFrame(records)
    best_idx = df["Total_Cost"].idxmin()
    best_row = df.loc[best_idx]
    best_threshold = float(best_row["Threshold"])
    return best_threshold, best_row.to_dict()


def plot_cost_curves(
    cost_df: pd.DataFrame,
    title: str = "Total Financial Cost vs Decision Threshold",
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Plot total business loss curves across decision thresholds for various C_FP friction costs.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    
    c_fps = sorted(cost_df["C_FP"].unique())
    palette = sns.color_palette("viridis", n_colors=len(c_fps))
    
    for c_fp, color in zip(c_fps, palette):
        sub_df = cost_df[cost_df["C_FP"] == c_fp].sort_values("Threshold")
        min_row = sub_df.loc[sub_df["Total_Cost"].idxmin()]
        ax.plot(
            sub_df["Threshold"],
            sub_df["Total_Cost"],
            marker="o",
            markersize=4,
            lw=2,
            color=color,
            label=f"C_FP = ${c_fp:.0f} (Min @ θ={min_row['Threshold']:.2f}, ${min_row['Total_Cost']:,.0f})"
        )
        ax.plot(min_row["Threshold"], min_row["Total_Cost"], marker="*", markersize=12, color=color)
        
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Fraud Decision Threshold (θ)", fontsize=10)
    ax.set_ylabel("Total Financial Loss ($ USD)", fontsize=10)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        
    return fig


def compute_permutation_importance(
    model: torch.nn.Module,
    X_num: np.ndarray,
    X_cat: np.ndarray,
    y_true: np.ndarray,
    numerical_feature_names: List[str],
    categorical_feature_names: List[str],
    device: Optional[torch.device] = None,
    metric: str = "PR-AUC",
    top_n_numerical: Optional[int] = None,
    batch_size: int = 2048,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Compute permutation feature importance on validation data for Hybrid PyTorch architecture.
    Measures the drop in validation PR-AUC when each feature is randomly shuffled.
    
    Args:
        model: Trained HybridFraudDetector.
        X_num: Preprocessed numerical feature array (N, num_features).
        X_cat: Preprocessed categorical feature array (N, num_cats).
        y_true: Ground truth binary targets (N,).
        numerical_feature_names: List of numerical column names.
        categorical_feature_names: List of categorical column names.
        device: PyTorch device ('cpu' or 'cuda').
        metric: 'PR-AUC' (default) or 'ROC-AUC'.
        top_n_numerical: Optional limit to evaluate top numerical candidates (if None, evaluates all).
        batch_size: Batch size for batched inference passes.
        random_state: Random seed for reproducibility.
        
    Returns:
        DataFrame sorted by Importance_Drop descending.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    rng = np.random.default_rng(random_state)
    
    @torch.no_grad()
    def predict_probs(x_n: np.ndarray, x_c: np.ndarray) -> np.ndarray:
        probs = []
        n_samples = len(x_n)
        for i in range(0, n_samples, batch_size):
            b_n = torch.tensor(x_n[i:i+batch_size], dtype=torch.float32, device=device)
            b_c = torch.tensor(x_c[i:i+batch_size], dtype=torch.long, device=device)
            logits = model(b_n, b_c)
            p = torch.sigmoid(logits).cpu().numpy().ravel()
            probs.append(p)
        return np.concatenate(probs)

    # 1. Baseline Score
    baseline_probs = predict_probs(X_num, X_cat)
    if metric == "PR-AUC":
        baseline_score = float(average_precision_score(y_true, baseline_probs))
    else:
        baseline_score = float(roc_auc_score(y_true, baseline_probs))
        
    records = []
    
    # 2. Evaluate all Categorical Features
    for idx, col_name in enumerate(categorical_feature_names):
        x_cat_perm = X_cat.copy()
        x_cat_perm[:, idx] = rng.permutation(x_cat_perm[:, idx])
        perm_probs = predict_probs(X_num, x_cat_perm)
        
        if metric == "PR-AUC":
            perm_score = float(average_precision_score(y_true, perm_probs))
        else:
            perm_score = float(roc_auc_score(y_true, perm_probs))
            
        drop = max(0.0, baseline_score - perm_score)
        records.append({
            "Feature": col_name,
            "Feature_Type": "Categorical",
            "Baseline_Score": round(baseline_score, 4),
            "Permuted_Score": round(perm_score, 4),
            "Importance_Drop": round(drop, 6),
            "Metric": metric
        })

    # 3. Evaluate Numerical Features
    num_indices = list(range(len(numerical_feature_names)))
    if top_n_numerical is not None and top_n_numerical < len(num_indices):
        # Quick heuristic sample or variance selection if restricted
        num_indices = num_indices[:top_n_numerical]
        
    for idx in num_indices:
        col_name = numerical_feature_names[idx]
        x_num_perm = X_num.copy()
        x_num_perm[:, idx] = rng.permutation(x_num_perm[:, idx])
        perm_probs = predict_probs(x_num_perm, X_cat)
        
        if metric == "PR-AUC":
            perm_score = float(average_precision_score(y_true, perm_probs))
        else:
            perm_score = float(roc_auc_score(y_true, perm_probs))
            
        drop = max(0.0, baseline_score - perm_score)
        records.append({
            "Feature": col_name,
            "Feature_Type": "Numerical",
            "Baseline_Score": round(baseline_score, 4),
            "Permuted_Score": round(perm_score, 4),
            "Importance_Drop": round(drop, 6),
            "Metric": metric
        })
        
    df_importance = pd.DataFrame(records).sort_values("Importance_Drop", ascending=False).reset_index(drop=True)
    return df_importance


def extract_embedding_weights(
    model: torch.nn.Module,
    feature_name: str,
    categorical_feature_names: List[str]
) -> np.ndarray:
    """
    Extract learned embedding weights matrix from HybridFraudDetector for a given categorical feature.
    
    Args:
        model: Trained HybridFraudDetector.
        feature_name: Name of categorical column (e.g. 'ProductCD', 'DeviceInfo').
        categorical_feature_names: List of all categorical feature names in model order.
        
    Returns:
        Numpy array of shape (vocab_size, embedding_dim).
    """
    if feature_name not in categorical_feature_names:
        raise ValueError(f"Feature '{feature_name}' not found in categorical features.")
    idx = categorical_feature_names.index(feature_name)
    emb_modules = getattr(model, "embeddings", getattr(model, "categorical_embeddings", None))
    if emb_modules is None:
        raise AttributeError("Model does not have 'embeddings' or 'categorical_embeddings' attribute.")
    emb_layer = emb_modules[idx]
    return emb_layer.weight.detach().cpu().numpy()


def explain_transaction(
    model: torch.nn.Module,
    x_num_single: np.ndarray,
    x_cat_single: np.ndarray,
    numerical_feature_names: List[str],
    categorical_feature_names: List[str],
    raw_transaction_dict: Optional[Dict[str, Any]] = None,
    threshold: float = 0.80,
    top_k: int = 5,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Generate local counterfactual perturbation explanation for an individual transaction.
    
    Identifies which features drove the fraud probability UP toward the decision threshold
    by measuring the drop in predicted probability when each feature is replaced with
    its neutral reference value.
    
    Args:
        model: Trained HybridFraudDetector.
        x_num_single: 1D array of preprocessed numerical features for this transaction.
        x_cat_single: 1D array of preprocessed categorical indices for this transaction.
        numerical_feature_names: List of numerical column names.
        categorical_feature_names: List of categorical column names.
        raw_transaction_dict: Optional raw input values for human-readable reporting.
        threshold: Operating decision threshold.
        top_k: Number of top evidence features to report.
        device: PyTorch device.
        
    Returns:
        Dictionary containing fraud probability, risk tier, decision, and ranked evidence list.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    
    t_num = torch.tensor(x_num_single.reshape(1, -1), dtype=torch.float32, device=device)
    t_cat = torch.tensor(x_cat_single.reshape(1, -1), dtype=torch.long, device=device)
    
    with torch.no_grad():
        base_logit = model(t_num, t_cat)
        base_prob = float(torch.sigmoid(base_logit).item())
        
    # Evaluate contribution of each numerical feature
    contributions = []
    
    # 1. Numerical feature sensitivity (perturbation to zero = standardized median/mean)
    for i, col in enumerate(numerical_feature_names):
        curr_val = x_num_single[i]
        if abs(curr_val) < 1e-4:
            continue
        perturbed_num = x_num_single.copy()
        perturbed_num[i] = 0.0  # reference baseline
        p_tensor = torch.tensor(perturbed_num.reshape(1, -1), dtype=torch.float32, device=device)
        with torch.no_grad():
            perturbed_prob = float(torch.sigmoid(model(p_tensor, t_cat)).item())
            
        prob_delta = base_prob - perturbed_prob
        raw_val = raw_transaction_dict.get(col, curr_val) if raw_transaction_dict else curr_val
        contributions.append({
            "Feature": col,
            "Feature_Type": "Numerical",
            "Observed_Value": raw_val,
            "Probability_Contribution": round(prob_delta, 4)
        })

    # 2. Categorical feature sensitivity (perturbation to UNK = index 1)
    for j, col in enumerate(categorical_feature_names):
        curr_idx = x_cat_single[j]
        perturbed_cat = x_cat_single.copy()
        perturbed_cat[j] = 1  # reference UNK token
        c_tensor = torch.tensor(perturbed_cat.reshape(1, -1), dtype=torch.long, device=device)
        with torch.no_grad():
            perturbed_prob = float(torch.sigmoid(model(t_num, c_tensor)).item())
            
        prob_delta = base_prob - perturbed_prob
        raw_val = raw_transaction_dict.get(col, curr_idx) if raw_transaction_dict else curr_idx
        contributions.append({
            "Feature": col,
            "Feature_Type": "Categorical",
            "Observed_Value": raw_val,
            "Probability_Contribution": round(prob_delta, 4)
        })
        
    # Sort by positive contribution (factors that increased fraud likelihood)
    df_contrib = pd.DataFrame(contributions).sort_values("Probability_Contribution", ascending=False)
    top_evidence = df_contrib.head(top_k).to_dict(orient="records")
    
    decision = "FLAG FOR MANUAL REVIEW" if base_prob >= threshold else "APPROVE"
    risk_level = "CRITICAL / HIGH RISK" if base_prob >= 0.80 else ("ELEVATED RISK" if base_prob >= threshold else "NORMAL / LOW RISK")
    
    return {
        "Fraud_Probability": round(base_prob, 4),
        "Decision_Threshold": round(threshold, 2),
        "Decision": decision,
        "Risk_Level": risk_level,
        "Top_Contributing_Evidence": top_evidence,
        "Methodology_Note": "Feature importance represents associative sensitivity (counterfactual probability delta), not causal proof."
    }


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    title: str = "Top Feature Importances (Permutation PR-AUC Drop)",
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Plot horizontal bar chart of top-N features color-coded by feature type.
    """
    top_df = importance_df.head(top_n).sort_values("Importance_Drop", ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#2980b9" if t == "Numerical" else "#e67e22" for t in top_df["Feature_Type"]]
    
    bars = ax.barh(top_df["Feature"], top_df["Importance_Drop"], color=colors, edgecolor="black", linewidth=0.5)
    
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2980b9", edgecolor="black", label="Numerical Feature"),
        Patch(facecolor="#e67e22", edgecolor="black", label="Categorical Feature (Learned Embedding)")
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True)
    
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Permutation Importance (Validation PR-AUC Drop)", fontsize=10)
    ax.set_ylabel("Feature", fontsize=10)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        
    return fig


