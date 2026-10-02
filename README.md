# Multimodal Financial Fraud Detector

Engineered an end-to-end financial anomaly detection pipeline in PyTorch utilizing categorical entity embeddings and custom Focal Loss to handle severe class imbalance; evaluated precision-recall thresholds to optimize business risk metrics.

## Project Structure

```text
├── data/
│   ├── raw/                  # Original unprocessed datasets
│   └── processed/            # Cleaned, encoded, and split datasets
├── notebooks/
│   ├── 01_eda.ipynb                     # EDA and initial data inspection
│   ├── 02_preprocessing_pipeline.ipynb  # Preprocessing and PyTorch dataset verification
│   ├── 03_baseline_models.ipynb         # Classical ML baselines (Logistic, RF, XGBoost)
│   ├── 04_pytorch_mlp.ipynb             # PyTorch Numerical MLP & BCE Weighting Experiments
│   ├── 05_categorical_embeddings.ipynb  # Hybrid Embedding Network & Ablation Study
│   ├── 06_focal_loss_experiments.ipynb  # Custom Loss Functions (Focal Loss)
│   ├── 07_threshold_business_risk.ipynb # Operating Decision Threshold & Cost-Risk Analysis
│   └── 08_explainability_feature_analysis.ipynb # Permutation Importance & PCA Embeddings
├── src/
│   ├── __init__.py
│   ├── preprocessing.py      # Cleaning, encoding, scaling, and train/val/test split
│   ├── dataset.py            # PyTorch Dataset and DataLoader implementations
│   ├── model.py              # Neural network with categorical entity embeddings
│   ├── losses.py             # Weighted BCE & custom Focal Loss
│   ├── train.py              # Training loop, early stopping, and validation
│   ├── evaluate.py           # Metrics, cost analysis, permutation importance, explanations
│   └── inference.py          # Real-time scoring and inference pipeline
├── models/                   # Saved model weights and scaler/encoder checkpoints
├── results/
│   ├── metrics/              # Evaluation logs, test reports, and metrics JSON/CSV
│   └── figures/              # PR curves, ROC curves, loss curves, confusion matrices
├── app/
│   └── app.py                # Streamlit interactive fraud scoring dashboard
├── requirements.txt
└── README.md
```

## Roadmap

1. [x] Project Setup & Architecture Skeleton
2. [x] Dataset Selection & Acquisition
3. [x] Exploratory Data Analysis (EDA) & Feature Groups
4. [x] Preprocessing & Feature Engineering (Leakage-Safe Pipeline)
5. [x] Baseline Model Comparison (Logistic Regression, Random Forest, XGBoost)
6. [x] PyTorch Numerical MLP & Imbalance-Weighted BCE Baseline
7. [x] PyTorch Categorical Embedding Network (Hybrid Tabular Architecture)
8. [x] Custom Loss Functions (Focal Loss Exploration)
9. [x] Evaluation & Threshold Optimization (Operating Cost Analysis)
10. [x] Ablation Study (Models A through F Benchmarked)
11. [x] Explainability & Feature Importance (Permutation, Embeddings, Tree Comparison)
12. [ ] Inference Pipeline & Streamlit Real-Time Dashboard
