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
│   ├── 08_explainability_feature_analysis.ipynb # Permutation Importance & PCA Embeddings
│   ├── 09_inference_pipeline.ipynb      # End-to-end inference engine & real-time latency
│   └── 10_streamlit_dashboard.ipynb     # Streamlit application verification & audit demo
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
├── tests/
│   └── test_integration_qa.py # Automated QA, leakage audit, and pipeline unit tests
├── reports/
│   ├── architecture/         # System architecture diagram (PNG) and specification (MD)
│   ├── figures/              # High-res publication-grade benchmark and ablation figures
│   ├── final_report/         # Comprehensive 12-chapter final technical report (MD)
│   └── viva/                 # Exhaustive oral examination & viva defense guide (MD)
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
12. [x] Final Inference Pipeline (`src/inference.py`, schema validation, batch scoring, latency benchmarks)
13. [x] Streamlit Interactive Fraud Scoring Dashboard (`app/app.py`)
14. [x] Final Integration, QA, Leakage Audit & Test Suite (`tests/test_integration_qa.py`)
15. [x] Final Project Report, Architecture Diagrams & Viva Preparation (`reports/`)

## Technical Documentation & Viva Defense Guide (Phase 13)

Phase 13 delivers comprehensive technical documentation, publication-grade figures, and oral defense guides:

* **[Final Technical Report (12 Chapters)](file:///c:/Ongoing%20Projects/2026.09.30%20Multimodal-Financial-Fraud-Detector/Multimodal-Financial-Fraud-Detector/reports/final_report/FINAL_TECHNICAL_REPORT.md)**: Full project documentation covering problem formulation, entity embedding math, 949-dim architecture, empirical ablation benchmarks, cost optimization, and leakage audits.
* **[System Architecture Specification](file:///c:/Ongoing%20Projects/2026.09.30%20Multimodal-Financial-Fraud-Detector/Multimodal-Financial-Fraud-Detector/reports/architecture/SYSTEM_ARCHITECTURE.md)**: Mathematical layer breakdowns, ASCII diagrams, and Mermaid workflow specifications.
* **[Viva Defense Preparation Guide](file:///c:/Ongoing%20Projects/2026.09.30%20Multimodal-Financial-Fraud-Detector/Multimodal-Financial-Fraud-Detector/reports/viva/VIVA_DEFENSE_PREPARATION.md)**: Targeted technical Q&A pairs covering architectural trade-offs, loss formulation comparisons, threshold calibration, and leakage prevention.
* **Architecture Visual**: `reports/architecture/system_architecture.png`
* **Benchmark Figures**: `reports/figures/01_model_pr_auc_comparison.png` through `05_model_roc_auc_comparison.png`.

## Inference Pipeline API (Phase 10)

The inference pipeline is exposed as a standalone, zero-leakage Python module:

```python
from src.inference import FraudInferencePipeline

# Load pipeline from frozen checkpoints
pipeline = FraudInferencePipeline.from_artifacts(
    model_path="models/pytorch_hybrid_weighted_bce.pt",
    preprocessor_path="models/tabular_preprocessor.joblib",
    threshold=0.80,
)

# Score real-time transaction with counterfactual explainability
result = pipeline.predict(transaction_payload, include_explanation=True, top_k=5)
```

Structured output schema:
```json
{
  "transaction_id": 3000002,
  "fraud_probability": 0.1897,
  "decision_threshold": 0.80,
  "decision": "APPROVE",
  "risk_tier": "NORMAL / LOW RISK",
  "top_contributing_evidence": [
    {"Feature": "V87", "Feature_Type": "Numerical", "Observed_Value": 3.0, "Probability_Contribution": 0.1554},
    {"Feature": "DeviceType", "Feature_Type": "Categorical", "Observed_Value": "mobile", "Probability_Contribution": 0.1465}
  ],
  "execution_time_ms": 176.28
}
```

- **Scoring-Only Fast Path Latency**: Mean ~147ms (CPU), Median ~147ms.
- **Batch Inference Throughput**: > 1,500 transactions/second (Batch size 2,048).
- **Fault-Tolerance**: Out-of-vocabulary categories mapped to `<UNK>` token; missing schema fields imputed via training distribution statistics.

## Streamlit Real-Time Dashboard (Phase 11)

The interactive web dashboard provides a graphical interface for analysts, risk officers, and auditors:

```bash
# Launch Streamlit dashboard locally
streamlit run app/app.py
```

### Dashboard Capabilities
1. **Single Transaction Scoring**: Guided input form & JSON editor with preset archetypes (Legitimate Retail, High-Risk Anomaly, Cross-Border Velocity Surge).
2. **Interactive Decision Policy**: Real-time slider ($\theta \in [0.05, 0.95]$) displaying how decisions transition between Approved, Elevated Risk, and Flagged for Review without re-running model forward passes.
3. **Local Explainability Evidence**: Altair horizontal sensitivity charts showing the top contributing features elevating transaction risk relative to baseline imputation.
4. **Batch Clearing & Risk Audit**: File uploader (.CSV) and built-in 500-transaction benchmark dataset with transaction throughput metrics, probability distributions, risk tier breakdowns, and full scored CSV export.

## Empirical Benchmark & Ablation Study Results

All experiments evaluated on an identical **strict chronological temporal test split** (15,000 transactions; $1.91\%$ fraud prevalence). Zero fabricated figures.

| Model Architecture | Input Features | Val PR-AUC | Test PR-AUC | Test ROC-AUC | Decision $\theta^*$ | Test Precision | Test Recall | Test F1 | Test False Positives |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** (Baseline) | 381 Numerical | 0.3444 | 0.2576 | 0.8106 | 0.90 | 32.84% | 36.39% | 0.3453 | 227 |
| **Random Forest** (100 Trees) | 381 Numerical | 0.5150 | 0.3077 | 0.8579 | 0.70 | 52.11% | 24.26% | 0.3311 | 68 |
| **XGBoost** (Depth 6, subsample 0.8) | 381 Numerical | 0.6100 | **0.4741** | 0.8759 | 0.80 | 54.35% | 49.18% | **0.5164** | 126 |
| **PyTorch Numerical MLP** (Standard BCE) | 381 Numerical | 0.5058 | 0.3828 | 0.8277 | 0.15 | 37.54% | 39.02% | 0.3826 | 198 |
| **PyTorch Numerical MLP** (Weighted BCE) | 381 Numerical | 0.4912 | 0.3778 | 0.7862 | 0.90 | 41.70% | 38.69% | 0.4014 | 165 |
| **PyTorch Hybrid** (Embeddings, Standard BCE) | 381 Num + 49 Cat | 0.5772 | 0.4623 | 0.8702 | 0.20 | 47.83% | 43.28% | 0.4544 | 144 |
| **PyTorch Hybrid** (Embeddings, Weighted BCE) | 381 Num + 49 Cat | 0.5368 | **0.4627** | **0.8799** | 0.95 | **58.33%** | 43.61% | 0.4991 | **95** |
| **PyTorch Hybrid** (Embeddings, Focal Loss $\gamma=3$) | 381 Num + 49 Cat | 0.5555 | 0.4257 | 0.8497 | 0.45 | 54.46% | 40.00% | 0.4612 | 102 |

*Key Findings*:
1. **Categorical Entity Embeddings**: Adding 49 categorical embeddings (568 embedding dimensions) boosted PyTorch Test PR-AUC from $0.3828$ to $0.4627$ ($+20.9\%$ relative gain).
2. **Operational Efficiency**: The Hybrid Weighted BCE model achieves the lowest false positive count among all competitive models ($95$ FP vs $126$ FP in XGBoost), yielding a peak Test Precision of $58.33\%$.
3. **Focal Loss Calibration**: Focal Loss ($\gamma=3$) shifted the optimal operational threshold to a well-calibrated mid-range ($\theta^* = 0.45$) while maintaining $54.46\%$ Precision.

## Zero Data Leakage Protocol & Audit Verification

A rigorous leakage audit was conducted across the codebase to ensure complete academic defensibility for project evaluation and viva:
1. **Temporal Splitting Prior to Preprocessing**: Data is chronologically ordered by `TransactionDT` and split into $70\%$ Train, $15\%$ Validation, and $15\%$ Test *before* any feature transformations are instantiated.
2. **Frozen Encoders & Scalers**: Imputation medians, standard scaling means/variances, and categorical vocabulary lookups are fitted strictly on `train_df`. Validation, test, and inference pipelines apply transform-only routines.
3. **Threshold Calibration on Validation Set Only**: Operating decision thresholds ($\theta^*$) and financial cost curves were optimized exclusively using validation-set predictions, preventing test-set data snooping.
4. **Decoupled Inference**: `src.inference.FraudInferencePipeline` loads frozen `.pt` and `.joblib` disk artifacts, containing zero calls to `.fit()` or `.fit_transform()`.

## Automated Integration & QA Test Suite

The project includes an automated regression test suite built on Python's standard `unittest` framework:

```bash
# Run automated integration & QA test suite
python -m unittest tests/test_integration_qa.py -v
```

Tests cover:
* Artifact integrity and non-empty checkpoint sizes.
* Zero-leakage static source code assertions.
* Input dimension verification ($381\text{ numerical} + 568\text{ embedding} = 949\text{ total inputs}$).
* Real-time single transaction scoring and latency.
* High-throughput batch scoring on 500 records.
* Robustness to out-of-vocabulary categories and sparse input payloads.
* Dynamic threshold controllability.
* Decoupled Streamlit presentation layer.



