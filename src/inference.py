"""
Inference Pipeline Module for Financial Fraud Detection.
Phase 10: Production-style, standalone inference engine that loads
fitted preprocessing artifacts and trained PyTorch checkpoints to score
single transactions or batches with risk tiering and explainability evidence.
"""

from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.evaluate import explain_transaction
from src.model import HybridFraudDetector, compute_embedding_dim
from src.preprocessing import TabularPreprocessor


class FraudInferencePipeline:
    """
    Production-style End-to-End Financial Fraud Inference Pipeline.
    
    Accepts raw transaction dictionaries, JSON payloads, or DataFrames,
    validates schema, executes leakage-safe preprocessing with saved artifacts,
    scores via Hybrid neural network, applies cost-calibrated decision thresholds,
    and returns risk classifications with local explainability evidence.
    """

    def __init__(
        self,
        model: nn.Module,
        preprocessor: TabularPreprocessor,
        threshold: float = 0.80,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            model: Trained HybridFraudDetector.
            preprocessor: Fitted TabularPreprocessor.
            threshold: Operating decision threshold (default 0.80).
            device: PyTorch device ('cpu' or 'cuda').
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.preprocessor = preprocessor
        self.threshold = float(threshold)

        self.numerical_cols: List[str] = preprocessor.fitted_numerical_cols_
        self.categorical_cols: List[str] = preprocessor.fitted_categorical_cols_
        self.all_expected_cols: List[str] = self.numerical_cols + self.categorical_cols

    @classmethod
    def from_artifacts(
        cls,
        model_path: Union[str, Path] = "models/pytorch_hybrid_weighted_bce.pt",
        preprocessor_path: Union[str, Path] = "models/tabular_preprocessor.joblib",
        threshold: float = 0.80,
        device: Optional[torch.device] = None
    ) -> "FraudInferencePipeline":
        """
        Instantiate pipeline directly from saved model weights and preprocessor artifacts.
        
        Args:
            model_path: Filepath to saved PyTorch checkpoint.
            preprocessor_path: Filepath to saved fitted TabularPreprocessor.
            threshold: Operating decision threshold.
            device: Computation device.
            
        Returns:
            Configured FraudInferencePipeline instance.
        """
        model_path = Path(model_path)
        preprocessor_path = Path(preprocessor_path)

        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at: {preprocessor_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

        # 1. Load fitted TabularPreprocessor
        preprocessor = joblib.load(preprocessor_path)

        # 2. Extract feature dimensions and cardinalities
        num_numerical = len(preprocessor.fitted_numerical_cols_)
        cat_cols = preprocessor.fitted_categorical_cols_
        cat_cardinalities = [preprocessor.vocab_sizes_[c] for c in cat_cols]
        embedding_dims = [compute_embedding_dim(v) for v in cat_cardinalities]

        # 3. Instantiate Hybrid architecture and restore weights
        target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = HybridFraudDetector(
            num_numerical_features=num_numerical,
            categorical_cardinalities=cat_cardinalities,
            embedding_dims=embedding_dims
        )
        model.load_state_dict(torch.load(model_path, map_location=target_device))
        model.eval()

        return cls(model=model, preprocessor=preprocessor, threshold=threshold, device=target_device)

    def validate_input(
        self,
        data: Union[Dict[str, Any], pd.Series, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Validate input schema, impute missing columns with NaN, and enforce DataFrame formatting.
        
        Args:
            data: Raw transaction dict, pandas Series, or DataFrame.
            
        Returns:
            Standardized pandas DataFrame ready for preprocessor.
        """
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.Series):
            df = pd.DataFrame([data.to_dict()])
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise TypeError(f"Unsupported input type: {type(data)}. Expected dict, Series, or DataFrame.")

        # Ensure all expected columns exist; add missing ones simultaneously to prevent fragmentation
        missing_cols = [col for col in self.all_expected_cols if col not in df.columns]
        if missing_cols:
            missing_df = pd.DataFrame(np.nan, index=df.index, columns=missing_cols)
            df = pd.concat([df, missing_df], axis=1)

        return df

    def predict_proba(
        self,
        data: Union[Dict[str, Any], pd.Series, pd.DataFrame],
        batch_size: int = 2048
    ) -> np.ndarray:
        """
        Compute predicted fraud probabilities for input transactions.
        
        Args:
            data: Input transaction(s).
            batch_size: Batch size for multi-row inferences.
            
        Returns:
            1D numpy array of predicted fraud probabilities in [0.0, 1.0].
        """
        df = self.validate_input(data)
        X_num, X_cat, _ = self.preprocessor.transform(df)

        all_probs = []
        n_samples = len(df)

        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                b_num = torch.tensor(X_num[i:i+batch_size], dtype=torch.float32, device=self.device)
                b_cat = torch.tensor(X_cat[i:i+batch_size], dtype=torch.long, device=self.device)
                logits = self.model(b_num, b_cat)
                probs = torch.sigmoid(logits).cpu().numpy().ravel()
                all_probs.append(probs)

        return np.concatenate(all_probs)

    def predict(
        self,
        transaction: Union[Dict[str, Any], pd.Series, pd.DataFrame],
        include_explanation: bool = True,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Run end-to-end real-time inference on a single transaction with risk tiering and evidence.
        
        Args:
            transaction: Single transaction data payload.
            include_explanation: Whether to compute local counterfactual attribution evidence.
            top_k: Number of top evidence features to return.
            
        Returns:
            Structured prediction dictionary.
        """
        start_time = time.perf_counter()

        df = self.validate_input(transaction)
        if len(df) != 1:
            raise ValueError(f"predict() expects a single transaction, received {len(df)} rows. Use predict_batch() instead.")

        # Preprocess features
        X_num, X_cat, _ = self.preprocessor.transform(df)
        x_num_single = X_num[0]
        x_cat_single = X_cat[0]

        # Compute probability
        t_num = torch.tensor(x_num_single.reshape(1, -1), dtype=torch.float32, device=self.device)
        t_cat = torch.tensor(x_cat_single.reshape(1, -1), dtype=torch.long, device=self.device)

        with torch.no_grad():
            logit = self.model(t_num, t_cat)
            prob = float(torch.sigmoid(logit).item())

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        decision = "FLAG FOR MANUAL REVIEW" if prob >= self.threshold else "APPROVE"
        if prob >= 0.80:
            risk_tier = "CRITICAL / HIGH RISK"
        elif prob >= self.threshold:
            risk_tier = "ELEVATED RISK"
        else:
            risk_tier = "NORMAL / LOW RISK"

        raw_dict = df.iloc[0].to_dict()
        txn_id = raw_dict.get("TransactionID", "TXN_LIVE")

        explanation_evidence = []
        if include_explanation:
            exp_result = explain_transaction(
                model=self.model,
                x_num_single=x_num_single,
                x_cat_single=x_cat_single,
                numerical_feature_names=self.numerical_cols,
                categorical_feature_names=self.categorical_cols,
                raw_transaction_dict=raw_dict,
                threshold=self.threshold,
                top_k=top_k,
                device=self.device
            )
            explanation_evidence = exp_result.get("Top_Contributing_Evidence", [])

        return {
            "transaction_id": txn_id,
            "fraud_probability": round(prob, 4),
            "decision_threshold": self.threshold,
            "decision": decision,
            "risk_tier": risk_tier,
            "top_contributing_evidence": explanation_evidence,
            "execution_time_ms": round(elapsed_ms, 2)
        }

    def predict_batch(
        self,
        df: pd.DataFrame,
        batch_size: int = 2048
    ) -> pd.DataFrame:
        """
        Run high-throughput batch scoring on an entire DataFrame of transactions.
        
        Args:
            df: DataFrame containing transaction records.
            batch_size: Inference batch size.
            
        Returns:
            Original DataFrame augmented with prediction columns:
            ['fraud_probability', 'decision', 'risk_tier'].
        """
        probs = self.predict_proba(df, batch_size=batch_size)
        
        result_df = df.copy()
        result_df["fraud_probability"] = np.round(probs, 4)
        result_df["decision"] = np.where(probs >= self.threshold, "FLAG FOR MANUAL REVIEW", "APPROVE")
        
        # Risk tiers
        risk_conditions = [
            probs >= 0.80,
            (probs >= self.threshold) & (probs < 0.80),
            probs < self.threshold
        ]
        risk_choices = [
            "CRITICAL / HIGH RISK",
            "ELEVATED RISK",
            "NORMAL / LOW RISK"
        ]
        result_df["risk_tier"] = np.select(risk_conditions, risk_choices, default="NORMAL / LOW RISK")
        
        return result_df


if __name__ == "__main__":
    print("Initializing FraudInferencePipeline test...")
    try:
        pipeline = FraudInferencePipeline.from_artifacts(
            model_path="models/pytorch_hybrid_weighted_bce.pt",
            preprocessor_path="models/tabular_preprocessor.joblib",
            threshold=0.80
        )
        print("Pipeline initialized successfully from artifacts.")
        
        # Test synthetic transaction
        sample_txn = {
            "TransactionID": 9999999,
            "TransactionAmt": 150.00,
            "ProductCD": "W",
            "card1": 17188,
            "card4": "visa",
            "P_emaildomain": "gmail.com",
            "DeviceInfo": "Windows"
        }
        res = pipeline.predict(sample_txn, include_explanation=True, top_k=3)
        print("Test prediction result:")
        print(res)
    except Exception as e:
        print(f"Test error: {e}")
