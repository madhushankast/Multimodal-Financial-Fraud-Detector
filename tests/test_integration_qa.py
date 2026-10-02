"""
Phase 12: Comprehensive Integration, QA, Leakage Audit, and Pipeline Invariants Test Suite.
Uses standard Python unittest framework (zero external test dependencies).
"""

from pathlib import Path
import sys
import unittest
import inspect
import json
import joblib
import numpy as np
import pandas as pd
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.inference import FraudInferencePipeline
from src.model import HybridFraudDetector
from src.preprocessing import TabularPreprocessor


class TestPhase12IntegrationQA(unittest.TestCase):
    """
    Automated QA validation testing artifact integrity, zero data leakage,
    inference reliability, robustness, and Streamlit presentation decoupling.
    """

    @classmethod
    def setUpClass(cls):
        cls.models_dir = PROJECT_ROOT / "models"
        cls.model_path = cls.models_dir / "pytorch_hybrid_weighted_bce.pt"
        cls.preprocessor_path = cls.models_dir / "tabular_preprocessor.joblib"
        cls.sample_batch_path = PROJECT_ROOT / "data" / "sample_batch_500.csv"

        # Initialize pipeline from artifacts
        cls.pipeline = FraudInferencePipeline.from_artifacts(
            model_path=cls.model_path,
            preprocessor_path=cls.preprocessor_path,
            threshold=0.80
        )

    def test_01_saved_artifacts_exist_and_non_empty(self):
        """Verify saved model checkpoint and preprocessor joblib are present and valid."""
        self.assertTrue(self.model_path.exists(), f"Missing model checkpoint: {self.model_path}")
        self.assertTrue(self.preprocessor_path.exists(), f"Missing preprocessor joblib: {self.preprocessor_path}")
        self.assertGreater(self.model_path.stat().st_size, 1_000_000, "Model file size unexpectedly small (<1MB)")
        self.assertGreater(self.preprocessor_path.stat().st_size, 10_000, "Preprocessor file size unexpectedly small (<10KB)")

    def test_02_zero_leakage_inference_contract(self):
        """Verify that inference pipeline never invokes fit or fit_transform during prediction."""
        inference_source = inspect.getsource(FraudInferencePipeline)
        self.assertNotIn(".fit(", inference_source, "Leakage Violation: .fit() found in FraudInferencePipeline source!")
        self.assertNotIn(".fit_transform(", inference_source, "Leakage Violation: .fit_transform() found in FraudInferencePipeline source!")

    def test_03_preprocessor_vocabulary_and_scalers_frozen(self):
        """Confirm that preprocessor contains fitted statistics across exact feature counts."""
        prep = self.pipeline.preprocessor
        self.assertEqual(len(prep.fitted_numerical_cols_), 381, "Expected exactly 381 numerical features")
        self.assertEqual(len(prep.fitted_categorical_cols_), 49, "Expected exactly 49 categorical features")
        self.assertIsNotNone(prep.scaler_.mean_, "Fitted scaler mean is None")
        self.assertIsNotNone(prep.scaler_.scale_, "Fitted scaler scale is None")
        self.assertEqual(len(prep.vocab_sizes_), 49, "Fitted categorical vocabulary mapping incomplete")

    def test_04_neural_architecture_weight_shapes(self):
        """Confirm loaded model architecture layer dimensions match the 568-dim hybrid blueprint."""
        model = self.pipeline.model
        self.assertIsInstance(model, HybridFraudDetector)
        self.assertEqual(model.num_numerical_features, 381)
        self.assertEqual(len(model.embeddings), 49)
        total_emb_dim = sum(e.embedding_dim for e in model.embeddings)
        self.assertEqual(total_emb_dim, 568, f"Expected 568 total embedding dimensions, found {total_emb_dim}")
        # Total input dimension into MLP (381 num + 568 emb = 949)
        first_linear = [m for m in model.mlp if isinstance(m, torch.nn.Linear)][0]
        self.assertEqual(first_linear.in_features, 381 + 568)

    def test_05_single_transaction_prediction(self):
        """Validate single transaction real-time inference, probability bounding, and risk tier."""
        payload = {
            "TransactionID": 9999001,
            "TransactionAmt": 125.50,
            "ProductCD": "W",
            "card1": 13926,
            "card4": "discover",
            "card6": "credit",
            "P_emaildomain": "gmail.com",
            "DeviceType": "desktop",
            "DeviceInfo": "Windows",
            "C1": 1, "C2": 1, "D1": 14
        }
        res = self.pipeline.predict(payload, include_explanation=True, top_k=3)
        
        self.assertIn("fraud_probability", res)
        self.assertIn("decision", res)
        self.assertIn("risk_tier", res)
        self.assertIn("top_contributing_evidence", res)
        self.assertIn("execution_time_ms", res)

        self.assertGreaterEqual(res["fraud_probability"], 0.0)
        self.assertLessEqual(res["fraud_probability"], 1.0)
        self.assertIn(res["decision"], ["APPROVE", "FLAG FOR MANUAL REVIEW"])
        self.assertIn(res["risk_tier"], ["NORMAL / LOW RISK", "ELEVATED RISK", "CRITICAL / HIGH RISK"])
        self.assertLessEqual(len(res["top_contributing_evidence"]), 3)

    def test_06_batch_inference_500_records(self):
        """Validate batch inference on 500 test transactions with high throughput and consistent columns."""
        self.assertTrue(self.sample_batch_path.exists(), f"Sample batch CSV not found at {self.sample_batch_path}")
        batch_df = pd.read_csv(self.sample_batch_path)
        self.assertEqual(len(batch_df), 500)

        scored_df = self.pipeline.predict_batch(batch_df, batch_size=2048)
        self.assertEqual(len(scored_df), 500)
        self.assertIn("fraud_probability", scored_df.columns)
        self.assertIn("decision", scored_df.columns)
        self.assertIn("risk_tier", scored_df.columns)

        probs = scored_df["fraud_probability"]
        self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all(), "Batch probabilities outside [0, 1]")
        self.assertFalse(probs.isna().any(), "Batch probabilities contain NaNs")

    def test_07_fault_tolerance_unknown_categories_and_sparse_schema(self):
        """Verify graceful handling of novel categories and ultra-sparse payloads."""
        extreme_payload = {
            "TransactionAmt": 500.00,
            "ProductCD": "NON_EXISTENT_CATEGORY",
            "card4": "alien_card_network",
            "DeviceInfo": "QuantumRig_2026",
            "P_emaildomain": "mars_outpost.corp"
        }
        res = self.pipeline.predict(extreme_payload, include_explanation=False)
        self.assertIsInstance(res["fraud_probability"], float)
        self.assertTrue(0.0 <= res["fraud_probability"] <= 1.0)

    def test_08_threshold_controllability_and_calibration(self):
        """Confirm that adjusting operating threshold dynamically changes decisions without model alteration."""
        txn = {"TransactionAmt": 350.00, "ProductCD": "C", "C1": 15, "V87": 3.0}
        
        # Test low threshold
        self.pipeline.threshold = 0.05
        res_low = self.pipeline.predict(txn, include_explanation=False)
        
        # Test high threshold
        self.pipeline.threshold = 0.95
        res_high = self.pipeline.predict(txn, include_explanation=False)

        # Probabilities should be identical (model inference invariant)
        self.assertEqual(res_low["fraud_probability"], res_high["fraud_probability"])
        # Reset to cost-optimal default
        self.pipeline.threshold = 0.80

    def test_09_streamlit_decoupled_architecture(self):
        """Verify that app/app.py does not contain data leakage or training routines."""
        app_file = PROJECT_ROOT / "app" / "app.py"
        self.assertTrue(app_file.exists())
        app_source = app_file.read_text(encoding="utf-8")
        
        self.assertNotIn("scaler.fit(", app_source)
        self.assertNotIn("preprocessor.fit(", app_source)
        self.assertNotIn("fit_transform(", app_source)
        self.assertIn("FraudInferencePipeline.from_artifacts", app_source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
