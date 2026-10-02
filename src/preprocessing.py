"""
Data Preprocessing Module for Multimodal Financial Fraud Detector.
Phase 1: Safe ingestion, column normalization, left join, schema validation,
feature identification, and temporal splitting for IEEE-CIS data.
Phase 3: Leakage-safe TabularPreprocessor (imputation, scaling, frequency thresholding,
categorical vocabulary encoding, and artifact persistence).
"""

import gc
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# IEEE-CIS official known categorical columns
# Note: id_01 to id_11 are numerical; id_12 to id_38 are categorical.
IEEE_CIS_CATEGORICAL_COLS = [
    "ProductCD",
    "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2",
    "P_emaildomain", "R_emaildomain",
    "DeviceType", "DeviceInfo",
] + [f"M{i}" for i in range(1, 10)] + [f"id_{i}" for i in range(12, 39)]

RAW_IDENTITY_PATTERN = r"^id[-_](\d{2})$"


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Downcast numerical datatypes to optimize RAM usage.
    
    Args:
        df: Input pandas DataFrame.
        verbose: If True, logs initial and final memory footprints.
        
    Returns:
        DataFrame with downcasted memory types.
    """
    start_mem = df.memory_usage().sum() / (1024 ** 2)
    
    for col in df.columns:
        col_type = df[col].dtype
        
        # Skip string objects and categoricals for numeric downcasting
        if col_type != object and not isinstance(col_type, pd.CategoricalDtype):
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
                    
    end_mem = df.memory_usage().sum() / (1024 ** 2)
    if verbose:
        pct_reduction = 100 * (start_mem - end_mem) / start_mem if start_mem > 0 else 0
        logger.info(
            f"Memory reduced from {start_mem:.2f} MB to {end_mem:.2f} MB ({pct_reduction:.1f}% reduction)"
        )
        
    return df


def normalize_identity_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize identity column names by replacing hyphens with underscores.
    
    In raw IEEE-CIS data:
      - train_identity.csv uses underscores (e.g., id_01, id_02, ..., id_38)
      - test_identity.csv uses hyphens (e.g., id-01, id-02, ..., id-38)
      
    This function ensures all identity columns consistently use underscores.
    
    Args:
        df: Input DataFrame containing identity columns.
        
    Returns:
        DataFrame with standardized column names.
    """
    rename_dict = {}
    for col in df.columns:
        if col.startswith("id-"):
            rename_dict[col] = col.replace("id-", "id_")
            
    if rename_dict:
        logger.info(f"Normalized {len(rename_dict)} identity column names (e.g., 'id-01' -> 'id_01').")
        df = df.rename(columns=rename_dict)
        
    return df


def load_transaction_data(
    data_dir: Union[str, Path],
    split: str = "train",
    nrows: Optional[int] = None,
    reduce_memory: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """Load transaction CSV file for specified split."""
    data_dir = Path(data_dir)
    trans_path = data_dir / f"{split}_transaction.csv"
    if not trans_path.exists():
        raise FileNotFoundError(f"Transaction file not found: {trans_path}")
        
    if verbose:
        logger.info(f"Loading {trans_path.name} (nrows={nrows})...")
        
    df = pd.read_csv(trans_path, nrows=nrows)
    if reduce_memory:
        df = reduce_mem_usage(df, verbose=verbose)
        
    if verbose:
        logger.info(f"Successfully loaded {trans_path.name} with shape: {df.shape}")
        
    return df


def load_identity_data(
    data_dir: Union[str, Path],
    split: str = "train",
    nrows: Optional[int] = None,
    reduce_memory: bool = True,
    verbose: bool = True
) -> Optional[pd.DataFrame]:
    """Load identity CSV file for specified split and normalize column names."""
    data_dir = Path(data_dir)
    id_path = data_dir / f"{split}_identity.csv"
    if not id_path.exists():
        if verbose:
            logger.warning(f"Identity file not found: {id_path}")
        return None
        
    if verbose:
        logger.info(f"Loading {id_path.name} (nrows={nrows})...")
        
    df = pd.read_csv(id_path, nrows=nrows)
    df = normalize_identity_columns(df)
    
    if reduce_memory:
        df = reduce_mem_usage(df, verbose=verbose)
        
    if verbose:
        logger.info(f"Successfully loaded {id_path.name} with shape: {df.shape}")
        
    return df


def load_and_merge_data(
    data_dir: Union[str, Path],
    split: str = "train",
    nrows: Optional[int] = None,
    reduce_memory: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load transaction and identity tables for IEEE-CIS, normalize identity column names,
    and perform a safe left join on TransactionID.
    """
    trans_df = load_transaction_data(
        data_dir=data_dir,
        split=split,
        nrows=nrows,
        reduce_memory=reduce_memory,
        verbose=verbose
    )
    
    id_df = load_identity_data(
        data_dir=data_dir,
        split=split,
        nrows=nrows,
        reduce_memory=reduce_memory,
        verbose=verbose
    )
    
    if id_df is not None:
        dup_id_count = id_df["TransactionID"].duplicated().sum()
        if dup_id_count > 0:
            logger.warning(f"Found {dup_id_count} duplicate TransactionIDs in {split}_identity. Deduplicating...")
            id_df = id_df.drop_duplicates(subset=["TransactionID"], keep="first")
            
        if verbose:
            logger.info(f"Merging {split} transaction ({len(trans_df)} rows) and identity ({len(id_df)} rows) on TransactionID...")
            
        merged_df = trans_df.merge(id_df, on="TransactionID", how="left")
        
        match_count = merged_df["DeviceType"].notna().sum() if "DeviceType" in merged_df.columns else len(id_df)
        match_pct = (match_count / len(merged_df)) * 100.0 if len(merged_df) > 0 else 0.0
        
        if verbose:
            logger.info(f"Merged dataset shape: {merged_df.shape}. Identity match rate: {match_pct:.2f}% ({match_count}/{len(merged_df)})")
            
        del trans_df, id_df
        gc.collect()
        return merged_df
        
    return trans_df


def validate_schema(df: pd.DataFrame, split: str = "train") -> Dict[str, Any]:
    """Validate schema and data integrity of the merged dataset."""
    logger.info(f"--- Running Schema Validation for '{split}' dataset ---")
    results: Dict[str, Any] = {
        "split": split,
        "num_rows": len(df),
        "num_cols": len(df.columns),
        "is_valid": True,
        "warnings": [],
        "errors": []
    }
    
    # 1. Primary Key Check
    if "TransactionID" not in df.columns:
        err = "CRITICAL: 'TransactionID' column is missing."
        logger.error(err)
        results["errors"].append(err)
        results["is_valid"] = False
    else:
        num_duplicates = int(df["TransactionID"].duplicated().sum())
        results["duplicate_transaction_ids"] = num_duplicates
        if num_duplicates > 0:
            err = f"CRITICAL: Found {num_duplicates} duplicate TransactionIDs in dataset."
            logger.error(err)
            results["errors"].append(err)
            results["is_valid"] = False
            
    # 2. Core Transaction Features
    for col in ["TransactionDT", "TransactionAmt"]:
        if col not in df.columns:
            err = f"CRITICAL: Core column '{col}' is missing."
            logger.error(err)
            results["errors"].append(err)
            results["is_valid"] = False
            
    # 3. Target Check (if train)
    if split == "train":
        if "isFraud" not in df.columns:
            err = "CRITICAL: Target column 'isFraud' is missing from training set."
            logger.error(err)
            results["errors"].append(err)
            results["is_valid"] = False
        else:
            unique_targets = set(df["isFraud"].dropna().unique().tolist())
            if not unique_targets.issubset({0, 1, 0.0, 1.0}):
                err = f"CRITICAL: 'isFraud' contains invalid non-binary values: {unique_targets}"
                logger.error(err)
                results["errors"].append(err)
                results["is_valid"] = False
                
            nan_targets = int(df["isFraud"].isna().sum())
            if nan_targets > 0:
                err = f"CRITICAL: 'isFraud' contains {nan_targets} NaN values."
                logger.error(err)
                results["errors"].append(err)
                results["is_valid"] = False
                
            fraud_counts = df["isFraud"].value_counts().to_dict()
            fraud_rate = float(df["isFraud"].mean())
            results["target_distribution"] = fraud_counts
            results["fraud_rate"] = fraud_rate
            logger.info(f"Target distribution: {fraud_counts} (Fraud rate: {fraud_rate * 100:.2f}%)")
            
    # 4. Check for Hyphens in columns
    hyphen_cols = [c for c in df.columns if "-" in c]
    if hyphen_cols:
        warn = f"WARNING: Found {len(hyphen_cols)} unnormalized column names with hyphens: {hyphen_cols[:5]}"
        logger.warning(warn)
        results["warnings"].append(warn)
        
    results["memory_mb"] = round(df.memory_usage().sum() / (1024 ** 2), 2)
    return results


def get_feature_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Categorize columns into Target, Identifiers, Timestamp, Categorical, and Numerical features."""
    all_cols = df.columns.tolist()
    
    target_col = ["isFraud"] if "isFraud" in all_cols else []
    id_cols = [c for c in ["TransactionID"] if c in all_cols]
    time_cols = [c for c in ["TransactionDT"] if c in all_cols]
    
    cat_cols = [c for c in IEEE_CIS_CATEGORICAL_COLS if c in all_cols]
    reserved = set(target_col + id_cols + time_cols + cat_cols)
    for c in all_cols:
        if c not in reserved:
            if df[c].dtype == object or isinstance(df[c].dtype, pd.CategoricalDtype):
                cat_cols.append(c)
                
    reserved = set(target_col + id_cols + time_cols + cat_cols)
    num_cols = [c for c in all_cols if c not in reserved]
    
    return {
        "target": target_col,
        "id": id_cols,
        "timestamp": time_cols,
        "categorical": cat_cols,
        "numerical": num_cols,
    }


def compute_missing_and_cardinality(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Generate missing percentage and unique value count summary table for given columns."""
    if columns is None:
        columns = df.columns.tolist()
        
    summary = []
    total_rows = len(df)
    
    for col in columns:
        missing_count = int(df[col].isna().sum())
        missing_pct = (missing_count / total_rows) * 100.0 if total_rows > 0 else 0.0
        n_unique = int(df[col].nunique(dropna=True))
        dtype_str = str(df[col].dtype)
        
        summary.append({
            "Feature": col,
            "Dtype": dtype_str,
            "Missing_Count": missing_count,
            "Missing_Pct": round(missing_pct, 2),
            "Unique_Values": n_unique
        })
        
    return pd.DataFrame(summary).sort_values("Missing_Pct", ascending=False).reset_index(drop=True)


def temporal_train_val_test_split(
    df: pd.DataFrame,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    time_col: str = "TransactionDT"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform a strict, chronological temporal split of the dataset."""
    if time_col not in df.columns:
        raise KeyError(f"Timestamp column '{time_col}' not found in DataFrame.")
        
    if val_ratio <= 0 or test_ratio <= 0 or (val_ratio + test_ratio) >= 1.0:
        raise ValueError(f"Invalid split ratios: val={val_ratio}, test={test_ratio}. Sum must be < 1.0.")
        
    logger.info(f"Sorting dataset by '{time_col}' for strict chronological temporal splitting...")
    df_sorted = df.sort_values(by=time_col).reset_index(drop=True)
    
    n_total = len(df_sorted)
    train_end_idx = int(n_total * (1.0 - val_ratio - test_ratio))
    val_end_idx = int(n_total * (1.0 - test_ratio))
    
    train_df = df_sorted.iloc[:train_end_idx].copy().reset_index(drop=True)
    val_df = df_sorted.iloc[train_end_idx:val_end_idx].copy().reset_index(drop=True)
    test_df = df_sorted.iloc[val_end_idx:].copy().reset_index(drop=True)
    
    def log_split_stats(name: str, split_df: pd.DataFrame) -> None:
        t_min = split_df[time_col].min()
        t_max = split_df[time_col].max()
        days_span = (t_max - t_min) / 86400.0 if t_max is not None and t_min is not None else 0
        fraud_str = ""
        if "isFraud" in split_df.columns:
            fraud_count = split_df["isFraud"].sum()
            fraud_pct = split_df["isFraud"].mean() * 100
            fraud_str = f", Frauds={fraud_count} ({fraud_pct:.2f}%)"
            
        logger.info(
            f"  [{name.upper()}] Rows: {len(split_df):,} ({len(split_df)/n_total*100:.1f}%) | "
            f"DT: [{t_min:,} - {t_max:,}] (~{days_span:.1f} days){fraud_str}"
        )
        
    logger.info("Temporal Split Completed Successfully:")
    log_split_stats("Train", train_df)
    log_split_stats("Val", val_df)
    log_split_stats("Test", test_df)
    
    return train_df, val_df, test_df


# =====================================================================
# PHASE 3: LEAKAGE-SAFE TABULAR PREPROCESSOR
# =====================================================================

class TabularPreprocessor:
    """
    Leakage-Safe Preprocessing Pipeline for Tabular / Hybrid Fraud Detection.
    
    Strict Design Principles:
      1. Zero Lookahead Contamination: All parameters (medians, StandardScaler statistics,
         and categorical vocabulary indices) are fitted strictly on `train_df`.
      2. Missingness as an Informative Signal:
         - Categorical missing values and unseen categories map to index 0 (`<UNK>`).
         - Numerical missing values are imputed with training set medians.
      3. Frequency Thresholding: Categories appearing fewer than `min_freq` times in the
         training split are mapped to index 0 (`<UNK>`) to prevent embedding table explosion.
      4. Numerical Skew Treatment: Applies `np.log1p(TransactionAmt)` before scaling.
      5. Full Serialization: Can be saved to disk and restored for production/inference.
    """
    def __init__(
        self,
        min_freq: int = 10,
        log_transform_amt: bool = True,
        categorical_cols: Optional[List[str]] = None,
        numerical_cols: Optional[List[str]] = None,
        drop_zero_variance: bool = True
    ):
        self.min_freq = min_freq
        self.log_transform_amt = log_transform_amt
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        self.drop_zero_variance = drop_zero_variance
        
        # Fitted state
        self.fitted_categorical_cols_: List[str] = []
        self.fitted_numerical_cols_: List[str] = []
        self.dropped_cols_: List[str] = []
        self.medians_: Dict[str, float] = {}
        self.scaler_: Optional[StandardScaler] = None
        self.vocabularies_: Dict[str, Dict[str, int]] = {}
        self.vocab_sizes_: Dict[str, int] = {}
        self.is_fitted_: bool = False

    def fit(self, train_df: pd.DataFrame) -> "TabularPreprocessor":
        """
        Fit numerical imputers, scalers, and categorical vocabularies strictly on train_df.
        """
        logger.info("Fitting TabularPreprocessor strictly on training data (Leakage-Safe)...")
        
        # Determine features if not explicitly provided
        groups = get_feature_groups(train_df)
        cat_candidates = self.categorical_cols if self.categorical_cols is not None else groups["categorical"]
        num_candidates = self.numerical_cols if self.numerical_cols is not None else groups["numerical"]
        
        # 1. Feature Selection: Filter out non-features (IDs, timestamps, targets)
        excluded = {"TransactionID", "TransactionDT", "isFraud", "Day"}
        cat_cols = [c for c in cat_candidates if c in train_df.columns and c not in excluded]
        num_cols = [c for c in num_candidates if c in train_df.columns and c not in excluded]
        
        # 2. Drop constant / zero-variance numerical features
        if self.drop_zero_variance and len(num_cols) > 0:
            num_stds = train_df[num_cols].std()
            zero_var_cols = num_stds[num_stds == 0].index.tolist()
            if zero_var_cols:
                logger.info(f"Dropping {len(zero_var_cols)} zero-variance numerical features: {zero_var_cols}")
                num_cols = [c for c in num_cols if c not in zero_var_cols]
                self.dropped_cols_.extend(zero_var_cols)
                
        self.fitted_categorical_cols_ = cat_cols
        self.fitted_numerical_cols_ = num_cols
        logger.info(f"Selected {len(self.fitted_numerical_cols_)} numerical and {len(self.fitted_categorical_cols_)} categorical features.")
        
        # 3. Fit Numerical Preprocessing (Medians + StandardScaler)
        if len(self.fitted_numerical_cols_) > 0:
            num_data = train_df[self.fitted_numerical_cols_].copy()
            if self.log_transform_amt and "TransactionAmt" in num_data.columns:
                num_data["TransactionAmt"] = np.log1p(num_data["TransactionAmt"].clip(lower=0))
                
            # Compute medians strictly on train
            train_medians = num_data.median()
            # If any median is NaN (e.g. 100% missing column), fill with 0.0
            self.medians_ = train_medians.fillna(0.0).to_dict()
            
            # Impute train numericals
            num_imputed = num_data.fillna(self.medians_)
            
            # Fit StandardScaler strictly on train
            self.scaler_ = StandardScaler()
            self.scaler_.fit(num_imputed)
            logger.info("Successfully fitted StandardScaler on imputed training numericals.")
            
        # 4. Fit Categorical Vocabularies with Frequency Thresholding
        for col in self.fitted_categorical_cols_:
            # Cast to string, treat NaNs as string '__MISSING__'
            s = train_df[col].fillna("__MISSING__").astype(str)
            val_counts = s.value_counts()
            
            # Keep categories with frequency >= min_freq
            valid_cats = val_counts[val_counts >= self.min_freq].index.tolist()
            
            # Index 0 is reserved for <UNK> / <MISSING> / rare categories
            # Contiguous indices start from 1
            vocab = {}
            for idx, cat_val in enumerate(valid_cats, start=1):
                vocab[cat_val] = idx
                
            self.vocabularies_[col] = vocab
            # vocab_size is number of valid categories + 1 (for index 0)
            self.vocab_sizes_[col] = len(vocab) + 1
            
        logger.info(f"Built vocabularies for {len(self.vocabularies_)} categorical features (min_freq={self.min_freq}).")
        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Transform a DataFrame using fitted parameters.
        
        Returns:
            X_num: np.ndarray [N, D_num] (float32)
            X_cat: np.ndarray [N, D_cat] (int64)
            y: Optional[np.ndarray] [N] (float32) if 'isFraud' exists in df, else None
        """
        if not self.is_fitted_:
            raise RuntimeError("TabularPreprocessor must be fitted before calling transform().")
            
        n_samples = len(df)
        
        # 1. Numerical Transformation
        if len(self.fitted_numerical_cols_) > 0:
            num_data = df[self.fitted_numerical_cols_].copy()
            if self.log_transform_amt and "TransactionAmt" in num_data.columns:
                num_data["TransactionAmt"] = np.log1p(num_data["TransactionAmt"].clip(lower=0))
                
            # Impute using train medians
            num_imputed = num_data.fillna(self.medians_)
            
            # Transform using train scaler
            X_num = self.scaler_.transform(num_imputed).astype(np.float32)
        else:
            X_num = np.empty((n_samples, 0), dtype=np.float32)
            
        # 2. Categorical Transformation (Vectorized)
        cat_arrays = []
        for col in self.fitted_categorical_cols_:
            s = df[col].fillna("__MISSING__").astype(str)
            vocab = self.vocabularies_[col]
            # Map category strings to integers; unmapped (rare/unseen/missing) default to 0
            encoded = s.map(vocab).fillna(0).astype(np.int64).values
            cat_arrays.append(encoded)
            
        if cat_arrays:
            X_cat = np.column_stack(cat_arrays).astype(np.int64)
        else:
            X_cat = np.empty((n_samples, 0), dtype=np.int64)
            
        # 3. Target Extraction (if present)
        y = None
        if "isFraud" in df.columns:
            y = df["isFraud"].values.astype(np.float32)
            
        return X_num, X_cat, y

    def fit_transform(self, train_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Fit on train_df and return transformed arrays."""
        return self.fit(train_df).transform(train_df)

    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize preprocessor state to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Saved TabularPreprocessor artifact to: {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "TabularPreprocessor":
        """Load serialized preprocessor state from disk."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Preprocessor file not found at: {filepath}")
        obj = joblib.load(filepath)
        logger.info(f"Loaded TabularPreprocessor artifact from: {filepath}")
        return obj


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test and validate Phase 1 and Phase 3 preprocessing pipeline.")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Path to raw data directory.")
    parser.add_argument("--nrows", type=int, default=50000, help="Number of rows to test with (None for full).")
    args = parser.parse_args()
    
    print("=" * 70)
    print("PHASE 1 & PHASE 3 PREPROCESSING VERIFICATION")
    print("=" * 70)
    
    # 1. Ingestion & Temporal Splitting
    df = load_and_merge_data(data_dir=args.data_dir, split="train", nrows=args.nrows, reduce_memory=True)
    train_split, val_split, test_split = temporal_train_val_test_split(df, val_ratio=0.15, test_ratio=0.15)
    
    # 2. Fit Preprocessor STRICTLY on Train
    preprocessor = TabularPreprocessor(min_freq=10, log_transform_amt=True)
    preprocessor.fit(train_split)
    
    # 3. Transform Splits
    X_num_train, X_cat_train, y_train = preprocessor.transform(train_split)
    X_num_val, X_cat_val, y_val = preprocessor.transform(val_split)
    X_num_test, X_cat_test, y_test = preprocessor.transform(test_split)
    
    print("\n--- Array Shape & Quality Diagnostics ---")
    print(f"Train Numerical:   {X_num_train.shape} (dtype: {X_num_train.dtype}, NaNs: {np.isnan(X_num_train).sum()})")
    print(f"Train Categorical: {X_cat_train.shape} (dtype: {X_cat_train.dtype}, Range: [{X_cat_train.min()}, {X_cat_train.max()}])")
    print(f"Train Labels:      {y_train.shape} (dtype: {y_train.dtype}, Fraud rate: {y_train.mean():.4f})")
    
    print(f"\nVal Numerical:     {X_num_val.shape} (NaNs: {np.isnan(X_num_val).sum()})")
    print(f"Val Categorical:   {X_cat_val.shape} (Range: [{X_cat_val.min()}, {X_cat_val.max()}])")
    print(f"Val Labels:        {y_val.shape} (Fraud rate: {y_val.mean():.4f})")
    
    print(f"\nTest Numerical:    {X_num_test.shape} (NaNs: {np.isnan(X_num_test).sum()})")
    print(f"Test Categorical:  {X_cat_test.shape} (Range: [{X_cat_test.min()}, {X_cat_test.max()}])")
    print(f"Test Labels:       {y_test.shape} (Fraud rate: {y_test.mean():.4f})")
    
    # 4. Test Preprocessor Serialization & Deserialization
    save_path = Path("models/tabular_preprocessor.joblib")
    preprocessor.save(save_path)
    loaded_prep = TabularPreprocessor.load(save_path)
    
    X_num_test_loaded, X_cat_test_loaded, _ = loaded_prep.transform(test_split)
    np.testing.assert_allclose(X_num_test, X_num_test_loaded)
    np.testing.assert_array_equal(X_cat_test, X_cat_test_loaded)
    print("\nSerialization check: Loaded preprocessor yields byte-identical transformations!")
    
    print("\n" + "=" * 70)
    print("ALL PHASE 3 PREPROCESSOR CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)
