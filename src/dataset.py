"""
PyTorch Dataset and DataLoader Module for Financial Fraud Detection.
Wraps preprocessed numerical arrays, categorical integer tensors, and binary targets.
"""

import logging
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FraudDataset(Dataset):
    """
    PyTorch Dataset yielding (numerical_tensor, categorical_tensor, label_tensor).
    
    Attributes:
        x_num: torch.FloatTensor of shape [N, num_numerical_features]
        x_cat: torch.LongTensor of shape [N, num_categorical_features]
        y: Optional torch.FloatTensor of shape [N] (for BCEWithLogitsLoss)
    """
    def __init__(
        self,
        numerical_data: Union[np.ndarray, torch.Tensor],
        categorical_data: Union[np.ndarray, torch.Tensor],
        labels: Optional[Union[np.ndarray, torch.Tensor]] = None
    ):
        # 1. Numerical features: FloatTensor
        if isinstance(numerical_data, np.ndarray):
            self.x_num = torch.from_numpy(numerical_data).float()
        elif isinstance(numerical_data, torch.Tensor):
            self.x_num = numerical_data.float()
        else:
            raise TypeError(f"Unsupported type for numerical_data: {type(numerical_data)}")

        # 2. Categorical features: LongTensor (required for nn.Embedding lookups)
        if isinstance(categorical_data, np.ndarray):
            self.x_cat = torch.from_numpy(categorical_data).long()
        elif isinstance(categorical_data, torch.Tensor):
            self.x_cat = categorical_data.long()
        else:
            raise TypeError(f"Unsupported type for categorical_data: {type(categorical_data)}")

        # 3. Target labels: FloatTensor [N]
        if labels is not None:
            if isinstance(labels, np.ndarray):
                self.y = torch.from_numpy(labels).float()
            elif isinstance(labels, torch.Tensor):
                self.y = labels.float()
            else:
                raise TypeError(f"Unsupported type for labels: {type(labels)}")
                
            if self.y.ndim > 1:
                self.y = self.y.squeeze()
        else:
            self.y = None

        # Sanity check dimension alignment
        assert len(self.x_num) == len(self.x_cat), (
            f"Row count mismatch between numerical ({len(self.x_num)}) and categorical ({len(self.x_cat)})"
        )
        if self.y is not None:
            assert len(self.x_num) == len(self.y), (
                f"Row count mismatch between features ({len(self.x_num)}) and labels ({len(self.y)})"
            )

    def __len__(self) -> int:
        return len(self.x_num)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y_val = self.y[idx] if self.y is not None else torch.empty(0, dtype=torch.float32)
        return self.x_num[idx], self.x_cat[idx], y_val

    @property
    def num_numerical_features(self) -> int:
        return self.x_num.shape[1] if self.x_num.ndim > 1 else 0

    @property
    def num_categorical_features(self) -> int:
        return self.x_cat.shape[1] if self.x_cat.ndim > 1 else 0


def create_dataloaders(
    train_dataset: FraudDataset,
    val_dataset: Optional[FraudDataset] = None,
    test_dataset: Optional[FraudDataset] = None,
    batch_size: int = 512,
    num_workers: int = 0,
    pin_memory: bool = False
) -> Dict[str, DataLoader]:
    """
    Construct training, validation, and test PyTorch DataLoaders.
    
    Args:
        train_dataset: FraudDataset for model training (shuffled).
        val_dataset: Optional FraudDataset for validation (sequential).
        test_dataset: Optional FraudDataset for testing (sequential).
        batch_size: Number of transactions per mini-batch.
        num_workers: Subprocesses for data loading.
        pin_memory: If True, copies Tensors into CUDA pinned memory before returning.
        
    Returns:
        Dictionary mapping split names ('train', 'val', 'test') to DataLoader objects.
    """
    loaders: Dict[str, DataLoader] = {}

    loaders["train"] = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    logger.info(f"Created 'train' DataLoader: {len(train_dataset):,} samples, {len(loaders['train']):,} batches (batch_size={batch_size})")

    if val_dataset is not None:
        loaders["val"] = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
        logger.info(f"Created 'val' DataLoader: {len(val_dataset):,} samples, {len(loaders['val']):,} batches (batch_size={batch_size})")

    if test_dataset is not None:
        loaders["test"] = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
        logger.info(f"Created 'test' DataLoader: {len(test_dataset):,} samples, {len(loaders['test']):,} batches (batch_size={batch_size})")

    return loaders


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 3: PYTORCH DATASET & DATALOADER VERIFICATION")
    print("=" * 70)

    # 1. Synthesize mock numerical and categorical data for unit testing
    N = 1000
    D_num = 380
    D_cat = 49

    mock_num = np.random.randn(N, D_num).astype(np.float32)
    mock_cat = np.random.randint(0, 100, size=(N, D_cat), dtype=np.int64)
    mock_labels = np.random.choice([0.0, 1.0], size=N, p=[0.97, 0.03]).astype(np.float32)

    # 2. Build FraudDataset
    dataset = FraudDataset(mock_num, mock_cat, mock_labels)
    assert len(dataset) == N
    assert dataset.num_numerical_features == D_num
    assert dataset.num_categorical_features == D_cat

    # 3. Fetch single item
    item_num, item_cat, item_y = dataset[0]
    assert item_num.shape == (D_num,)
    assert item_cat.shape == (D_cat,)
    assert item_y.shape == ()
    assert item_num.dtype == torch.float32
    assert item_cat.dtype == torch.int64
    assert item_y.dtype == torch.float32

    # 4. Build DataLoader
    loaders = create_dataloaders(train_dataset=dataset, batch_size=64)
    train_loader = loaders["train"]

    # 5. Inspect batch
    batch_num, batch_cat, batch_y = next(iter(train_loader))
    print(f"\nBatch Shapes:")
    print(f"  Numerical:   {batch_num.shape} (dtype: {batch_num.dtype})")
    print(f"  Categorical: {batch_cat.shape} (dtype: {batch_cat.dtype})")
    print(f"  Labels:      {batch_y.shape} (dtype: {batch_y.dtype})")

    assert batch_num.shape == (64, D_num)
    assert batch_cat.shape == (64, D_cat)
    assert batch_y.shape == (64,)
    print("\n" + "=" * 70)
    print("DATASET & DATALOADER VERIFICATION PASSED!")
    print("=" * 70)
