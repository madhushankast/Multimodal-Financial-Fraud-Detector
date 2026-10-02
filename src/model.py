"""
PyTorch Model Architecture Module for Financial Fraud Detection.
Phase 5: Numerical MLP Baseline (dense layers, batch normalization, dropout).
Phase 6: Hybrid Multimodal Fraud Detector with Categorical Entity Embeddings.
"""

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn


def compute_embedding_dim(vocab_size: int, max_dim: int = 50, min_dim: int = 4) -> int:
    """
    Standard empirical rule-of-thumb for categorical embedding dimension:
      dim = min(max_dim, max(min_dim, round(6 * (vocab_size ** 0.25))))
      
    Args:
        vocab_size: Number of unique categories (including index 0 for <UNK>).
        max_dim: Maximum allowable embedding dimension to cap parameter growth.
        min_dim: Minimum embedding dimension to retain sufficient capacity.
        
    Returns:
        Integer embedding dimension.
    """
    return min(max_dim, max(min_dim, int(round(6 * (vocab_size ** 0.25)))))


class NumericalMLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for numerical tabular fraud detection.
    
    Architecture:
        Input [N, in_features]
          ↓
        Linear(in_features, 256) -> BatchNorm1d(256) -> ReLU -> Dropout(0.30)
          ↓
        Linear(256, 128) -> BatchNorm1d(128) -> ReLU -> Dropout(0.30)
          ↓
        Linear(128, 64) -> BatchNorm1d(64) -> ReLU -> Dropout(0.20)
          ↓
        Linear(64, 1)
          ↓
        Raw Fraud Logit [N]
    """
    def __init__(
        self,
        in_features: int,
        hidden_dims: tuple = (256, 128, 64),
        dropout_rates: tuple = (0.3, 0.3, 0.2),
        use_batch_norm: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        
        layers = []
        prev_dim = in_features
        
        for h_dim, drop_rate in zip(hidden_dims, dropout_rates):
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            if drop_rate > 0:
                layers.append(nn.Dropout(drop_rate))
            prev_dim = h_dim
            
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x_num: torch.Tensor, x_cat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for numerical MLP.
        x_cat is optional and ignored in this numerical-only model.
        """
        logits = self.network(x_num)
        return logits.squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, x_num: torch.Tensor, x_cat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute fraud probabilities in [0.0, 1.0]."""
        self.eval()
        logits = self.forward(x_num, x_cat)
        return torch.sigmoid(logits)


class HybridFraudDetector(nn.Module):
    """
    Hybrid Neural Network for Multimodal / Tabular Financial Fraud Detection.
    Combines numerical transaction features with categorical entity embeddings.
    
    Architecture:
        Categorical inputs [N, K] -> 49 Embedding tables -> Concatenate embeddings [N, D_emb]
        Numerical inputs [N, D_num] -> BatchNorm1d [N, D_num]
                              ↓
              Concatenate: [Numerical, Embeddings] [N, D_num + D_emb]
                              ↓
              Linear(D_total, 256) -> BatchNorm1d -> ReLU -> Dropout(0.30)
                              ↓
              Linear(256, 128)    -> BatchNorm1d -> ReLU -> Dropout(0.30)
                              ↓
              Linear(128, 64)     -> BatchNorm1d -> ReLU -> Dropout(0.20)
                              ↓
              Linear(64, 1)       -> Raw Fraud Logit [N]
    """
    def __init__(
        self,
        num_numerical_features: int,
        categorical_cardinalities: List[int],
        embedding_dims: Optional[List[int]] = None,
        hidden_dims: tuple = (256, 128, 64),
        dropout_rates: tuple = (0.3, 0.3, 0.2),
        use_batch_norm: bool = True
    ):
        super().__init__()
        self.num_numerical_features = num_numerical_features
        self.categorical_cardinalities = categorical_cardinalities
        
        # 1. Compute embedding dimensions per categorical feature
        if embedding_dims is None:
            self.embedding_dims = [compute_embedding_dim(c) for c in categorical_cardinalities]
        else:
            self.embedding_dims = embedding_dims
            
        # 2. Embedding layers (padding_idx=0 maps <UNK>/<MISSING> to zero vector)
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim, padding_idx=0)
            for vocab_size, emb_dim in zip(categorical_cardinalities, self.embedding_dims)
        ])
        
        total_emb_dim = sum(self.embedding_dims)
        
        # 3. Batch Normalization for raw numerical features
        self.num_bn = nn.BatchNorm1d(num_numerical_features) if use_batch_norm and num_numerical_features > 0 else nn.Identity()
        
        # 4. Dense MLP Layers
        in_dense_dim = num_numerical_features + total_emb_dim
        layers = []
        prev_dim = in_dense_dim
        
        for h_dim, drop_rate in zip(hidden_dims, dropout_rates):
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            if drop_rate > 0:
                layers.append(nn.Dropout(drop_rate))
            prev_dim = h_dim
            
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for hybrid embedding network.
        
        Args:
            x_num: FloatTensor of shape [batch_size, num_numerical_features]
            x_cat: LongTensor of shape [batch_size, num_categorical_features]
            
        Returns:
            Raw logits of shape [batch_size]
        """
        # Embed each categorical feature
        embedded = []
        for i, emb_layer in enumerate(self.embeddings):
            col_emb = emb_layer(x_cat[:, i])  # [N, emb_dim_i]
            embedded.append(col_emb)
            
        cat_repr = torch.cat(embedded, dim=1)  # [N, total_emb_dim]
        num_repr = self.num_bn(x_num)          # [N, num_numerical_features]
        
        # Concatenate numerical and categorical representations
        fused = torch.cat([num_repr, cat_repr], dim=1)
        logits = self.mlp(fused).squeeze(-1)
        return logits

    @torch.no_grad()
    def predict_proba(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """Compute fraud probabilities in [0.0, 1.0]."""
        self.eval()
        logits = self.forward(x_num, x_cat)
        return torch.sigmoid(logits)


if __name__ == "__main__":
    print("=" * 60)
    print("VERIFYING HYBRID FRAUD DETECTOR ARCHITECTURE")
    print("=" * 60)
    
    batch_size = 16
    D_num = 381
    cat_cardinalities = [5, 4, 4, 2, 60, 866, 120, 10, 15]  # mock cardinalities
    
    hybrid_model = HybridFraudDetector(
        num_numerical_features=D_num,
        categorical_cardinalities=cat_cardinalities
    )
    
    x_num = torch.randn(batch_size, D_num, dtype=torch.float32)
    x_cat = torch.column_stack([
        torch.randint(0, card, size=(batch_size,), dtype=torch.long)
        for card in cat_cardinalities
    ])
    
    logits = hybrid_model(x_num, x_cat)
    probs = hybrid_model.predict_proba(x_num, x_cat)
    
    print(f"Numerical shape:    {x_num.shape}")
    print(f"Categorical shape:  {x_cat.shape}")
    print(f"Total Emb Dim:      {sum(hybrid_model.embedding_dims)}")
    print(f"Output logit shape: {logits.shape}")
    print(f"Output prob shape:  {probs.shape}")
    
    assert logits.shape == (batch_size,)
    assert probs.shape == (batch_size,)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    
    total_params = sum(p.numel() for p in hybrid_model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")
    print("=" * 60)
    print("HYBRID ARCHITECTURE VERIFICATION PASSED!")
    print("=" * 60)
