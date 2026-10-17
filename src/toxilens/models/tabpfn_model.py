from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


class TabularMLP(nn.Module):
    """
    Lightweight, fast Tabular Neural Network with LayerNorm and Dropout
    for physicochemical descriptor classification.
    """

    def __init__(self, in_features: int = 10, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TabularDescriptorModel:
    """
    Tabular baseline model on RDKit physicochemical descriptors.
    Acts as the offline tabular descriptor benchmark for model comparison.
    """

    def __init__(self, in_features: int = 10, hidden_dim: int = 64):
        self.scaler = StandardScaler()
        self.model = TabularMLP(in_features=in_features, hidden_dim=hidden_dim)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 15, lr: float = 0.01) -> "TabularDescriptorModel":
        X_scaled = self.scaler.fit_transform(X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        # Class balancing weight
        n_pos = float(np.sum(y == 1))
        n_neg = float(np.sum(y == 0))
        pos_weight = torch.tensor([n_neg / n_pos]) if n_pos > 0 else torch.tensor([1.0])

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            logits = self.model(X_t)
            loss = criterion(logits, y_t)
            loss.backward()
            optimizer.step()

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(X_t)
            probs = torch.sigmoid(logits).numpy()
        return probs


def train_tabpfn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_train_samples: int = 1000,
    random_state: int = 42,
):
    """
    Train tabular model on descriptors.
    """
    model = TabularDescriptorModel(in_features=X_train.shape[1], hidden_dim=64)
    model.fit(X_train, y_train)
    return model


def predict_tabpfn_proba(model, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """
    Predict toxicity probabilities for tabular descriptors.
    """
    return model.predict_proba(X)
