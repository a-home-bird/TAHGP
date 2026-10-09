import torch
import torch.nn as nn


class DepositClassificationHead(nn.Module):
    """MLP head for deposit classification."""

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        hidden_dim: int,
        dropout: float = 0.2,
    ):
        super().__init__()

        hidden_dim = hidden_dim or embedding_dim

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        deposit_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        return self.classifier(deposit_embeddings)