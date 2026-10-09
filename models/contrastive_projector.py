import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple,Dict

class ContrastiveProjector(nn.Module):
    """Projection heads for anchor-view contrastive learning."""

    def __init__(
        self,
        embedding_dim: int,
        projection_dim: int = 64,
    ):
        super().__init__()

        self.anchor_projector = nn.Sequential(
            nn.Linear(
                embedding_dim,
                embedding_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                embedding_dim,
                projection_dim,
            ),
        )

        self.view_projector = nn.Sequential(
            nn.Linear(
                embedding_dim,
                embedding_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                embedding_dim,
                projection_dim,
            ),
        )

    def forward(
        self,
        anchor: torch.Tensor,
        fusion: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        anchor_projection = F.normalize(
            self.anchor_projector(anchor),
            dim=-1,
        )

        view_projection = F.normalize(
            self.view_projector(fusion),
            dim=-1,
        )

        return (
            anchor_projection,
            view_projection,
        )