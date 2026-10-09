
import math

import torch
import torch.nn as nn

from typing import Tuple
class ReliabilityFusion(nn.Module):
    """Lightweight anchor-conditioned meta-path MoE fusion."""

    def __init__(
        self,
        num_metapaths: int,
        embedding_dim: int,
        router_dim: int = 16,
        temperature: float = 1.0,
    ):
        super().__init__()

        self.temperature = temperature

        # Shared low-rank router.
        self.query = nn.Linear(
            embedding_dim,
            router_dim,
            bias=False,
        )

        self.key = nn.Linear(
            embedding_dim,
            router_dim,
            bias=False,
        )

        # Global meta-path importance.
        self.global_logits = nn.Parameter(
            torch.zeros(num_metapaths)
        )

        self.scale = math.sqrt(router_dim)

    @property
    def global_weights(self) -> torch.Tensor:
        """Global routing prior across meta-paths."""
        return torch.softmax(
            self.global_logits,
            dim=0,
        )

    def forward(
        self,
        anchor: torch.Tensor,
        offsets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        anchor
            [N, D]
        offsets
            [P, N, D]

        Returns
        -------
        residual
            [N, D]
        weights
            [P, N, 1]
        """
        # Query: [N, R]
        query = self.query(anchor)

        # Keys: [P, N, R]
        keys = self.key(offsets)

        # Node-specific expert routing scores: [P, N]
        scores = (
            query.unsqueeze(0) * keys
        ).sum(dim=-1) / self.scale

        # Add global meta-path importance.
        scores = scores + self.global_logits[:, None]

        # Normalize across meta-path experts.
        weights = torch.softmax(
            scores / self.temperature,
            dim=0,
        ).unsqueeze(-1)

        # Weighted expert fusion: [N, D]
        residual = (
            weights * offsets
        ).sum(dim=0)

        return residual, weights