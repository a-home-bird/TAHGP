import torch
import torch.nn as nn
from typing import Tuple

class ReliabilityFusion(nn.Module):
    """Global and node-level meta-path semantic fusion."""

    def __init__(
        self,
        num_metapaths: int,
        embedding_dim: int,
        hidden_dim: int = None,
    ):
        super().__init__()

        hidden_dim = hidden_dim or embedding_dim

        self.global_logits = nn.Parameter(
            torch.zeros(num_metapaths)
        )

        self.gate = nn.Sequential(
            nn.Linear(
                embedding_dim * 3,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    @property
    def global_weights(self) -> torch.Tensor:
        return torch.sigmoid(
            self.global_logits
        )

    def forward(
        self,
        anchor: torch.Tensor,
        offsets: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Parameters
        ----------
        anchor
            [N, D]
        offsets
            [P, N, D]
        """
        num_views = offsets.size(0)

        anchor_view = anchor.unsqueeze(0).expand(
            num_views,
            -1,
            -1,
        )

        gate_input = torch.cat(
            [
                anchor_view,
                offsets,
                torch.abs(
                    anchor_view - offsets
                ),
            ],
            dim=-1,
        )

        node_weights = torch.sigmoid(
            self.gate(gate_input)
        )

        global_weights = self.global_weights.view(
            -1,
            1,
            1,
        )

        weights = (
            global_weights
            * node_weights
        )

        residual = (
            weights * offsets
        ).sum(dim=0) / (
            weights.sum(dim=0) + 1e-12
        )

        """ final_embedding = (
            anchor + residual
        ) """

        return residual, weights