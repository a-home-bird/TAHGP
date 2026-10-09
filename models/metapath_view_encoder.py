from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaPathViewEncoder(nn.Module):
    """Encode each meta-path independently."""

    def __init__(
        self,
        num_layers: int = 2,
    ):
        super().__init__()
        self.num_layers = num_layers

    def forward(
        self,
        x: torch.Tensor,
        metapaths: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """
        Returns
        -------
        Tensor
            Shape [num_metapaths, num_nodes, embedding_dim].
        """
        views = []

        for adj in metapaths:
            h = x
            layers = []

            for _ in range(self.num_layers):
                h = torch.sparse.mm(adj, h)
                h = F.relu(h)
                layers.append(h)

            views.append(
                torch.stack(layers).mean(dim=0)
            )

        return torch.stack(views, dim=0)