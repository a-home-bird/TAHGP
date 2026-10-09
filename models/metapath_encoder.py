from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, List,Tuple
class MetaPathEncoder(nn.Module):
    """Weighted multi-meta-path graph encoder."""

    def __init__(
        self,
        num_metapaths: int,
        num_layers: int = 2,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.weight_logits = nn.Parameter(
            torch.zeros(num_metapaths)
        )

    @property
    def metapath_weights(self) -> torch.Tensor:
        return torch.sigmoid(self.weight_logits)

    def propagate(
        self,
        x: torch.Tensor,
        metapaths: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        outputs = [
            weight * torch.sparse.mm(adj, x)
            for weight, adj in zip(
                self.metapath_weights,
                metapaths,
            )
        ]

        return torch.stack(outputs).sum(dim=0)

    def forward(
        self,
        x: torch.Tensor,
        metapaths: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        layer_embeddings = []

        for _ in range(self.num_layers):
            x = self.propagate(x, metapaths)
            x = F.relu(x)
            layer_embeddings.append(x)

        return torch.stack(layer_embeddings).mean(dim=0)