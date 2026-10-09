
import torch
import torch.nn as nn


class EdgeDecoder(nn.Module):
    """Lightweight graph decoder for adjacency reconstruction."""

    def __init__(
        self,
        embedding_dim: int,
        decoder_dim: int = 32,
    ):
        super().__init__()

        self.projection = nn.Linear(
            embedding_dim,
            decoder_dim,
            bias=False,
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        graph: torch.Tensor = None,
    ) -> torch.Tensor:
        """Decode node embeddings using masked graph structure."""
        """ propagated = torch.sparse.mm(
            graph,
            embeddings,
        )

        hidden = 0.5 * (
            embeddings + propagated
        )

        return self.projection(hidden) """

        if graph is not None:
            hidden = self.projection(embeddings)

            propagated = torch.sparse.mm(
                graph,
                hidden,
            )
            return propagated
        else:
            return self.projection(embeddings)