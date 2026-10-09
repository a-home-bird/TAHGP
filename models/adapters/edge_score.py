import torch
import torch.nn as nn

class BilinearEdgeScorer(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        scorer_dim: int = 16,
    ):
        super().__init__()

        self.projection = nn.Linear(
            embedding_dim,
            scorer_dim,
            bias=False,
        )

        self.bias = nn.Parameter(
            torch.zeros(1)
        )

    def forward(
        self,
        src_embedding,
        dst_embedding,
    ):
        src = self.projection(
            src_embedding
        )

        dst = self.projection(
            dst_embedding
        )

        return (
            src * dst
        ).sum(dim=-1) + self.bias
    

class EdgeScorer(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 32,
    ):
        super().__init__()

        self.scorer = nn.Sequential(
            nn.Linear(
                embedding_dim * 3,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                1,
            ),
        )

    def forward(
        self,
        src_embedding: torch.Tensor,
        dst_embedding: torch.Tensor,
    ) -> torch.Tensor:
        pair_features = torch.cat(
            [
                0.5 * (
                    src_embedding
                    + dst_embedding
                ),
                torch.abs(
                    src_embedding
                    - dst_embedding
                ),
                src_embedding
                * dst_embedding,
            ],
            dim=-1,
        )

        return (
            self.scorer(
                pair_features
            )
            .squeeze(-1)
        )