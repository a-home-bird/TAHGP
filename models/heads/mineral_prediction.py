import torch
import torch.nn as nn


class MineralPredictionHead(nn.Module):
    """Score deposit-mineral pairs by embedding similarity."""

    def forward(
        self,
        deposit_embeddings: torch.Tensor,
        mineral_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score matched deposit-mineral pairs.

        Parameters
        ----------
        deposit_embeddings : Tensor
            Shape [batch_size, embedding_dim].
        mineral_embeddings : Tensor
            Shape [batch_size, embedding_dim].

        Returns
        -------
        Tensor
            Pair scores with shape [batch_size].
        """
        return (
            deposit_embeddings
            * mineral_embeddings
        ).sum(dim=-1)

    def score_all(
        self,
        deposit_embeddings: torch.Tensor,
        mineral_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score deposits against all candidate minerals.

        Returns
        -------
        Tensor
            Score matrix with shape
            [num_deposits, num_minerals].
        """
        return deposit_embeddings @ mineral_embeddings.T