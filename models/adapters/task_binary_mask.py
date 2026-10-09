import torch
import torch.nn as nn

from typing import Tuple
class TaskBinaryMask(nn.Module):
    """Task-specific learnable binary meta-path mask."""

    def __init__(
        self,
        num_metapaths: int,
        temperature: float = 1.0,
        init_logit: float = 2.0,
    ):
        super().__init__()

        self.temperature = temperature

        self.logits = nn.Parameter(
            torch.full(
                (num_metapaths,),
                init_logit,
            )
        )

    @property
    def probabilities(self) -> torch.Tensor:
        return torch.sigmoid(
            self.logits / self.temperature
        )

    def forward(
        self,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Returns
        -------
        mask
            STE binary mask, shape [P].

        probabilities
            Soft selection probabilities, shape [P].
        """
        probabilities = self.probabilities

        hard_mask = (
            probabilities >= 0.5
        ).to(probabilities.dtype)

        if self.training:
            mask = (
                hard_mask.detach()
                - probabilities.detach()
                + probabilities
            )
        else:
            mask = hard_mask

        return mask, probabilities