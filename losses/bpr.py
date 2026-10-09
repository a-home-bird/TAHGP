import torch
import torch.nn.functional as F


def bpr_loss(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
) -> torch.Tensor:
    """Bayesian Personalized Ranking loss."""
    return F.softplus(
        negative_scores - positive_scores
    ).mean()