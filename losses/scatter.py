import torch
import torch.nn.functional as F



def scatter_Isotropy(
    embeddings: torch.Tensor,
    covariance_weight: float = 1.0,
) -> torch.Tensor:
    """Encourage uniform Deposit embeddings on the unit sphere."""
    if embeddings.size(0) < 2:
        return embeddings.sum() * 0.0

    embeddings = F.normalize(
        embeddings,
        p=2,
        dim=-1,
    )

    num_nodes, embedding_dim = embeddings.shape

    # First moment.
    center = embeddings.mean(dim=0)

    mean_loss = center.pow(2).sum()

    # Second moment.
    covariance = (
        embeddings.T @ embeddings
    ) / num_nodes

    target = torch.eye(
        embedding_dim,
        dtype=embeddings.dtype,
        device=embeddings.device,
    ) / embedding_dim

    covariance_loss = (
        covariance - target
    ).pow(2).sum()

    return (
        mean_loss
        + covariance_weight * covariance_loss
    )

def scatter_loss(
    embeddings: torch.Tensor,
) -> torch.Tensor:
    """Center-away representation scattering loss.

    Parameters
    ----------
    embeddings
        Deposit embeddings with shape [B, D].

    Returns
    -------
    torch.Tensor
        Scalar scattering loss.
    """
    if embeddings.size(0) < 2:
        return embeddings.sum() * 0.0

    embeddings = F.normalize(
        embeddings,
        p=2,
        dim=-1,
    )

    center = embeddings.mean(
        dim=0,
    )

    return center.pow(2).sum()