import torch


def information_bottleneck_loss(
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    """KL divergence to standard Gaussian."""
    kl = -0.5 * (
        1
        + logvar
        - mu.pow(2)
        - logvar.exp()
    )

    return kl.mean()