import torch
import torch.nn.functional as F

def contrastive_loss(
    anchor: torch.Tensor,
    fusion: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """ mini-batch InfoNCE."""

    anchor = torch.nn.functional.normalize(anchor)
    fusion = torch.nn.functional.normalize(fusion)

    batch_size = anchor.size(0)

    if batch_size < 2:
        return anchor.sum() * 0.0

    labels = torch.arange(
        batch_size,
        device=anchor.device,
    )

    logits = anchor @ fusion.T / temperature

    loss = F.cross_entropy(logits, labels)
    return loss

def contrastive_loss_multi(
    anchor: torch.Tensor,
    views: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """
    Mini-batch cross-view InfoNCE.

    Parameters
    ----------
    anchor
        [B, D]
    views
        [P, B, D]
    """
    batch_size = anchor.size(0)

    if batch_size < 2:
        return anchor.sum() * 0.0

    labels = torch.arange(
        batch_size,
        device=anchor.device,
    )

    losses = []

    for view in views:
        logits = (
            anchor @ view.T
        ) / temperature

        anchor_to_view = F.cross_entropy(
            logits,
            labels,
        )

        view_to_anchor = F.cross_entropy(
            logits.T,
            labels,
        )

        losses.append(
            0.5
            * (
                anchor_to_view
                + view_to_anchor
            )
        )

    return torch.stack(losses).mean()