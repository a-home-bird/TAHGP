import torch
import torch.nn.functional as F


def label_orthogonality_loss(
    label_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Encourage different classes to use distinct label queries."""
    num_classes = label_embeddings.size(0)

    if num_classes < 2:
        return label_embeddings.sum() * 0.0

    normalized = F.normalize(
        label_embeddings,
        p=2,
        dim=-1,
    )

    similarity = (
        normalized @ normalized.T
    )

    identity = torch.eye(
        num_classes,
        dtype=similarity.dtype,
        device=similarity.device,
    )

    off_diagonal = (
        similarity - identity
    )

    return (
        off_diagonal.pow(2).sum()
        / (
            num_classes
            * (num_classes - 1)
        )
    )