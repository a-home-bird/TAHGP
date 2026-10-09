from pathlib import Path

import torch


def read_checkpoint(
    path: str ,
    device: torch.device,
) -> dict:
    """Read checkpoint."""
    return torch.load(
        path,
        map_location=device,
        weights_only=False,
    )


def save_checkpoint(
    path: str ,
    model: torch.nn.Module,
    metadata: dict ,
) -> None:
    """Save model parameters and metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": metadata or {},
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    device: torch.device,
) -> dict:
    """Load model parameters and return metadata."""
    checkpoint = read_checkpoint(path, device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return checkpoint.get("metadata", {})