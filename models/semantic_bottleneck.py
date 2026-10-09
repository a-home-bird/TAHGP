import torch
import torch.nn as nn
from typing import Tuple
import torch.nn.functional as F
class SemanticBottleneck(nn.Module):
    """Variational bottleneck for meta-path semantic offsets."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        hidden_dim = hidden_dim or embedding_dim

        input_dim = embedding_dim * 3

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            #nn.ReLU(),
        )

        self.ib_size = hidden_dim // 2

        """ self.mu = nn.Linear(
            hidden_dim,
            embedding_dim,
        )
        self.logvar = nn.Linear(
            hidden_dim,
            embedding_dim,
        ) """
        self.decode = nn.Linear(self.ib_size,embedding_dim)
    def forward(
        self,
        views: torch.Tensor,
        anchor: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Parameters
        ----------
        views
            [P, N, D]
        anchor
            [N, D]
        """
        num_views = views.size(0)
        if len(views.shape) == 3:
            anchor = anchor.unsqueeze(0).expand(
                num_views,
                -1,
                -1,
            )
            print("ib shape error")
            exit(-1)

        features = torch.cat(
            [
                views,
                anchor,
                torch.abs(views - anchor),
            ],
            dim=-1,
        )

        hidden = self.encoder(features)

        """ mu = self.mu(hidden)
        logvar = self.logvar(hidden) """

        mu = hidden[:,:self.ib_size]
        logvar = F.softplus(hidden[:,self.ib_size:])

        if self.training:
            std = torch.exp(0.5 * logvar)
            noise = torch.randn_like(std)

            offsets = mu + std * noise
        else:
            offsets = mu

        offsets_d = self.decode(offsets)
        return offsets_d, mu, logvar