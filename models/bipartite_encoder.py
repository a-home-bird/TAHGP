import torch
import torch.nn as nn
from typing import Tuple, Dict
import torch.nn.functional as F
class BipartiteEncoder(nn.Module):
    #LightGCN encoder for the deposit-mineral graph.

    def __init__(self, num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers

    def forward(
        self,
        deposit_embeddings: torch.Tensor,
        mineral_embeddings: torch.Tensor,
        graph: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_deposits = deposit_embeddings.size(0)

        x = torch.cat(
            [deposit_embeddings, mineral_embeddings],
            dim=0,
        )

        layers = [x]

        for _ in range(self.num_layers):
            x = torch.sparse.mm(graph, x)
            layers.append(x)

        x = torch.stack(layers).mean(dim=0)

        return (
            x[:num_deposits],
            x[num_deposits:],
        )

""" class BipartiteEncoder(nn.Module):
    #LightGCN-style Deposit-Mineral encoder with XSimGCL noise.

    def __init__(
        self,
        num_layers: int,
        eps: float = 0.1,
        contrastive_layer: int = 1,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.eps = eps
        self.contrastive_layer = contrastive_layer

    def _add_noise(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        noise = torch.rand_like(
            embeddings
        )

        noise = F.normalize(
            noise,
            p=2,
            dim=-1,
        )

        perturbation = (
            torch.sign(embeddings)
            * noise
            * self.eps
        )

        return embeddings + perturbation

    def forward(
        self,
        deposit_embedding: torch.Tensor,
        mineral_embedding: torch.Tensor,
        graph: torch.Tensor,
        perturbed: bool = False,
    ):
        num_deposits = deposit_embedding.size(0)
        num_minerals = mineral_embedding.size(0)

        embeddings = torch.cat(
            [
                deposit_embedding,
                mineral_embedding,
            ],
            dim=0,
        )

        layer_embeddings = []
        contrastive_embedding = None

        for layer in range(self.num_layers):
            embeddings = torch.sparse.mm(
                graph,
                embeddings,
            )

            if perturbed:
                embeddings = self._add_noise(
                    embeddings
                )

            layer_embeddings.append(
                embeddings
            )

            if (
                layer
                == self.contrastive_layer
            ):
                contrastive_embedding = (
                    embeddings
                )

        final_embedding = torch.stack(
            layer_embeddings,
            dim=0,
        ).mean(dim=0)

        if contrastive_embedding is None:
            contrastive_embedding = (
                layer_embeddings[-1]
            )

        deposit_final, mineral_final = torch.split(
            final_embedding,
            [
                num_deposits,
                num_minerals,
            ],
            dim=0,
        )

        deposit_cl, mineral_cl = torch.split(
            contrastive_embedding,
            [
                num_deposits,
                num_minerals,
            ],
            dim=0,
        )

        return (
            deposit_final,
            mineral_final,
            deposit_cl,
            mineral_cl,
        ) """