import torch
import torch.nn as nn

from losses.bpr import bpr_loss
from losses.contrastive import contrastive_loss
from losses.information_bottleneck import (
    information_bottleneck_loss,
)
from losses.reconstruction import reconstruction_loss,reconstruction_weight
from models.pretrain_model import PretrainOutput
from losses.scatter import scatter_loss,scatter_Isotropy
from typing import Tuple,Dict,List
class PretrainObjective(nn.Module):
    """Joint single-stage pre-training objective."""

    def __init__(
        self,
        rec_weight: float = 0.1,
        contrastive_weight: float = 0.001,
        ib_weight: float = 1e-3,
        temperature: float = 0.2,
        negative_ratio: float = 1.0,
        scatter_weight:float = 1e-6,
    ):
        super().__init__()

        self.rec_weight = rec_weight
        self.contrastive_weight = contrastive_weight
        self.ib_weight = ib_weight
        self.scatter_weight = scatter_weight
        self.temperature = temperature
        self.negative_ratio = negative_ratio

        self.rec_gamma = 2
        self.rec_row_batch_size = 128

    def forward(
        self,
        output: PretrainOutput,
        deposit_ids: torch.Tensor,
        positive_ids: torch.Tensor,
        negative_ids: torch.Tensor,
        deposit_metapaths: List[torch.Tensor],
        mineral_metapaths: List[torch.Tensor],
        edge_decoder: torch.nn.Module,
    ) -> Tuple[
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        deposit_emb = output.deposit.final[
            deposit_ids
        ]

        positive_emb = output.mineral.final[
            positive_ids
        ]

        negative_emb = output.mineral.final[
            negative_ids
        ]

        positive_scores = (
            deposit_emb * positive_emb
        ).sum(dim=-1)

        negative_scores = (
            deposit_emb * negative_emb
        ).sum(dim=-1)

        loss_bpr = bpr_loss(
            positive_scores,
            negative_scores,
        )

        deposit_nodes = torch.unique(
            deposit_ids
        )

        mineral_nodes = torch.unique(
            torch.cat(
                [
                    positive_ids,
                    negative_ids,
                ]
            )
        )

       

        deposit_embeddings = (
            output.deposit.final[
                deposit_nodes
            ]
        )

        loss_scatter_deposit = scatter_loss(
            deposit_embeddings
        )
        loss_scatter_mineral = scatter_loss(
            output.mineral.final[
                mineral_nodes
            ]
        )
        loss_scatter = 0.5 * (loss_scatter_deposit + loss_scatter_mineral)

        deposit_rec = reconstruction_loss(
                    views=output.deposit.views,
                    masked_graphs=output.deposit.masked_graphs,
                    masked_edges=output.deposit.masked_edges,
                    fusion_weights=output.deposit.fusion_weights,
                    metapaths=deposit_metapaths,
                    batch_nodes=deposit_nodes,
                    decoder=edge_decoder,
                    gamma=self.rec_gamma,
                    row_batch_size=self.rec_row_batch_size,
                )
        
        mineral_rec = reconstruction_loss(
            views=output.mineral.views,
            masked_graphs=output.mineral.masked_graphs,
            masked_edges=output.mineral.masked_edges,
            fusion_weights=output.mineral.fusion_weights,
            metapaths=mineral_metapaths,
            batch_nodes=mineral_nodes,
            decoder=edge_decoder,
            gamma=self.rec_gamma,
            row_batch_size=self.rec_row_batch_size,
        )


        loss_rec = 0.5 * (
            deposit_rec + mineral_rec
        )
        # --------------------------------------------------
        # Mini-batch information bottleneck
        # --------------------------------------------------
        deposit_ib = information_bottleneck_loss(
            mu=output.deposit.mu[
                deposit_nodes,
                :,
            ],
            logvar=output.deposit.logvar[
                deposit_nodes,
                :,
            ],
        )

        mineral_ib = information_bottleneck_loss(
            mu=output.mineral.mu[
                mineral_nodes,
                :,
            ],
            logvar=output.mineral.logvar[
                mineral_nodes,
                :,
            ],
        )

        loss_ib = 0.5 * (
            deposit_ib + mineral_ib
        )

        

        total_loss = (
            loss_bpr
            + self.rec_weight * loss_rec
            + self.scatter_weight * loss_scatter
            + self.ib_weight * loss_ib
        )

        losses = {
            "total": total_loss,
            "bpr": loss_bpr,
            "reconstruction": loss_rec,
            "scatter": loss_scatter,
            "information_bottleneck": loss_ib,
        }

        return total_loss, losses