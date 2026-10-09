import math

import torch
import torch.nn as nn
from typing import Tuple
import torch.nn.functional as F
class ClassMetaPathMoE(nn.Module):
    """Label-aware class-wise meta-path expert fusion."""

    def __init__(
        self,
        embedding_dim: int,
        num_metapaths: int,
        num_classes: int,
        attention_dim: int = 32,
    ):
        super().__init__()

        self.num_metapaths = num_metapaths
        self.num_classes = num_classes
        #attention_dim = 16
        self.key_projection = nn.Linear(
            embedding_dim,
            attention_dim,
            bias=False,
        )

        # Label embeddings act as class-specific queries.
        self.label_embeddings = nn.Parameter(
            torch.empty(
                num_classes,
                #embedding_dim,
                attention_dim,
            )
        )

        # Expert 0 = DM anchor;
        # Expert 1...P = meta-path views.
        self.expert_embeddings = nn.Parameter(
            torch.empty(
                num_metapaths,
                attention_dim,
            )
        )

        # Shared classifier avoids path-specific classifier inflation.
        """ self.classifier = nn.Linear(
            embedding_dim,
            num_classes,
        ) """
        self.classifier = nn.Sequential(
                    nn.Linear(embedding_dim, embedding_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(embedding_dim, num_classes),
                )

        

        self.scale = math.sqrt(
            attention_dim
        )

        nn.init.xavier_uniform_(
            self.label_embeddings
        )

        nn.init.normal_(
            self.expert_embeddings,
            std=0.02,
        )
        self.adapter_scale = nn.Parameter(
        torch.tensor(0.0)
    )

    def straight_through_topk_softmax(
        self,
        scores: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """Hard Top-K in forward, dense softmax gradient in backward."""
        num_experts = scores.size(1)

        soft_attention = torch.softmax(
            scores,
            dim=1,
        )

        if k >= num_experts:
            return soft_attention

        _, topk_indices = torch.topk(
            scores,
            k=k,
            dim=1,
        )

        topk_mask = torch.zeros_like(
            scores,
            dtype=torch.bool,
        )

        topk_mask = topk_mask.scatter(
            dim=1,
            index=topk_indices,
            value=True,
        )

        masked_scores = scores.masked_fill(
            ~topk_mask,
            float("-inf"),
        )


        hard_attention = torch.softmax(
            masked_scores,
            dim=1,
        )

        attention = (
            hard_attention.detach()
            - soft_attention.detach()
            + soft_attention
        )

        return attention
    def forward(
        self,
        anchor: torch.Tensor,
        views: torch.Tensor,
        final_embs:torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Parameters
        ----------
        anchor
            [B, D]

        views
            [P, B, D]

        Returns
        -------
        logits
            [B, C]

        attention
            [B, P + 1, C]

        expert_logits
            [B, P + 1, C]
        """
        """ experts = torch.cat(
            [
                anchor.unsqueeze(0),
                views,
            ],
            dim=0,
        ) """

        experts = views
        # [P + 1, B, D]
        #keys = experts
        keys = self.key_projection(
            experts
        )
        # [P + 1, B, R]

        """ keys = (
            keys
            + self.expert_embeddings[
                :,
                None,
                :
            ]
        ) """
        """ keys = F.normalize(keys,p=2,dim=-1)
        label_embeddings = F.normalize(self.label_embeddings,p=2,dim=-1) """
        # [B, P + 1, C]
        scores = torch.einsum(
            "ebr,cr->bec",
            keys,
            self.label_embeddings,
        )/ self.scale
        #) / self.scale

        # Normalize across experts for each class.
        """ attention = torch.softmax(
            scores,
            dim=1,
        ) """

        attention = self.straight_through_topk_softmax(scores,2)

        #print(attention.detach().cpu().numpy())
        # [P + 1, B, C]
        expert_logits = self.classifier(
            experts
        )

        # [B, P + 1, C]
        expert_logits = (
            expert_logits.permute(
                1,
                0,
                2,
            )
        )

        # Prediction-level class-wise MoE.
        logits = (
            attention
            * expert_logits
        ).sum(dim=1)
    
        final_logits = self.classifier(
            final_embs
        )
        final_logits = final_logits + 0.3 * logits
        return (
            final_logits,
            scores.detach(),
            expert_logits,
        )