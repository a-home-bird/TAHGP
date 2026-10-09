from typing import List, Tuple

import torch
import torch.nn as nn

from data.metapath_loader import normalize_sparse
from models.adapters.class_metapath_moe import ClassMetaPathMoE
from models.adapters.task_edge_binary_mask import TaskEdgeBinaryMask
from models.adapters.TaskEdgeScorerMask import TaskEdgeScorerMask
from models.metapath_view_encoder import MetaPathViewEncoder

from typing import Dict,Tuple,List
class DepositClassificationAdapter(nn.Module):
    """Task-edge adaptation + class-wise meta-path MoE."""

    def __init__(
        self,
        deposit_metapaths: List[torch.Tensor],
        embedding_dim: int,
        num_classes: int,
        num_metapath_layers: int,
        attention_dim: int = 32,
        mask_temperature: float = 1.0,
        mask_init_logit: float = 2.0,
    ):
        super().__init__()

        self.num_metapaths = len(deposit_metapaths)
        self.meta_path = deposit_metapaths
        self.edge_masks = nn.ModuleList(
            [
                TaskEdgeBinaryMask(
                    adjacency=adj,
                    temperature=mask_temperature,
                    init_logit=mask_init_logit,
                )
                for adj in deposit_metapaths
            ]
        )

        # Parameter-free encoder. Keep the same propagation depth as pretraining.
        self.view_encoder = MetaPathViewEncoder(num_metapath_layers)

        self.class_moe = ClassMetaPathMoE(
            embedding_dim=embedding_dim,
            num_metapaths=self.num_metapaths,
            num_classes=num_classes,
            attention_dim=attention_dim,
        )

    def build_task_graphs(
        self,
        pretrained_embedding: torch.Tensor,
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
    ]:
        graphs = []
        hard_masks = []
        probabilities = []

        for edge_mask in self.edge_masks:
            masked_adj, hard_mask, soft, probability = edge_mask(pretrained_embedding)
           
            normalized_adj = normalize_sparse(masked_adj)
            #normalized_adj = masked_adj
            graphs.append(normalized_adj)
            hard_masks.append(hard_mask)
            probabilities.append(probability)

        return graphs, hard_masks, probabilities

    def forward(
        self,
        deposit_embedding: torch.Tensor,
        anchor: torch.Tensor,
        final_embs:torch.Tensor,
    ) -> dict:

        task_graphs = self.meta_path
        #task_graphs = [normalize_sparse(i) for i in task_graphs]
        # Full-batch propagation over all deposits.
        # Do not use torch.no_grad() here: gradients must reach edge masks.
        views = self.view_encoder(
            final_embs,
            task_graphs,
        )
        # [P, N, D]

        logits, attention, expert_logits = self.class_moe(
            anchor=anchor,
            views=views,
            final_embs=final_embs,
        )

        return {
            "logits": logits,
            "views": views,
            "attention": attention,
            "expert_logits": expert_logits,
            #"hard_masks": hard_masks,
            #"mask_probabilities": mask_probabilities,
        }
