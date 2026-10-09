
from dataclasses import dataclass

import torch
import torch.nn as nn

from models.bipartite_encoder import BipartiteEncoder
from models.contrastive_projector import ContrastiveProjector
from models.dynamic_mask import (
    DynamicEdgeMask,
    compute_metapath_richness,
)
from models.metapath_view_encoder import MetaPathViewEncoder
from models.reliability_fusion import ReliabilityFusion
from models.semantic_bottleneck import SemanticBottleneck

from typing import Tuple,Dict,List,Sequence
from data.metapath_loader import normalize_sparse
from models.edge_decoder import EdgeDecoder

@dataclass
class DomainOutput:
    anchor: torch.Tensor
    #contrastive: torch.Tensor

    views: torch.Tensor
    offsets: torch.Tensor

    mu: torch.Tensor
    logvar: torch.Tensor

    final: torch.Tensor

    contrast_anchor: torch.Tensor
    contrast_views: torch.Tensor

    fusion_weights: torch.Tensor

    masked_edges: List[torch.Tensor]
    masked_graphs: List[torch.Tensor]

    progress:float


@dataclass
class PretrainOutput:
    deposit: DomainOutput
    mineral: DomainOutput


class PretrainModel(nn.Module):
    """Joint heterogeneous graph pre-training model."""

    def __init__(
        self,
        num_deposits: int,
        num_minerals: int,
        embedding_dim: int,
        num_deposit_metapaths: int,
        num_mineral_metapaths: int,
        num_metapath_layers: int = 2,
        num_graph_layers: int = 2,
        bottleneck_hidden_dim: int = None,
        projection_dim: int = 64,
        min_mask_ratio: float = 0.05,
        max_mask_ratio: float = 0.30,
    ):
        super().__init__()

        self.deposit_embedding = nn.Embedding(
            num_deposits,
            embedding_dim,
        )
        self.mineral_embedding = nn.Embedding(
            num_minerals,
            embedding_dim,
        )

        self.deposit_richness = None
        self.mineral_richness = None

        self.deposit_clean_metapaths = None
        self.mineral_clean_metapaths = None

        nn.init.xavier_uniform_(
            self.deposit_embedding.weight
        )
        nn.init.xavier_uniform_(
            self.mineral_embedding.weight
        )

        self.anchor_encoder = BipartiteEncoder(
            num_layers=num_graph_layers
        )

        self.deposit_view_encoder = MetaPathViewEncoder(
            num_layers=num_metapath_layers
        )
        self.mineral_view_encoder = MetaPathViewEncoder(
            num_layers=num_metapath_layers
        )

        self.deposit_bottleneck = SemanticBottleneck(
            embedding_dim,
            bottleneck_hidden_dim,
        )
        self.mineral_bottleneck = SemanticBottleneck(
            embedding_dim,
            bottleneck_hidden_dim,
        )

        self.deposit_fusion = ReliabilityFusion(
            num_deposit_metapaths,
            embedding_dim,
        )
        self.mineral_fusion = ReliabilityFusion(
            num_mineral_metapaths,
            embedding_dim,
        )

        self.deposit_projector = ContrastiveProjector(
            embedding_dim,
            projection_dim,
        )
        self.mineral_projector = ContrastiveProjector(
            embedding_dim,
            projection_dim,
        )

        self.edge_mask = DynamicEdgeMask(
            min_ratio=min_mask_ratio,
            max_ratio=max_mask_ratio,
        )

        self.edge_decoder = EdgeDecoder(
                    embedding_dim=embedding_dim,
                    decoder_dim=32,
                )
        
    def prepare_metapaths(
        self,
        deposit_metapaths: Sequence[torch.Tensor],
        mineral_metapaths: Sequence[torch.Tensor],
    ) -> None:
        """Cache clean normalized graphs and richness."""
        self.deposit_clean_metapaths = [
            normalize_sparse(adj)
            for adj in deposit_metapaths
        ]

        self.mineral_clean_metapaths = [
            normalize_sparse(adj)
            for adj in mineral_metapaths
        ]

        self.deposit_richness = (
            compute_metapath_richness(
                list(deposit_metapaths)
            )
        )

        self.mineral_richness = (
            compute_metapath_richness(
                list(mineral_metapaths)
            )
        )
    def _mask_metapaths(
        self,
        metapaths: Sequence[torch.Tensor],
        richness: torch.Tensor,
        progress: float,
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
    ]:
        masked_graphs = []
        masked_edges = []

        for path_id, adj in enumerate(
            metapaths
        ):
            masked_adj, edge_index = (
                self.edge_mask(
                    adj,
                    progress,
                    richness[path_id],
                )
            )

            masked_graphs.append(
                masked_adj
            )

            masked_edges.append(
                edge_index
            )

        return (
            masked_graphs,
            masked_edges,
        )

    def _encode_domain(
        self,
        anchor: torch.Tensor,
        base_embedding: torch.Tensor,
        metapaths: Sequence[torch.Tensor],
        clean_metapaths: Sequence[torch.Tensor],
        richness: torch.Tensor,
        view_encoder: MetaPathViewEncoder,
        bottleneck: SemanticBottleneck,
        fusion: ReliabilityFusion,
        projector: ContrastiveProjector,
        progress: float,
        apply_mask: bool,
    ) -> DomainOutput:
        if apply_mask:
            graphs, masked_edges = (
                self._mask_metapaths(
                    metapaths,
                    richness,
                    progress,
                )
            )

        else:
            graphs = clean_metapaths
            masked_edges = []

        views = view_encoder(
            base_embedding,
            graphs,
        )

        residual, fusion_weights = fusion(
                anchor,
                views,
            )
        offsets, mu, logvar = bottleneck(
            residual,
            anchor,
        )

        final = anchor + 0.1 *  residual
        contrast_anchor, contrast_views = anchor, anchor + residual
        return DomainOutput(
            anchor=anchor,
            views=views,
            offsets=offsets,
            mu=mu,
            logvar=logvar,
            final=final,
            contrast_anchor=contrast_anchor,
            contrast_views=contrast_views,
            fusion_weights=fusion_weights,
            masked_edges=masked_edges,
            masked_graphs=graphs,
            progress=progress
        )

    def forward(
        self,
        deposit_metapaths: Sequence[torch.Tensor],
        mineral_metapaths: Sequence[torch.Tensor],
        graph: torch.Tensor,
        progress: float = 1.0,
    ) -> PretrainOutput:

        deposit_anchor, mineral_anchor = (
                    self.anchor_encoder(
                        self.deposit_embedding.weight,
                        self.mineral_embedding.weight,
                        graph,
                    )
                )

        deposit = self._encode_domain(
            anchor=deposit_anchor,
            base_embedding=self.deposit_embedding.weight,
            metapaths=deposit_metapaths,
            clean_metapaths=self.deposit_clean_metapaths,
            richness=self.deposit_richness,
            view_encoder=self.deposit_view_encoder,
            bottleneck=self.deposit_bottleneck,
            fusion=self.deposit_fusion,
            projector=self.deposit_projector,
            progress=progress,
            apply_mask=self.training,
        )

        mineral = self._encode_domain(
            anchor=mineral_anchor,
            base_embedding=self.mineral_embedding.weight,
            metapaths=mineral_metapaths,
            clean_metapaths=self.mineral_clean_metapaths,
            richness=self.mineral_richness,
            view_encoder=self.mineral_view_encoder,
            bottleneck=self.mineral_bottleneck,
            fusion=self.mineral_fusion,
            projector=self.mineral_projector,
            progress=progress,
            apply_mask=self.training,
        )

        return PretrainOutput(
            deposit=deposit,
            mineral=mineral,
        )

    def encode(
        self,
        deposit_metapaths: Sequence[torch.Tensor],
        mineral_metapaths: Sequence[torch.Tensor],
        graph: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate clean embeddings for downstream tasks."""
        was_training = self.training

        try:
            if not was_training:
                self.eval()

            output = self.forward(
                deposit_metapaths=deposit_metapaths,
                mineral_metapaths=mineral_metapaths,
                graph=graph,
                progress=1.0,
            )
        finally:
            self.train(was_training)

        return (
            output.deposit.final,
            output.mineral.final,
        )

        


    def encode_deposit_components(
        self,
        graph: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return clean Deposit anchor and meta-path views."""

        anchor_deposit, _ = self.bipartite_encoder(
            graph,
            self.deposit_embedding.weight,
            self.mineral_embedding.weight,
        )

        views = self.deposit_view_encoder(
            self.deposit_clean_metapaths,
            self.deposit_embedding.weight,
        )

        return anchor_deposit, views
    @property
    def deposit_metapath_weights(self) -> torch.Tensor:
        return self.deposit_fusion.global_weights

    @property
    def mineral_metapath_weights(self) -> torch.Tensor:
        return self.mineral_fusion.global_weights