import torch
import torch.nn as nn

from data.metapath_loader import normalize_sparse
from typing import Tuple,List

def compute_metapath_richness(
    metapaths: List[torch.Tensor],
) -> torch.Tensor:
    """Compute relative richness of meta-path graphs."""
    scores = []

    for adj in metapaths:
        adj = adj.coalesce()

        indices = adj.indices()
        num_nodes = adj.size(0)

        covered_nodes = torch.unique(
            torch.cat(
                [indices[0], indices[1]]
            )
        ).numel()

        coverage = (
            covered_nodes / num_nodes
        )

        density = (
            adj._nnz()
            / (num_nodes * num_nodes)
        )

        score = (
            torch.log1p(
                torch.tensor(
                    float(adj._nnz()),
                    device=adj.device,
                )
            )
            * coverage
            * (1.0 + density)
        )

        scores.append(score)

    scores = torch.stack(scores)

    if len(scores) == 1:
        return torch.ones_like(
            scores
        )

    return (
        scores - scores.min()
    ) / (
        scores.max()
        - scores.min()
        + 1e-12
    )


class DynamicEdgeMask(nn.Module):
    """Dynamic richness-aware meta-path masking."""

    def __init__(
        self,
        min_ratio: float = 0.05,
        max_ratio: float = 0.30,
    ):
        super().__init__()

        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def forward(
        self,
        adj: torch.Tensor,
        progress: float,
        richness: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        adj = adj.coalesce()

        ratio = (
            self.min_ratio
            + progress
            * (
                self.max_ratio
                - self.min_ratio
            )
            * richness
        )

        indices = adj.indices()
        values = adj.values()

        row = indices[0]
        col = indices[1]

        num_nodes = adj.size(0)

        # Treat (i, j) and (j, i) as one edge pair.
        left = torch.minimum(row, col)
        right = torch.maximum(row, col)

        pair_ids = (
            left * num_nodes + right
        )

        unique_pairs, inverse = torch.unique(
            pair_ids,
            return_inverse=True,
        )

        pair_row = (
            unique_pairs // num_nodes
        )
        pair_col = (
            unique_pairs % num_nodes
        )

        keep_pair = (
            torch.rand(
                unique_pairs.size(0),
                device=adj.device,
            )
            >= ratio
        )

        keep_pair = self._restore_isolated_nodes(
            keep_pair,
            pair_row,
            pair_col,
            num_nodes,
        )

        keep_edge = keep_pair[inverse]

        masked_adj = torch.sparse_coo_tensor(
            indices[:, keep_edge],
            values[keep_edge],
            adj.shape,
            device=adj.device,
        ).coalesce()

        # Reconstruction targets:
        # one canonical edge per undirected pair.
        removed_pair = ~keep_pair

        masked_edges = torch.stack(
            [
                pair_row[removed_pair],
                pair_col[removed_pair],
            ],
            dim=0,
        )

        # Recompute normalization AFTER masking.
        masked_adj = normalize_sparse(
            masked_adj
        )

        return (
            masked_adj,
            masked_edges,
        )

    @staticmethod
    def _restore_isolated_nodes(
        keep_pair: torch.Tensor,
        pair_row: torch.Tensor,
        pair_col: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """Restore one edge if masking isolates a covered node."""
        original_degree = torch.zeros(
            num_nodes,
            dtype=torch.long,
            device=keep_pair.device,
        )

        kept_degree = torch.zeros_like(
            original_degree
        )

        ones = torch.ones_like(
            pair_row,
            dtype=torch.long,
        )

        original_degree.scatter_add_(
            0,
            pair_row,
            ones,
        )
        original_degree.scatter_add_(
            0,
            pair_col,
            ones,
        )

        kept_ids = torch.nonzero(
            keep_pair,
            as_tuple=False,
        ).squeeze(-1)

        if kept_ids.numel() > 0:
            kept_degree.scatter_add_(
                0,
                pair_row[kept_ids],
                torch.ones_like(
                    kept_ids
                ),
            )

            kept_degree.scatter_add_(
                0,
                pair_col[kept_ids],
                torch.ones_like(
                    kept_ids
                ),
            )

        isolated = torch.nonzero(
            (original_degree > 0)
            & (kept_degree == 0),
            as_tuple=False,
        ).squeeze(-1)

        for node in isolated.tolist():
            incident = torch.nonzero(
                (
                    (pair_row == node)
                    | (pair_col == node)
                )
                & (~keep_pair),
                as_tuple=False,
            ).squeeze(-1)

            if incident.numel() > 0:
                keep_pair[
                    incident[0]
                ] = True

        return keep_pair