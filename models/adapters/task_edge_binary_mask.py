import torch
import torch.nn as nn

from typing import Tuple
import math

class TaskEdgeBinaryMask(nn.Module):
    """Task-specific binary mask over existing meta-path edges."""

    def __init__(
        self,
        adjacency: torch.Tensor,
        temperature: float = 1.0,
        init_logit: float = 2.0,
    ):
        super().__init__()

        adjacency = adjacency.coalesce()
        self.keep_ratio = 0.7
        self.num_nodes = adjacency.size(0)
        self.temperature = temperature

        indices = adjacency.indices()
        values = adjacency.values()

        row = indices[0]
        col = indices[1]

        # --------------------------------------------------
        # Fixed self-loops
        # --------------------------------------------------
        self_loop = row == col

        self.register_buffer(
            "self_indices",
            indices[:, self_loop],
        )
        self.register_buffer(
            "self_values",
            values[self_loop],
        )

        # --------------------------------------------------
        # Learnable non-self edges
        # --------------------------------------------------
        non_self = ~self_loop

        row = row[non_self]
        col = col[non_self]
        values = values[non_self]

        # Canonical undirected pair: (min(i, j), max(i, j))
        src = torch.minimum(row, col)
        dst = torch.maximum(row, col)

        pair_ids = (
            src * self.num_nodes
            + dst
        )

        unique_pair_ids, inverse = torch.unique(
            pair_ids,
            sorted=True,
            return_inverse=True,
        )

        pair_src = (
            unique_pair_ids
            // self.num_nodes
        )
        pair_dst = (
            unique_pair_ids
            % self.num_nodes
        )

        self.register_buffer(
            "edge_indices",
            torch.stack(
                [row, col],
                dim=0,
            ),
        )

        self.register_buffer(
            "edge_values",
            values,
        )

        self.register_buffer(
            "edge_to_pair",
            inverse,
        )

        # Useful for later interpretation/export.
        self.register_buffer(
            "pair_indices",
            torch.stack(
                [pair_src, pair_dst],
                dim=0,
            ),
        )

        self.num_edge_pairs = (
            unique_pair_ids.numel()
        )

        self.logits = nn.Parameter(
            torch.full(
                (self.num_edge_pairs,),
                init_logit,
            )
        )

        """ self.edge_scorer = EdgeScorer(
            embedding_dim=64,
            hidden_dim=32,
        ) """
        

    @property
    def probabilities(
        self,
    ) -> torch.Tensor:
        return torch.sigmoid(
            self.logits
            / self.temperature
        )


    def _hard_topk_mask(
        self,
    ) -> torch.Tensor:
        """Return hard Top-K edge-pair mask."""
        if self.keep_ratio >= 1.0:
            return torch.ones_like(
                self.logits
            )

        num_keep = max(
            1,
            math.ceil(
                self.keep_ratio
                * self.num_edge_pairs
            ),
        )

        _, topk_indices = torch.topk(
            self.logits,
            k=num_keep,
            dim=0,
        )

        hard_mask = torch.zeros_like(
            self.logits
        )

        hard_mask = hard_mask.scatter(
            dim=0,
            index=topk_indices,
            value=1.0,
        )

        return hard_mask
    
    def forward(
        self,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Returns
        -------
        masked_adj
            Raw sparse adjacency after task-specific masking.

        pair_mask
            STE binary mask for undirected edge pairs [E_pair].

        probabilities
            Soft keep probabilities [E_pair].
        """
        probabilities = self.probabilities

        """ hard_mask = (
            probabilities >= 0.5
        ).to(probabilities.dtype) """

        hard_mask = (
            self._hard_topk_mask()
        )


        if self.training:
            # Hard forward, soft backward.
            pair_mask = (
                hard_mask.detach()
                - probabilities.detach()
                + probabilities
            )
        else:
            pair_mask = hard_mask

        edge_mask = pair_mask[
            self.edge_to_pair
        ].to(self.edge_values.dtype)

        masked_values = (
            self.edge_values
            * edge_mask
        )

        # Reinsert fixed self-loops.
        if self.self_values.numel() > 0:
            output_indices = torch.cat(
                [
                    self.edge_indices,
                    self.self_indices,
                ],
                dim=1,
            )

            output_values = torch.cat(
                [
                    masked_values,
                    self.self_values,
                ],
                dim=0,
            )
        else:
            output_indices = (
                self.edge_indices
            )
            output_values = (
                masked_values
            )

        masked_adj = torch.sparse_coo_tensor(
            output_indices,
            output_values,
            (
                self.num_nodes,
                self.num_nodes,
            ),
            dtype=output_values.dtype,
            device=output_values.device,
        ).coalesce()

        #print(self.logits.grad)

        
        return (
            masked_adj,
            pair_mask,
            probabilities,
        )