
import torch

from typing import List,Dict
import math

def reconstruction_weight(
    base_weight: float,
    progress: float,
    active_ratio: float = 0.7,
) -> float:
    if progress >= active_ratio:
        return 0.0

    local_progress = (
        progress / active_ratio
    )

    return (
        base_weight
        * 0.5
        * (
            1.0
            + math.cos(
                math.pi
                * local_progress
            )
        )
    )

def sce_adjacency_loss(
    embeddings: torch.Tensor,
    target_adj: torch.Tensor,
    row_nodes: torch.Tensor,
    gamma: float = 2.0,
    row_batch_size: int = 128,
) -> torch.Tensor:
    """Scaled cosine error for selected adjacency rows."""
    target_adj = target_adj.coalesce()

    src, dst = target_adj.indices()
    valid_edges = src != dst

    src = src[valid_edges]
    dst = dst[valid_edges]

    num_nodes = embeddings.size(0)
    total_loss = embeddings.new_zeros(())
    total_rows = 0

    for rows in row_nodes.split(row_batch_size):
        num_rows = rows.numel()

        # Reconstruct selected rows against all nodes.
        scores = (
            embeddings[rows] @ embeddings.T
        )

        #predicted = torch.sigmoid(scores)
        predicted = scores
        # Exclude diagonal entries.
        predicted = predicted.scatter(
            1,
            rows.unsqueeze(1),
            0.0,
        )

        # Map global node IDs to local row indices.
        row_map = torch.full(
            (num_nodes,),
            -1,
            dtype=torch.long,
            device=rows.device,
        )
        row_map[rows] = torch.arange(
            num_rows,
            device=rows.device,
        )

        local_rows = row_map[src]
        selected = local_rows >= 0

        local_rows = local_rows[selected]
        local_cols = dst[selected]

        # Binary adjacency targets.
        # Compute cosine numerator without densifying A.
        numerator = torch.zeros(
            num_rows,
            device=embeddings.device,
            dtype=embeddings.dtype,
        )

        numerator.scatter_add_(
            0,
            local_rows,
            predicted[local_rows, local_cols],
        )

        # Binary target row L2 norms.
        degrees = torch.bincount(
            local_rows,
            minlength=num_rows,
        ).to(embeddings.dtype)

        target_norm = degrees.sqrt()
        pred_norm = predicted.norm(
            p=2,
            dim=1,
        )

        valid = degrees > 0

        cosine = numerator / (
            target_norm * pred_norm + 1e-12
        )

        loss = (
            1.0 - cosine[valid]
        ).clamp(min=0).pow(gamma)

        total_loss = total_loss + loss.sum()
        total_rows += int(valid.sum().item())

    if total_rows == 0:
        return embeddings.sum() * 0.0

    return total_loss / total_rows

import torch
import torch.nn.functional as F


def weighted_mse_adjacency_loss(
    embeddings: torch.Tensor,
    target_adj: torch.Tensor,
    row_nodes: torch.Tensor,
    row_batch_size: int = 128,
) -> torch.Tensor:
    """Balanced MSE for sparse adjacency reconstruction."""
    target_adj = target_adj.coalesce()

    indices = target_adj.indices()
    src = indices[0]
    dst = indices[1]

    num_nodes = embeddings.size(0)

    total_loss = embeddings.new_zeros(())
    total_rows = 0

    for rows in row_nodes.split(row_batch_size):
        num_rows = rows.numel()

        # [B, N]
        scores = (
            embeddings[rows]
            @ embeddings.T
        )

        # [B, N]
        #predicted = torch.sigmoid(scores)
        predicted = scores
        # Dense binary target for selected rows.
        target = torch.zeros_like(
            predicted
        )

        # Map global row node ID -> local row index.
        row_map = torch.full(
            (num_nodes,),
            -1,
            dtype=torch.long,
            device=rows.device,
        )

        row_map[rows] = torch.arange(
            num_rows,
            device=rows.device,
        )

        local_rows = row_map[src]

        valid_edges = (
            (local_rows >= 0)
            & (src != dst)
        )

        local_rows = local_rows[valid_edges]
        local_cols = dst[valid_edges]

        target[
            local_rows,
            local_cols,
        ] = 1.0

        # --------------------------------------------------
        # Ignore diagonal instead of modifying predicted.
        # --------------------------------------------------
        valid_mask = torch.ones_like(
            target,
            dtype=torch.bool,
        )

        valid_mask[
            torch.arange(
                num_rows,
                device=rows.device,
            ),
            rows,
        ] = False

        positive_mask = (
            (target > 0)
            & valid_mask
        )

        negative_mask = (
            (target == 0)
            & valid_mask
        )

        if positive_mask.any():
            positive_loss = F.mse_loss(
                predicted[positive_mask],
                target[positive_mask],
            )
        else:
            positive_loss = (
                embeddings.sum() * 0.0
            )

        if negative_mask.any():
            negative_loss = F.mse_loss(
                predicted[negative_mask],
                target[negative_mask],
            )
        else:
            negative_loss = (
                embeddings.sum() * 0.0
            )

        # Balance positive and negative reconstruction.
        batch_loss = 0.5 * (
            positive_loss
            + negative_loss
        )

        total_loss = (
            total_loss
            + batch_loss * num_rows
        )

        total_rows += num_rows

    if total_rows == 0:
        return embeddings.sum() * 0.0

    return total_loss / total_rows

def neighborhood_distribution_loss(
    embeddings: torch.Tensor,
    target_adj: torch.Tensor,
    row_nodes: torch.Tensor,
    temperature: float = 1.0,
    row_batch_size: int = 128,
) -> torch.Tensor:
    """Row-wise neighborhood distribution reconstruction."""
    target_adj = target_adj.coalesce()

    indices = target_adj.indices()
    values = target_adj.values()

    src = indices[0]
    dst = indices[1]

    num_nodes = embeddings.size(0)

    total_loss = embeddings.new_zeros(())
    total_rows = 0

    for rows in row_nodes.split(row_batch_size):
        scores = (
            embeddings[rows]
            @ embeddings.T
        ) / temperature

        # Prevent trivial self reconstruction.
        local_ids = torch.arange(
            rows.numel(),
            device=rows.device,
        )
        scores[local_ids, rows] = -torch.inf

        log_prob = torch.log_softmax(
            scores,
            dim=1,
        )

        row_map = torch.full(
            (num_nodes,),
            -1,
            dtype=torch.long,
            device=rows.device,
        )

        row_map[rows] = torch.arange(
            rows.numel(),
            device=rows.device,
        )

        local_rows = row_map[src]

        valid = (
            (local_rows >= 0)
            & (src != dst)
        )

        local_rows = local_rows[valid]
        local_cols = dst[valid]
        edge_values = values[valid]

        if local_rows.numel() == 0:
            continue

        row_sum = torch.zeros(
            rows.numel(),
            dtype=edge_values.dtype,
            device=edge_values.device,
        )

        row_sum.scatter_add_(
            0,
            local_rows,
            edge_values,
        )

        target_prob = (
            edge_values
            / row_sum[local_rows].clamp_min(1e-12)
        )

        edge_loss = (
            -target_prob
            * log_prob[
                local_rows,
                local_cols,
            ]
        )

        row_loss = torch.zeros(
            rows.numel(),
            dtype=edge_loss.dtype,
            device=edge_loss.device,
        )

        row_loss.scatter_add_(
            0,
            local_rows,
            edge_loss,
        )

        valid_rows = row_sum > 0

        total_loss += (
            row_loss[valid_rows].sum()
        )

        total_rows += int(
            valid_rows.sum().item()
        )

    if total_rows == 0:
        return embeddings.sum() * 0.0

    return total_loss / total_rows

def reconstruction_loss(
    views: torch.Tensor,
    masked_graphs: List[torch.Tensor],
    masked_edges: List[torch.Tensor],
    metapaths: List[torch.Tensor],
    fusion_weights: torch.Tensor,
    batch_nodes: torch.Tensor,
    decoder: torch.nn.Module,
    gamma: float = 2.0,
    row_batch_size: int = 128,
) -> torch.Tensor:
    """Mini-batch masked meta-path adjacency reconstruction."""
    losses = []
    weights = []

    # Shared representation across meta-path views.
    decode_views = []
    for path_id, masked_edges_p in enumerate(masked_graphs):
        decode_views.append(decoder(
            embeddings=views[path_id],
            graph=masked_graphs[path_id],
        ))
    decode_views = torch.stack(decode_views,0)
    decoded = (
        fusion_weights.detach() * decode_views
    ).sum(dim=0)

    # Decode only once for all meta-paths.
    """ decoded = decoder(
        embeddings=shared_view,
    ) """

    for path_id, masked_edges_p in enumerate(masked_edges):
        if masked_edges_p.numel() == 0:
            continue

        # Nodes affected by the current edge mask.
        masked_nodes = torch.unique(
            masked_edges_p.reshape(-1)
        )

        # Reconstruct only current mini-batch nodes
        # that have at least one masked edge.
        row_nodes = batch_nodes[
            torch.isin(batch_nodes, masked_nodes)
        ]

        if row_nodes.numel() == 0:
            continue

        """ decoded = decoder(
            embeddings=views[path_id],
            graph=masked_graphs[path_id],
        ) """

        loss = sce_adjacency_loss(
            embeddings=decoded,
            target_adj=metapaths[path_id],
            row_nodes=row_nodes,
            #gamma=gamma,
            #row_batch_size=row_batch_size,
        )

        # Average MoE routing weight over reconstruction nodes.
        path_weight = fusion_weights[
            path_id,
            row_nodes,
            0,
        ].detach().mean()
        weights.append(path_weight)
        losses.append(loss)

    if not losses:
        return views.sum() * 0.0

    losses = torch.stack(losses)
    weights = torch.stack(weights)

    return (
        (weights * losses).sum()
        / weights.sum().clamp_min(1e-12)
    )
    #return torch.stack(losses).mean()