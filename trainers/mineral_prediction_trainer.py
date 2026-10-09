from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from data.dataset import DepositMineralDataset
from data.sampler import BPRSampler
from losses.bpr import bpr_loss
from utils.metrics import evaluate_topk
from typing import Dict,Tuple,List

class MineralPredictionTrainer:
    """Trainer for potential mineral prediction."""

    def __init__(
        self,
        model: nn.Module,
        head: nn.Module,
        dataset: DepositMineralDataset,
        sampler: BPRSampler,
        deposit_metapaths: Sequence[torch.Tensor],
        mineral_metapaths: Sequence[torch.Tensor],
        graph: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        batch_size: int = 2048,
        reg_weight: float = 1e-4,
        top_k: int = 20,
    ):
        self.model = model
        self.head = head
        self.dataset = dataset
        self.sampler = sampler
        self.optimizer = optimizer
        self.device = device

        self.batch_size = batch_size
        self.reg_weight = reg_weight
        self.top_k = top_k

        self.deposit_metapaths = [
            matrix.to(device)
            for matrix in deposit_metapaths
        ]
        self.mineral_metapaths = [
            matrix.to(device)
            for matrix in mineral_metapaths
        ]
        self.graph = graph.to(device)

    def train_epoch(self) -> float:
        if not isinstance(self.model, dict):
            self.model.train()
        self.head.train()

        triples = self.sampler.sample()
        np.random.shuffle(triples)

        total_loss = 0.0
        num_batches = 0

        for start in range(0, len(triples), self.batch_size):
            batch = torch.as_tensor(
                triples[start:start + self.batch_size],
                dtype=torch.long,
                device=self.device,
            )

            deposit_ids = batch[:, 0]
            positive_ids = batch[:, 1]
            negative_ids = batch[:, 2]

            if not  isinstance(self.model, dict):
                deposit_emb, mineral_emb = self.model.encode(
                    self.deposit_metapaths,
                    self.mineral_metapaths,
                    self.graph,
                )
            else:
                deposit_emb, mineral_emb = self.model["deposit_embedding"], self.model["mineral_embedding"]

            deposit_batch = deposit_emb[deposit_ids]
            positive_batch = mineral_emb[positive_ids]
            negative_batch = mineral_emb[negative_ids]

            positive_scores = self.head(
                deposit_batch,
                positive_batch,
            )
            negative_scores = self.head(
                deposit_batch,
                negative_batch,
            )

            loss = bpr_loss(
                positive_scores,
                negative_scores,
            )

            loss += self.reg_weight * self._regularization(
                deposit_ids,
                positive_ids,
                negative_ids,
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def _regularization(
        self,
        deposit_ids: torch.Tensor,
        positive_ids: torch.Tensor,
        negative_ids: torch.Tensor,
    ) -> torch.Tensor:
        deposit_emb = self.model.deposit_embedding(
            deposit_ids
        )
        positive_emb = self.model.mineral_embedding(
            positive_ids
        )
        negative_emb = self.model.mineral_embedding(
            negative_ids
        )

        return (
            deposit_emb.pow(2).sum()
            + positive_emb.pow(2).sum()
            + negative_emb.pow(2).sum()
        ) / (2 * len(deposit_ids))

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        if not isinstance(self.model, dict):
            self.model.eval()
        self.head.eval()

        if not  isinstance(self.model, dict):
            deposit_emb, mineral_emb = self.model.encode(
                self.deposit_metapaths,
                self.mineral_metapaths,
                self.graph,
            )
        else:
            deposit_emb, mineral_emb = self.model["deposit_embedding"], self.model["mineral_embedding"]
        

        test_deposits = list(self.dataset.test_dict)
        predictions = []
        ground_truth = []

        for start in range(
            0,
            len(test_deposits),
            self.batch_size,
        ):
            batch_ids = test_deposits[
                start:start + self.batch_size
            ]

            batch = torch.as_tensor(
                batch_ids,
                dtype=torch.long,
                device=self.device,
            )

            scores = self.head.score_all(
                deposit_emb[batch],
                mineral_emb,
            )

            for row, deposit_id in enumerate(batch_ids):
                train_minerals = self.dataset.train_positive[
                    deposit_id
                ]
                scores[row, train_minerals] = -torch.inf

            top_items = torch.topk(
                scores,
                k=max(self.top_k),
                dim=1,
            ).indices

            predictions.append(
                top_items.cpu().numpy()
            )

            ground_truth.extend(
                self.dataset.test_dict[deposit_id]
                for deposit_id in batch_ids
            )

        predictions = np.concatenate(
            predictions,
            axis=0,
        )

        return evaluate_topk(
            predictions,
            ground_truth,
            self.top_k,
        )

    def fit(
        self,
        epochs: int,
        eval_interval: int = 5,
    ) -> Dict[str, float]:
        metrics = {}

        for epoch in range(1, epochs + 1):
            loss = self.train_epoch()

            if epoch % eval_interval != 0:
                print(
                    f"Epoch {epoch:04d} | "
                    f"loss={loss:.4f}"
                )
                continue

            metrics = self.evaluate()

            recall = metrics[
                f"recall@{self.top_k}"
            ]
            ndcg = metrics[
                f"ndcg@{self.top_k}"
            ]

            print(
                f"Epoch {epoch:04d} | "
                f"loss={loss:.4f} | "
                f"Recall@{self.top_k}={recall:.4f} | "
                f"NDCG@{self.top_k}={ndcg:.4f}"
            )

        return metrics