from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from data.dataset import DepositMineralDataset
from data.metapath_loader import to_torch_sparse
from data.sampler import BPRSampler
from utils.metrics import evaluate_topk
from typing import Tuple,Dict,Sequence,List
from pathlib import Path
import os
class PretrainTrainer:
    """Single-stage heterogeneous graph pre-training."""

    def __init__(
        self,
        model: nn.Module,
        objective: nn.Module,
        dataset: DepositMineralDataset,
        deposit_metapaths: Sequence[torch.Tensor],
        mineral_metapaths: Sequence[torch.Tensor],
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        batch_size: int = 2048,
        top_k: int = 20,
        validation_ratio: float = 0.1,
        reg_weight: float = 1e-4,
    ):
        self.model = model
        self.objective = objective
        self.dataset = dataset
        self.optimizer = optimizer
        self.device = device

        self.batch_size = batch_size
        self.top_k = top_k
        self.reg_weight = reg_weight

        self.deposit_metapaths = [
            matrix.to(device)
            for matrix in deposit_metapaths
        ]
        self.mineral_metapaths = [
            matrix.to(device)
            for matrix in mineral_metapaths
        ]



        self.validation_dict = dataset.dev_dict

        # Validation edges must not participate in training.

        self.graph = to_torch_sparse(
            self.dataset.graph
        ).to(device)

        self.sampler = BPRSampler(
            self.dataset
        )

        self.model.prepare_metapaths(
            self.deposit_metapaths,
            self.mineral_metapaths,
        )
        
    def _tempol_save(self, data,path):
        Path(path).parent.mkdir(
            parents=True,
            exist_ok=True
        )
        with open(path,'w') as f:
            for key, value in data.items():
                f.write(str(key))
                for mineral in value:
                    f.write(" "+str(mineral))
                f.write("\n")
        

    @staticmethod
    def _split_validation(
        interactions: Dict[int, List[int]],
        ratio: float,
    ) -> Tuple[
        Dict[int, List[int]],
        Dict[int, List[int]],
    ]:
        """
        Hold out a proportion of training edges for validation.

        Each deposit keeps at least one mineral in the training set.
        """


        total_edges = sum(
            len(minerals)
            for minerals in interactions.values()
        )

        max_validation = sum(
            max(len(minerals) - 1, 0)
            for minerals in interactions.values()
        )

        target = min(
            int(round(total_edges * ratio)),
            max_validation,
        )

        candidates = [
            (deposit_id, mineral_id)
            for deposit_id, minerals in interactions.items()
            if len(minerals) > 1
            for mineral_id in minerals
        ]

        np.random.shuffle(candidates)

        remaining = {
            deposit_id: len(minerals)
            for deposit_id, minerals in interactions.items()
        }

        validation = defaultdict(set)
        validation_count = 0
        for deposit_id, mineral_id in candidates:
            if validation_count >= target:
                break

            if remaining[deposit_id] <= 1:
                continue

            validation[deposit_id].add(mineral_id)
            remaining[deposit_id] -= 1
            validation_count += 1
            
        train_dict = {
            deposit_id: [
                mineral_id
                for mineral_id in minerals
                if mineral_id not in validation.get(deposit_id, set())
            ]
            for deposit_id, minerals in interactions.items()
        }

        validation_dict = {
            deposit_id: list(minerals)
            for deposit_id, minerals in validation.items()
        }

        
        return train_dict, validation_dict

    def train_epoch(
        self,
        progress: float,
    ) -> Dict[str, float]:
        self.model.train()

        triples = self.sampler.sample()
        np.random.shuffle(triples)

        total_losses = defaultdict(float)
        num_batches = 0

        for start in range(
            0,
            len(triples),
            self.batch_size,
        ):
            batch = torch.as_tensor(
                triples[
                    start:start + self.batch_size
                ],
                dtype=torch.long,
                device=self.device,
            )

            deposit_ids = batch[:, 0]
            positive_ids = batch[:, 1]
            negative_ids = batch[:, 2]

            output = self.model(
                deposit_metapaths=self.deposit_metapaths,
                mineral_metapaths=self.mineral_metapaths,
                graph=self.graph,
                progress=progress,
            )

            loss, loss_dict = self.objective(
                output=output,
                deposit_ids=deposit_ids,
                positive_ids=positive_ids,
                negative_ids=negative_ids,
                deposit_metapaths=self.deposit_metapaths,
                mineral_metapaths=self.mineral_metapaths,
                edge_decoder=self.model.edge_decoder
            )

            reg_loss = self._regularization(
                deposit_ids,
                positive_ids,
                negative_ids,
            )

            loss = (
                loss
                + self.reg_weight * reg_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_losses["total"] += loss.item()
            total_losses["regularization"] += (
                reg_loss.item()
            )

            for name, value in loss_dict.items():
                if name == "total":
                    continue

                total_losses[name] += (
                    value.item()
                )

            num_batches += 1

        return {
            name: value / num_batches
            for name, value
            in total_losses.items()
        }

    def _regularization(
        self,
        deposit_ids: torch.Tensor,
        positive_ids: torch.Tensor,
        negative_ids: torch.Tensor,
    ) -> torch.Tensor:
        """L2 regularization of shared ID embeddings."""
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
    def evaluate_validation(
        self,
    ) -> Dict[str, float]:
        self.model.eval()

        deposit_emb, mineral_emb = (
            self.model.encode(
                self.deposit_metapaths,
                self.mineral_metapaths,
                self.graph,
            )
        )

        deposits = list(
            self.validation_dict
        )

        predictions = []
        ground_truth = []

        for start in range(
            0,
            len(deposits),
            self.batch_size,
        ):
            batch_ids = deposits[
                start:start + self.batch_size
            ]

            batch = torch.as_tensor(
                batch_ids,
                dtype=torch.long,
                device=self.device,
            )

            scores = (
                deposit_emb[batch]
                @ mineral_emb.T
            )

            for row, deposit_id in enumerate(
                batch_ids
            ):
                train_minerals = (
                    self.dataset.train_positive[
                        deposit_id
                    ]
                )

                scores[
                    row,
                    train_minerals,
                ] = -torch.inf

            top_items = torch.topk(
                scores,
                k=self.top_k,
                dim=1,
            ).indices

            predictions.append(
                top_items.cpu().numpy()
            )

            ground_truth.extend(
                self.validation_dict[
                    deposit_id
                ]
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
        patience: int = 20,
    ) -> Dict:
        best_recall = -1.0
        best_ndcg = 0.0
        best_epoch = 0
        best_state = None

        early_stop = 0
        best_losses = {}
        for epoch in range(1, epochs + 1):
            progress = (
                1.0
                if epochs == 1
                else (epoch - 1) / (epochs - 1)
            )

            losses = self.train_epoch(
                progress
            )

            self._print_train(
                epoch,
                progress,
                losses,
            )

            if epoch % eval_interval != 0:
                continue

            metrics = (
                self.evaluate_validation()
            )

            recall = metrics[
                f"recall@{self.top_k}"
            ]
            ndcg = metrics[
                f"ndcg@{self.top_k}"
            ]

            print(
                f"Validation | "
                f"Recall@{self.top_k}="
                f"{recall:.4f} | "
                f"NDCG@{self.top_k}="
                f"{ndcg:.4f}"
            )

            if recall > best_recall:
                best_recall = recall
                best_ndcg = ndcg
                best_epoch = epoch

                best_state = deepcopy(
                    self.model.state_dict()
                )
                best_losses = losses.copy()
                early_stop = 0
            else:
                early_stop += 1

            if early_stop >= patience:
                print(
                    f"Early stopping | "
                    f"best_epoch={best_epoch}"
                )
                break

        if best_state is not None:
            self.model.load_state_dict(
                best_state
            )

        return {
            "best_epoch": best_epoch,
            f"val_recall@{self.top_k}":
                best_recall,
            f"val_ndcg@{self.top_k}":
                best_ndcg,
            "train_losses": best_losses,
        }

    @staticmethod
    def _print_train(
        epoch: int,
        progress: float,
        losses: Dict[str, float],
    ) -> None:
        print(
            f"Epoch {epoch:04d} | "
            f"progress={progress:.3f} | "
            f"total={losses['total']:.4f} | "
            f"bpr={losses['bpr']:.4f} | "
            f"rec={losses['reconstruction']:.4f} | "
            f"scatter={losses['scatter']:.4f} | "
            #f"contrastive={losses['contrastive']:.4f} | "
            f"ib={losses['information_bottleneck']:.4f} | "
            f"reg={losses['regularization']:.4f}"
        )