from typing import Dict

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from data.deposit_label_loader import LabelSplit
from losses.adapter import label_orthogonality_loss
from typing import List
class DepositClassificationTrainer:
    """Full-batch trainer for deposit classification adapter."""

    def __init__(
        self,
        adapter: nn.Module,
        deposit_embedding: torch.Tensor,
        deposit_anchor: torch.Tensor,
        deposit_final_embs: torch.Tensor,
        train_data: LabelSplit,
        test_data: LabelSplit,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ):
        self.adapter = adapter
        self.optimizer = optimizer
        self.device = device

        # for pretrain finetine and Scratch
        """ self.model = model
        self.deposit_metapaths = deposit_metapaths
        self.mineral_metapaths = mineral_metapaths
        self. graph = graph """

        # Pretrained tensors are fixed during downstream adaptation.
        self.deposit_embedding = deposit_embedding.detach().to(device)
        self.deposit_anchor = deposit_anchor.detach().to(device)
        self.deposit_final_embs = deposit_final_embs.detach().to(device)
        self.train_ids = torch.as_tensor(
            train_data.deposit_ids,
            dtype=torch.long,
            device=device,
        )
        self.train_labels = torch.as_tensor(
            train_data.labels,
            dtype=torch.long,
            device=device,
        )

        self.test_ids = torch.as_tensor(
            test_data.deposit_ids,
            dtype=torch.long,
            device=device,
        )
        self.test_labels = torch.as_tensor(
            test_data.labels,
            dtype=torch.long,
            device=device,
        )

        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self) -> float:
        self.adapter.train()

        # One full-graph forward per epoch.
        output = self.adapter(
            deposit_embedding=self.deposit_embedding,
            anchor=self.deposit_anchor,
            final_embs=self.deposit_final_embs,
        )

        logits = output["logits"][self.train_ids]

        loss_orthogonal = label_orthogonality_loss(self.adapter.class_moe.label_embeddings)
        loss = self.criterion(
            logits,
            self.train_labels,
        ) + 0.0001 * loss_orthogonal

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #print(f"loss_orthogonal:{loss_orthogonal.item()}")
        return loss.item()

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        self.adapter.eval()


        output = self.adapter(
                    deposit_embedding=self.deposit_embedding,
                    anchor=self.deposit_anchor,
                    final_embs=self.deposit_final_embs,
                )
        logits = output["logits"][self.test_ids]
        scores = output["attention"][self.test_ids]
        loss = self.criterion(
            logits,
            self.test_labels,
        )

        predictions = logits.argmax(dim=1)
        probabilities = torch.softmax(logits, dim=1)

        return {
            "loss": loss.item(),
            "accuracy": self._accuracy(
                predictions,
                self.test_labels,
            ),
            "macro_f1": self._macro_f1(
                predictions,
                self.test_labels,
            ),
            "auc": self._auc(
                probabilities,
                self.test_labels,
            ),
        }

    def fit(
        self,
        epochs: int,
        log_interval: int = 10,
    ) -> Dict[str, float]:
        for epoch in range(1, epochs + 1):
            loss = self.train_epoch()

            """ for path_id, edge_mask in enumerate(
                self.adapter.edge_masks
            ):

                print(
                    path_id,
                    edge_mask.logits.std(),
                ) """
            if epoch % log_interval == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:04d} | "
                    f"train_loss={loss:.4f}"
                )

        metrics = self.evaluate()

        print(
            f"Test | "
            f"loss={metrics['loss']:.4f} | "
            f"accuracy={metrics['accuracy']:.4f} | "
            f"macro_f1={metrics['macro_f1']:.4f} | "
            f"auc={metrics['auc']:.4f}"
        )

        return metrics

    @staticmethod
    def _accuracy(
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        return (
            predictions == targets
        ).float().mean().item()

    @staticmethod
    def _auc(
        probabilities: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        y_true = targets.cpu().numpy()
        y_prob = probabilities.cpu().numpy()

        if y_prob.shape[1] == 2:
            return roc_auc_score(
                y_true,
                y_prob[:, 1],
            )

        return roc_auc_score(
            y_true,
            y_prob,
            multi_class="ovr",
            average="macro",
        )

    @staticmethod
    def _macro_f1(
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        f1_scores = []

        for label in torch.unique(targets):
            true_positive = (
                (predictions == label)
                & (targets == label)
            ).sum()

            false_positive = (
                (predictions == label)
                & (targets != label)
            ).sum()

            false_negative = (
                (predictions != label)
                & (targets == label)
            ).sum()

            precision = true_positive / (
                true_positive + false_positive
            ).clamp(min=1)

            recall = true_positive / (
                true_positive + false_negative
            ).clamp(min=1)

            f1 = (
                2 * precision * recall
                / (precision + recall).clamp(min=1e-12)
            )

            f1_scores.append(f1)

        return torch.stack(f1_scores).mean().item()
