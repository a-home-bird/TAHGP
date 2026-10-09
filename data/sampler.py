import numpy as np

from data.dataset import DepositMineralDataset


class BPRSampler:
    """Sample deposit-positive-negative triples for BPR training."""

    def __init__(
        self,
        dataset: DepositMineralDataset,
    ):
        self.dataset = dataset
        self.active_deposits = np.asarray(
            [
                deposit_id
                for deposit_id, minerals in enumerate(
                    dataset.train_positive
                )
                if len(minerals) > 0
            ],
            dtype=np.int64,
        )

    def sample(self) -> np.ndarray:
        """
        Generate one epoch of BPR triples.

        Returns
        -------
        np.ndarray
            Shape [num_train_interactions, 3]:
            deposit, positive_mineral, negative_mineral.
        """
        num_samples = len(self.dataset)
        deposits = np.random.choice(
            self.active_deposits,
            size=num_samples,
        )

        triples = np.empty(
            (num_samples, 3),
            dtype=np.int64,
        )

        for i, deposit_id in enumerate(deposits):
            positives = self.dataset.train_positive[deposit_id]

            positive = np.random.choice(positives)
            negative = self._sample_negative(positives)

            triples[i] = (
                deposit_id,
                positive,
                negative,
            )

        return triples

    def _sample_negative(
        self,
        positives: np.ndarray,
    ) -> int:
        """Sample a mineral not observed in the training set."""
        while True:
            mineral_id = np.random.randint(
                self.dataset.num_minerals
            )

            if mineral_id not in positives:
                return mineral_id