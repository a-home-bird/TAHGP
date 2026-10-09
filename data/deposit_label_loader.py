from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Dict, List,Tuple

@dataclass
class LabelSplit:
    deposit_ids: np.ndarray
    labels: np.ndarray


class DepositLabelLoader:
    """Load downstream labels for deposit nodes."""

    def __init__(
        self,
        label_path: str ,
        id_map_path: str,
        source_id_col: str = "mindat_id",
        deposit_id_col: str = "deposit_model_id",
    ):
        labels = pd.read_csv(label_path)
        id_map = pd.read_csv(id_map_path)

        # Keep only deposits that exist in the graph ID mapping.
        self.data = labels.merge(
            id_map[[source_id_col, deposit_id_col]],
            on=source_id_col,
            how="inner",
        )

        self.deposit_id_col = deposit_id_col

    def classification_split(
        self,
        label_col: str,
        train_ratio: float = 0.8,
        seed: int = 2023,
        ignore_label = -1,
    ) -> Tuple[LabelSplit, LabelSplit]:
        """Stratified train-test split for classification."""
        data = self.data

        if ignore_label is not None:
            data = data[data[label_col] != ignore_label]


        train_indices = []
        test_indices = []

        for label in sorted(data[label_col].unique()):
            indices = data.index[
                data[label_col] == label
            ].to_numpy()

            np.random.shuffle(indices)

            num_train = max(
                1,
                int(round(len(indices) * train_ratio)),
            )

            train_indices.extend(indices[:num_train])
            test_indices.extend(indices[num_train:])

        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)

        train = data.loc[train_indices]
        test = data.loc[test_indices]

        return (
            self._to_split(train, label_col),
            self._to_split(test, label_col),
        )

    def regression_split(
        self,
        label_col: str,
        train_ratio: float = 0.8,
        seed: int = 2023,
    ) -> Tuple[LabelSplit, LabelSplit]:
        """Random train-test split for regression."""
        data = self.data.dropna(subset=[label_col]).copy()
        data = data[data[label_col] != -1]

        indices = data.index.to_numpy()

        np.random.shuffle(indices)

        num_train = int(round(
            len(indices) * train_ratio
        ))

        train = data.loc[indices[:num_train]]
        test = data.loc[indices[num_train:]]

        return (
            self._to_split(train, label_col),
            self._to_split(test, label_col),
        )

    def _to_split(
        self,
        data: pd.DataFrame,
        label_col: str,
    ) -> LabelSplit:
        return LabelSplit(
            deposit_ids=data[
                self.deposit_id_col
            ].to_numpy(dtype=np.int64),
            labels=data[label_col].to_numpy(),
        )