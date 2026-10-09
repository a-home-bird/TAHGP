from pathlib import Path

import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Tuple

class DepositMineralDataset:
    """Deposit-mineral interaction dataset."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

        self.num_deposits = self._count_ids("deposit_id_map.csv")
        self.num_minerals = self._count_ids("mineral_id_map.csv")

        self.train_dict = self._read_interactions("train.txt")
        self.test_dict = self._read_interactions("test.txt")
        self.dev_dict = self._read_interactions("dev.txt")
        
        self.train_deposits, self.train_minerals = self._flatten(
            self.train_dict
        )

        self.interaction_matrix = self._build_interaction_matrix()
        self.graph = self._build_bipartite_graph()

        self.train_positive = [
            self.interaction_matrix[i].indices
            for i in range(self.num_deposits)
        ]

    def _count_ids(self, filename: str) -> int:
        """Count nodes from an ID mapping CSV."""
        with open(
            self.data_dir / filename,
            "r",
            encoding="utf-8",
        ) as file:
            return sum(1 for _ in file) - 1

    def _read_interactions(
        self,
        filename: str,
    ):
        """
        Read interaction file.

        Format:
            deposit_id mineral_id mineral_id ...
        """
        interactions = {}

        with open(
            self.data_dir / filename,
            "r",
            encoding="utf-8",
        ) as file:
            for line in file:
                values = list(map(int, line.split()))
                if not values:
                    continue

                deposit_id = values[0]
                interactions[deposit_id] = values[1:]

        return interactions

    def set_train_interactions(
        self,
        interactions: Dict[int, List[int]],
    ) -> None:
        """Replace training interactions and rebuild the graph."""
        self.train_dict = interactions

        self.train_deposits, self.train_minerals = self._flatten(
            self.train_dict
        )

        self.interaction_matrix = self._build_interaction_matrix()
        self.graph = self._build_bipartite_graph()

        self.train_positive = [
            self.interaction_matrix[i].indices
            for i in range(self.num_deposits)
        ]
    @staticmethod
    def _flatten(
        interactions,
    ) :
        """Convert interaction dictionary to edge arrays."""
        deposits = []
        minerals = []

        for deposit_id, mineral_ids in interactions.items():
            deposits.extend([deposit_id] * len(mineral_ids))
            minerals.extend(mineral_ids)

        return (
            np.asarray(deposits, dtype=np.int64),
            np.asarray(minerals, dtype=np.int64),
        )

    def _build_interaction_matrix(self) -> sp.csr_matrix:
        """
        Construct deposit-mineral matrix R.

        Shape:
            [num_deposits, num_minerals]
        """
        values = np.ones(
            len(self.train_deposits),
            dtype=np.float32,
        )

        return sp.csr_matrix(
            (
                values,
                (self.train_deposits, self.train_minerals),
            ),
            shape=(
                self.num_deposits,
                self.num_minerals,
            ),
        )

    def _build_bipartite_graph(self) -> sp.csr_matrix:
        """
        Construct normalized LightGCN adjacency matrix:

            [ 0   R  ]
            [ R.T 0  ]
        """
        zero_d = sp.csr_matrix(
            (self.num_deposits, self.num_deposits),
            dtype=np.float32,
        )
        zero_m = sp.csr_matrix(
            (self.num_minerals, self.num_minerals),
            dtype=np.float32,
        )

        graph = sp.bmat(
            [
                [zero_d, self.interaction_matrix],
                [self.interaction_matrix.T, zero_m],
            ],
            format="csr",
        )

        degree = np.asarray(graph.sum(axis=1)).ravel()

        d_inv_sqrt = np.zeros_like(
            degree,
            dtype=np.float32,
        )
        mask = degree > 0
        d_inv_sqrt[mask] = degree[mask] ** -0.5

        d_mat = sp.diags(d_inv_sqrt)

        return (d_mat @ graph @ d_mat).tocsr()

    def __len__(self) -> int:
        return len(self.train_deposits)