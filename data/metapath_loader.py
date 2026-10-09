from collections import OrderedDict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from typing import Dict, List,Tuple
DEPOSIT_METAPATHS = OrderedDict({
    "alteration": "deposit_alteration_deposit_processed.npz",
    "mineral_pattern": "deposit_mineral_pattern_deposit_processed.npz",
    "rockunit": "deposit_rockunit_deposit_processed.npz",
    "structure": "deposit_structure_deposit_processed.npz",
})

MINERAL_METAPATHS = OrderedDict({
    "electronegativity":
        "mineral_element_electronegativity_element_mineral_processed.npz",
    "family":"mineral_element_family_element_mineral_processed.npz",
    "ionization":"mineral_element_ionization_element_mineral_processed.npz",
    "element":"mineral_element_mineral_processed.npz",
    "radius":"mineral_element_radius_element_mineral_processed.npz",
})


def normalize_adj(adj: sp.spmatrix) -> sp.coo_matrix:
    adj = adj.tocsr().astype(np.float32)
    degree = np.asarray(adj.sum(axis=1)).ravel()

    d_inv_sqrt = np.zeros_like(degree)
    mask = degree > 0
    d_inv_sqrt[mask] = degree[mask] ** -0.5

    d_mat = sp.diags(d_inv_sqrt)
    return (d_mat @ adj @ d_mat).tocoo()


def to_torch_sparse(matrix: sp.spmatrix) -> torch.Tensor:
    matrix = matrix.tocoo().astype(np.float32)

    indices = torch.from_numpy(
        np.vstack((matrix.row, matrix.col)).astype(np.int64)
    )
    values = torch.from_numpy(matrix.data)

    return torch.sparse_coo_tensor(
        indices,
        values,
        matrix.shape,
    ).coalesce()


def normalize_sparse(
    adj: torch.Tensor,
) -> torch.Tensor:
    """Symmetric normalization: D^(-1/2) A D^(-1/2)."""
    adj = adj.coalesce()

    indices = adj.indices()
    values = adj.values()

    row = indices[0]
    col = indices[1]

    degree = torch.zeros(
        adj.size(0),
        dtype=values.dtype,
        device=values.device,
    )

    degree.scatter_add_(
        0,
        row,
        values,
    )

    inv_sqrt_degree = torch.zeros_like(
        degree
    )

    valid = degree > 0

    inv_sqrt_degree[valid] = (
        degree[valid].pow(-0.5)
    )

    normalized_values = (
        values
        * inv_sqrt_degree[row]
        * inv_sqrt_degree[col]
    )

    return torch.sparse_coo_tensor(
        indices,
        normalized_values,
        adj.shape,
        dtype=values.dtype,
        device=adj.device,
    ).coalesce()

def load_metapaths(
    data_dir: str,
    file_map: Dict[str, str],
) -> Tuple[List[str], List[torch.Tensor]]:
    data_dir = Path(data_dir)

    names = []
    matrices = []

    for name, filename in file_map.items():
        matrix = sp.load_npz(data_dir / filename)
        #matrix = normalize_adj(matrix)

        names.append(name)
        matrices.append(to_torch_sparse(matrix))

    return names, matrices


def load_all_metapaths(data_dir: str) -> dict:
    deposit_names, deposit_matrices = load_metapaths(
        data_dir,
        DEPOSIT_METAPATHS,
    )

    mineral_names, mineral_matrices = load_metapaths(
        data_dir,
        MINERAL_METAPATHS,
    )

    return {
        "deposit": {
            "names": deposit_names,
            "matrices": deposit_matrices,
        },
        "mineral": {
            "names": mineral_names,
            "matrices": mineral_matrices,
        },
    }

""" def select_metapaths(
    file_map: Dict[str, str],
    selected_names: List[str],
) -> OrderedDict:


    if selected_names is None:

        return OrderedDict(
            file_map
        )

    # --------------------------------------------------------
    # Check duplicate names
    # --------------------------------------------------------

    if len(selected_names) != len(
        set(selected_names)
    ):

        raise ValueError(
            f"Duplicated meta-path names: "
            f"{selected_names}"
        )

    # --------------------------------------------------------
    # Check validity
    # --------------------------------------------------------

    invalid_names = [
        name
        for name in selected_names
        if name not in file_map
    ]

    if invalid_names:

        raise ValueError(
            f"Unknown meta-path names: "
            f"{invalid_names}. "
            f"Available names: "
            f"{list(file_map.keys())}"
        )

    selected_map = OrderedDict()

    for name in selected_names:

        selected_map[name] = (
            file_map[name]
        )

    return selected_map

def load_metapaths(
    data_dir: str,
    file_map: Dict[str, str],
    selected_names: List[str] = None,
) -> Tuple[List[str], List[torch.Tensor]]:

    data_dir = Path(
        data_dir
    )

    # --------------------------------------------------------
    # Select requested meta-paths
    # --------------------------------------------------------

    selected_file_map = select_metapaths(
        file_map=file_map,
        selected_names=selected_names,
    )

    names = []
    matrices = []

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    for name, filename in selected_file_map.items():

        file_path = (
            data_dir
            / filename
        )

        if not file_path.exists():

            raise FileNotFoundError(
                f"Meta-path file not found: "
                f"{file_path}"
            )

        matrix = sp.load_npz(
            file_path
        )

        # 如果当前实验原本没有 normalization，
        # 这里仍然不要打开，保持与主实验完全一致。
        #
        # matrix = normalize_adj(matrix)

        names.append(
            name
        )

        matrices.append(
            to_torch_sparse(
                matrix
            )
        )

    return names, matrices


# ============================================================
# Load deposit + mineral meta-paths
# ============================================================

def load_all_metapaths(
    data_dir: str,
    deposit_names: List[str] = None,
    mineral_names: List[str] = None,
) -> dict:

    # --------------------------------------------------------
    # Deposit
    # --------------------------------------------------------

    deposit_names_loaded, deposit_matrices = (
        load_metapaths(
            data_dir=data_dir,
            file_map=DEPOSIT_METAPATHS,
            selected_names=deposit_names,
        )
    )

    # --------------------------------------------------------
    # Mineral
    # --------------------------------------------------------

    mineral_names_loaded, mineral_matrices = (
        load_metapaths(
            data_dir=data_dir,
            file_map=MINERAL_METAPATHS,
            selected_names=mineral_names,
        )
    )

    return {
        "deposit": {
            "names":
                deposit_names_loaded,

            "matrices":
                deposit_matrices,
        },

        "mineral": {
            "names":
                mineral_names_loaded,

            "matrices":
                mineral_matrices,
        },
    } """