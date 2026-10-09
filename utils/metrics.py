import numpy as np

from typing import Sequence
from typing import Dict, List,Tuple

def recall_at_k(
    predictions: np.ndarray,
    ground_truth: List[List[int]],
    k: int,
) -> float:
    """Mean Recall@K over all evaluated deposits."""
    recalls = []

    for pred, target in zip(predictions[:, :k], ground_truth):
        hits = len(set(pred) & set(target))
        recalls.append(hits / len(target))

    return float(np.mean(recalls))


def ndcg_at_k(
    predictions: np.ndarray,
    ground_truth: List[List[int]],
    k: int,
) -> float:
    """Mean NDCG@K over all evaluated deposits."""
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    ndcgs = []

    for pred, target in zip(predictions[:, :k], ground_truth):
        target = set(target)

        relevance = np.asarray(
            [item in target for item in pred],
            dtype=np.float32,
        )

        dcg = np.sum(relevance * discounts)
        ideal_len = min(len(target), k)
        idcg = np.sum(discounts[:ideal_len])

        ndcgs.append(dcg / idcg)

    return float(np.mean(ndcgs))


def evaluate_topk(
    predictions: np.ndarray,
    ground_truth: List[List[int]],
    k: list,
) -> Dict[str, float]:
    """Evaluate Top-K mineral prediction."""
    if isinstance(k,list):
        result = []
        for k_item in k:
            result.append(
                {
                f"recall@{k_item}": recall_at_k(
                    predictions,
                    ground_truth,
                    k_item,
                ),
                f"ndcg@{k_item}": ndcg_at_k(
                    predictions,
                    ground_truth,
                    k_item,
                ),
                }
            )
        return result
    else:
        return {
            f"recall@{k}": recall_at_k(
                predictions,
                ground_truth,
                k,
            ),
            f"ndcg@{k}": ndcg_at_k(
                predictions,
                ground_truth,
                k,
            ),
        }