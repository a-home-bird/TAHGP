import csv
import json
from pathlib import Path


RESULT_FIELDS = [
    "stage",
    "task",
    "label",
    "train_ratio",
    "train_size",
    "test_size",
    "seed",
    "best_epoch",
    "loss",
    "recall",
    "ndcg",
    "accuracy",
    "macro_f1",
    "auc",
]


def save_json(
    path: str,
    data: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


def append_result(
    path: str,
    result: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    exists = path.exists()

    row = {
        field: result.get(field, "")
        for field in RESULT_FIELDS
    }

    with path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=RESULT_FIELDS,
        )

        if not exists:
            writer.writeheader()

        writer.writerow(row)