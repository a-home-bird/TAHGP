
import argparse
import pickle
from pathlib import Path

import torch

from data.dataset import DepositMineralDataset
from data.metapath_loader import load_all_metapaths, to_torch_sparse
from models.heads.deposit_classification import DepositClassificationHead
from models.pretrain_model import PretrainModel
from trainers.deposit_classification_trainer import (
    DepositClassificationTrainer,
)
from utils.checkpoint import read_checkpoint
from utils.result_logger import append_result, save_json
from utils.seed import set_seed
import os
from models.adapters.deposit_classification_adapter import (
    DepositClassificationAdapter,
)
import pickle
CLASSIFICATION_TASKS = [
    "deposit_type_label",
    "tonnage_class_label",
    "max_age_class_label",
]

TRAIN_RATIOS = [0.1, 0.2, 0.4, 0.6]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune deposit classification"
    )

    parser.add_argument("--data_dir", type=str, default="dataset/")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="",
    )

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)

    # Adapter hyperparameters. Loss regularization is added later.
    parser.add_argument("--attention_dim", type=int, default=32)
    parser.add_argument("--mask_temperature", type=float, default=1.0)
    parser.add_argument("--mask_init_logit", type=float, default=0.3)

    parser.add_argument(
        "--freeze_backbone",
        #default=True
        action="store_true",
    )

    return parser.parse_args()


def build_pretrained_model(
    dataset,
    metapaths,
    checkpoint,
    metadata,
    device,
):
    """Initialize the backbone with pretrained parameters."""

    model = PretrainModel(
        num_deposits=dataset.num_deposits,
        num_minerals=dataset.num_minerals,
        embedding_dim=metadata["embedding_dim"],
        num_deposit_metapaths=len(
            metapaths["deposit"]["matrices"]
        ),
        num_mineral_metapaths=len(
            metapaths["mineral"]["matrices"]
        ),
        num_metapath_layers=metadata["metapath_layers"],
        num_graph_layers=metadata["graph_layers"],
        bottleneck_hidden_dim=metadata["bottleneck_hidden_dim"],
        projection_dim=metadata["projection_dim"],
        min_mask_ratio=metadata["min_mask_ratio"],
        max_mask_ratio=metadata["max_mask_ratio"],
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return model


def load_label_split(
    data_dir: Path,
    task: str,
    ratio: float,
):
    """Load the predefined train/test split."""

    split_dir = (
        data_dir
        / task
        / f"ratio_{ratio:.1f}"
    )

    with (split_dir / "train.pkl").open("rb") as file:
        train_data = pickle.load(file)

    with (split_dir / "test.pkl").open("rb") as file:
        test_data = pickle.load(file)

    return train_data, test_data

def extract_pretrained_deposit_components(
    model: PretrainModel,
    graph: torch.Tensor,
):
    """Extract fixed Deposit ID embedding and clean DM anchor."""
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    with torch.no_grad():
        deposit_embedding = (
            model.deposit_embedding.weight.detach()
        )

        deposit_anchor, _ = model.anchor_encoder(
            model.deposit_embedding.weight,
            model.mineral_embedding.weight,
            graph,
        )

    return (
        deposit_embedding.detach(),
        deposit_anchor.detach(),
    )


def run_experiment(
    task,
    ratio,
    args,
    dataset,
    metapaths,
    graph,
    checkpoint,
    metadata,
    device,
):
    """Run one deposit classification experiment."""

    deposit_metapaths = metapaths["deposit"]["matrices"]
    mineral_metapaths = metapaths["mineral"]["matrices"]

    # Each experiment starts from the same pretrained checkpoint.
    model = build_pretrained_model(
        dataset=dataset,
        metapaths=metapaths,
        checkpoint=checkpoint,
        metadata=metadata,
        device=device,
    )

    # Initialize clean meta-path graphs and richness.
    model.prepare_metapaths(
        deposit_metapaths,
        mineral_metapaths,
    )

    (
        deposit_embedding,
        deposit_anchor,
    ) = extract_pretrained_deposit_components(
        model=model,
        graph=graph,
    )

    model.eval()
    with torch.no_grad():
        deposit_final_emb, _ = model.encode(
                        deposit_metapaths,
                        mineral_metapaths,
                        graph,
                    )
        deposit_final_emb = deposit_final_emb.detach()
    # Load the predefined label split.
    train_data, test_data = load_label_split(
        data_dir=Path(args.data_dir),
        task=task,
        ratio=ratio,
    )

    num_classes = int(
        max(
            train_data.labels.max(),
            test_data.labels.max(),
        ) + 1
    )


    adapter = DepositClassificationAdapter(
        deposit_metapaths=deposit_metapaths,
        embedding_dim=metadata["embedding_dim"],
        num_classes=num_classes,
        num_metapath_layers=metadata["metapath_layers"],
        attention_dim=args.attention_dim,
        mask_temperature=args.mask_temperature,
        mask_init_logit=args.mask_init_logit,
    ).to(device)

    optimizer = torch.optim.Adam(
        adapter.parameters(),
        lr=args.lr,
    )


    trainer = DepositClassificationTrainer(
        adapter=adapter,
        deposit_embedding=deposit_embedding,
        deposit_anchor=deposit_anchor,
        deposit_final_embs=deposit_final_emb,
        train_data=train_data,
        test_data=test_data,
        optimizer=optimizer,
        device=device,
    )

    print(
        f"\nTask: {task} | "
        f"Ratio: {ratio:.1f} | "
        f"Train: {len(train_data.deposit_ids)} | "
        f"Test: {len(test_data.deposit_ids)}"
    )

    metrics = trainer.fit(
        epochs=args.epochs,
    )

    return {
        "label": task,
        "train_ratio": ratio,
        "train_size": len(train_data.deposit_ids),
        "test_size": len(test_data.deposit_ids),
        **metrics,
    }


def main():
    args = parse_args()

    device = torch.device(
        args.device
        if torch.cuda.is_available()
        else "cpu"
    )

    data_dir = Path(args.data_dir)
    result_dir = Path(args.result_dir)

    # Load the common graph data once.
    dataset = DepositMineralDataset(data_dir)
    metapaths = load_all_metapaths(data_dir)

    deposit_metapaths = [
        adj.to(device)
        for adj in metapaths["deposit"]["matrices"]
    ]

    mineral_metapaths = [
        adj.to(device)
        for adj in metapaths["mineral"]["matrices"]
    ]

    metapaths["deposit"]["matrices"] = deposit_metapaths
    metapaths["mineral"]["matrices"] = mineral_metapaths

    graph = to_torch_sparse(
        dataset.graph
    ).to(device)

    # Load pretrained model configuration.
    checkpoint = read_checkpoint(
        args.checkpoint,
        device="cpu",
    )
    
    metadata = checkpoint["metadata"]

    results = []

    for task in CLASSIFICATION_TASKS:
        for ratio in TRAIN_RATIOS:
            set_seed(args.seed)
            result = run_experiment(
                task=task,
                ratio=ratio,
                args=args,
                dataset=dataset,
                metapaths=metapaths,
                graph=graph,
                checkpoint=checkpoint,
                metadata=metadata,
                device=device,
            )

            results.append(result)


    # Save results.
    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    save_json(
        result_dir / "deposit_classification.json",
        {
            "seed": args.seed,
            "freeze_backbone": args.freeze_backbone,
            "tasks": CLASSIFICATION_TASKS,
            "train_ratios": TRAIN_RATIOS,
            "results": results,
        },
    )

    for result in results:
        append_result(
            result_dir / "summary.csv",
            {
                "stage": "finetune",
                "task": "deposit_classification",
                "label": result["label"],
                "train_ratio": result["train_ratio"],
                "train_size": result["train_size"],
                "test_size": result["test_size"],
                "accuracy": result["accuracy"],
                "macro_f1": result["macro_f1"],
                "auc": result["auc"],
            },
        )

    print("\nDeposit classification completed.")
    print(
        f"Results: "
        f"{result_dir / 'deposit_classification.json'}"
    )
    print(
        f"Summary: "
        f"{result_dir / 'summary.csv'}"
    )


if __name__ == "__main__":
    main()