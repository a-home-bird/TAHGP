import argparse
from pathlib import Path

import torch

from data.dataset import DepositMineralDataset
from data.metapath_loader import load_all_metapaths
from losses.pretrain_objective import PretrainObjective
from models.pretrain_model import PretrainModel
from trainers.pretrain_trainer import PretrainTrainer
from utils.checkpoint import save_checkpoint
from utils.result_logger import append_result, save_json
from utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Heterogeneous graph pre-training"
    )

    # Data
    parser.add_argument(
        "--data_dir",
        type=str,
        default="dataset/",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/experiment_test_seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    # Backbone
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--metapath_layers",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--graph_layers",
        type=int,
        default=2,
    )

    # Semantic bottleneck
    parser.add_argument(
        "--bottleneck_hidden_dim",
        type=int,
        default=8,
    )

    # Contrastive learning
    parser.add_argument(
        "--projection_dim",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
    )

    # Dynamic meta-path masking
    parser.add_argument(
        "--min_mask_ratio",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--max_mask_ratio",
        type=float,
        default=0.30,
    )

    # Joint objective
    parser.add_argument(
        "--rec_weight",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--contrastive_weight",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--ib_weight",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--negative_ratio",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--reg_weight",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--scatter_weight",
        type=float,
        default=1e-6,
    )

    # Training
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-3,
    )

    # Validation
    parser.add_argument(
        "--validation_ratio",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
    )

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(
        args.device
        if torch.cuda.is_available()
        else "cpu"
    )

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Data
    # --------------------------------------------------
    dataset = DepositMineralDataset(
        data_dir
    )

    
    metapaths = load_all_metapaths(
        data_dir
    )

    

    deposit_metapaths = (
        metapaths["deposit"]["matrices"]
    )
    mineral_metapaths = (
        metapaths["mineral"]["matrices"]
    )

    print(
        "\nSelected Deposit meta-paths:"
    )

    for name in metapaths["deposit"]["names"]:
        print(f"  {name}")


    print(
        "\nSelected Mineral meta-paths:"
    )

    for name in metapaths["mineral"]["names"]:
        print(f"  {name}")
    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    model = PretrainModel(
        num_deposits=dataset.num_deposits,
        num_minerals=dataset.num_minerals,
        embedding_dim=args.embedding_dim,
        num_deposit_metapaths=len(
            deposit_metapaths
        ),
        num_mineral_metapaths=len(
            mineral_metapaths
        ),
        num_metapath_layers=(
            args.metapath_layers
        ),
        num_graph_layers=(
            args.graph_layers
        ),
        bottleneck_hidden_dim=(
            args.bottleneck_hidden_dim
        ),
        projection_dim=(
            args.projection_dim
        ),
        min_mask_ratio=(
            args.min_mask_ratio
        ),
        max_mask_ratio=(
            args.max_mask_ratio
        ),
    ).to(device)

    # --------------------------------------------------
    # Joint objective
    # --------------------------------------------------
    objective = PretrainObjective(
        rec_weight=args.rec_weight,
        contrastive_weight=(
            args.contrastive_weight
        ),
        ib_weight=args.ib_weight,
        temperature=args.temperature,
        negative_ratio=(
            args.negative_ratio
        ),
        scatter_weight=args.scatter_weight,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    # --------------------------------------------------
    # Trainer
    # --------------------------------------------------
    
    trainer = PretrainTrainer(
        model=model,
        objective=objective,
        dataset=dataset,
        deposit_metapaths=deposit_metapaths,
        mineral_metapaths=mineral_metapaths,
        optimizer=optimizer,
        device=device,
        batch_size=args.batch_size,
        top_k=args.top_k,
        validation_ratio=(
            args.validation_ratio
        ),
        reg_weight=args.reg_weight,
    )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------
    result = trainer.fit(
        epochs=args.epochs,
        eval_interval=args.eval_interval,
        patience=args.patience,
    )

    # --------------------------------------------------
    # Checkpoint
    # --------------------------------------------------
    checkpoint_path = (
        output_dir / "pretrained.pt"
    )

    metadata = {
        "num_deposits":
            dataset.num_deposits,
        "num_minerals":
            dataset.num_minerals,

        "embedding_dim":
            args.embedding_dim,
        "metapath_layers":
            args.metapath_layers,
        "graph_layers":
            args.graph_layers,

        "bottleneck_hidden_dim":
            args.bottleneck_hidden_dim,
        "projection_dim":
            args.projection_dim,

        "min_mask_ratio":
            args.min_mask_ratio,
        "max_mask_ratio":
            args.max_mask_ratio,

        "deposit_metapaths":
            metapaths["deposit"]["names"],
        "mineral_metapaths":
            metapaths["mineral"]["names"],

        "seed":
            args.seed,
    }

    save_checkpoint(
        checkpoint_path,
        model,
        metadata=metadata,
    )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------
    train_losses = result[
        "train_losses"
    ]

    pretrain_result = {
        "seed": args.seed,

        "best_epoch":
            result["best_epoch"],

        "validation_ratio":
            args.validation_ratio,

        "top_k":
            args.top_k,

        f"val_recall@{args.top_k}":
            result[
                f"val_recall@{args.top_k}"
            ],

        f"val_ndcg@{args.top_k}":
            result[
                f"val_ndcg@{args.top_k}"
            ],

        "loss": {
            "total":
                train_losses.get(
                    "total"
                ),
            "bpr":
                train_losses.get(
                    "bpr"
                ),
            "reconstruction":
                train_losses.get(
                    "reconstruction"
                ),
            "contrastive":
                train_losses.get(
                    "contrastive"
                ),
            "information_bottleneck":
                train_losses.get(
                    "information_bottleneck"
                ),
            "regularization":
                train_losses.get(
                    "regularization"
                ),
        },

        "loss_weights": {
            "reconstruction":
                args.rec_weight,
            "contrastive":
                args.contrastive_weight,
            "information_bottleneck":
                args.ib_weight,
            "regularization":
                args.reg_weight,
        },

        "dynamic_mask": {
            "min_ratio":
                args.min_mask_ratio,
            "max_ratio":
                args.max_mask_ratio,
        },

        "temperature":
            args.temperature,

        "negative_ratio":
            args.negative_ratio,

        "deposit_metapath_weights": {
            name: weight.item()
            for name, weight in zip(
                metapaths[
                    "deposit"
                ]["names"],
                model
                .deposit_metapath_weights
                .detach()
                .cpu(),
            )
        },

        "mineral_metapath_weights": {
            name: weight.item()
            for name, weight in zip(
                metapaths[
                    "mineral"
                ]["names"],
                model
                .mineral_metapath_weights
                .detach()
                .cpu(),
            )
        },
    }

    save_json(
        output_dir / "pretrain.json",
        pretrain_result,
    )

    append_result(
        output_dir / "summary.csv",
        {
            "stage": "pretrain",
            "task": "pretrain",
            "seed": args.seed,
            "best_epoch":
                result["best_epoch"],
            "loss":
                train_losses.get(
                    "total"
                ),
            f"recall@{args.top_k}":
                result[
                    f"val_recall@{args.top_k}"
                ],
            f"ndcg@{args.top_k}":
                result[
                    f"val_ndcg@{args.top_k}"
                ],
        },
    )

    # --------------------------------------------------
    # Print
    # --------------------------------------------------
    print("\nPre-training completed.")

    print(
        f"Best epoch: "
        f"{result['best_epoch']}"
    )

    print(
        f"Validation Recall@{args.top_k}: "
        f"{result[f'val_recall@{args.top_k}']:.4f}"
    )

    print(
        f"Validation NDCG@{args.top_k}: "
        f"{result[f'val_ndcg@{args.top_k}']:.4f}"
    )

    print("\nDeposit meta-path weights:")

    for name, weight in zip(
        metapaths["deposit"]["names"],
        model
        .deposit_metapath_weights
        .detach()
        .cpu(),
    ):
        print(
            f"  {name}: "
            f"{weight.item():.4f}"
        )

    print("\nMineral meta-path weights:")

    for name, weight in zip(
        metapaths["mineral"]["names"],
        model
        .mineral_metapath_weights
        .detach()
        .cpu(),
    ):
        print(
            f"  {name}: "
            f"{weight.item():.4f}"
        )

    print(
        f"\nCheckpoint: "
        f"{checkpoint_path}"
    )

    print(
        f"Results: "
        f"{output_dir / 'pretrain.json'}"
    )


if __name__ == "__main__":
    main()