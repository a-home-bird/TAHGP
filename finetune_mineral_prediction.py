
import argparse
from pathlib import Path

import torch

from data.dataset import DepositMineralDataset
from data.metapath_loader import load_all_metapaths, to_torch_sparse
from data.sampler import BPRSampler
from models.heads.mineral_prediction import MineralPredictionHead
from models.pretrain_model import PretrainModel
from trainers.mineral_prediction_trainer import MineralPredictionTrainer
from utils.checkpoint import read_checkpoint, save_checkpoint
from utils.result_logger import append_result, save_json
from utils.seed import set_seed
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune potential mineral prediction"
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

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--reg_weight", type=float, default=1e-4)

    parser.add_argument("--top_k", type=list, default=[5,10,20])
    parser.add_argument("--log_interval", type=int, default=5)

    parser.add_argument(
        "--eval_only",
        #default=True,
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
    """Initialize the backbone from the pretrained checkpoint."""

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


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )

    data_dir = Path(args.data_dir)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 1. Load graph data
    # --------------------------------------------------
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

    graph = to_torch_sparse(dataset.graph).to(device)

    # --------------------------------------------------
    # 2. Load pretrained backbone
    # --------------------------------------------------
    checkpoint = read_checkpoint(
        args.checkpoint,
        device="cpu",
    )

    metadata = checkpoint["metadata"]

    model = build_pretrained_model(
        dataset=dataset,
        metapaths=metapaths,
        checkpoint=checkpoint,
        metadata=metadata,
        device=device,
    )

    model.prepare_metapaths(
        deposit_metapaths,
        mineral_metapaths,
    )

    # --------------------------------------------------
    # 3. Prediction head and optimizer
    # --------------------------------------------------
    head = MineralPredictionHead().to(device)

    sampler = BPRSampler(dataset)
    parameters = (
                list(model.parameters())
                + list(head.parameters())
            )
    optimizer = torch.optim.Adam(
        parameters,
        lr=args.lr,
    )


    trainer = MineralPredictionTrainer(
        model=model,
        head=head,
        dataset=dataset,
        sampler=sampler,
        deposit_metapaths=deposit_metapaths,
        mineral_metapaths=mineral_metapaths,
        graph=graph,
        optimizer=optimizer,
        device=device,
        batch_size=args.batch_size,
        reg_weight=args.reg_weight,
        top_k=args.top_k,
    )

    # --------------------------------------------------
    # 4. Fine-tuning
    # --------------------------------------------------
    final_loss = None

    if not args.eval_only:
        for epoch in range(1, args.epochs + 1):
            final_loss = trainer.train_epoch()

            if (
                epoch % args.log_interval == 0
                or epoch == args.epochs
            ):
                print(
                    f"Epoch {epoch:04d} | "
                    f"train_loss={final_loss:.4f}"
                )

        save_checkpoint(
            result_dir / "mineral_prediction.pt",
            model,
            metadata={
                **metadata,
                "task": "mineral_prediction",
                "finetune_epochs": args.epochs,
            },
        )

    # --------------------------------------------------
    # 5. Final test evaluation
    # --------------------------------------------------
    metrics = trainer.evaluate()

    for index,k in enumerate(args.top_k):
        recall = metrics[index][f"recall@{k}"]
        ndcg = metrics[index][f"ndcg@{k}"]

        print(
            f"\nTest | "
            f"Recall@{k}={recall:.4f} | "
            f"NDCG@{k}={ndcg:.4f}"
        )

    # --------------------------------------------------
    # 6. Save results
    # --------------------------------------------------
    result = {
        "task": "mineral_prediction",
        "mode": (
            "eval_only" if args.eval_only else "finetune"
        ),
        "seed": args.seed,
        "epochs": 0 if args.eval_only else args.epochs,
        "train_size": len(dataset),
        "test_size": sum(
            len(items)
            for items in dataset.test_dict.values()
        ),
        "top_k": args.top_k,
        "train_loss": final_loss,
        "metrics":metrics,
    }

    save_json(
        result_dir / "mineral_prediction.json",
        result,
    )


    print("\nMineral prediction completed.")
    print(
        f"Results: "
        f"{result_dir / 'mineral_prediction.json'}"
    )
    print(
        f"Summary: "
        f"{result_dir / 'summary.csv'}"
    )


if __name__ == "__main__":
    main()