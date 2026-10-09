import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict,List,Tuple

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run pretrain and downstream fine-tuning pipeline"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="dataset/",
    )
    parser.add_argument(
        "--result_root",
        type=str,
        default="results",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
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

    # Pretrain
    parser.add_argument(
        "--pretrain_epochs",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--pretrain_lr",
        type=float,
        default=5e-3,
    )

    # Mineral prediction
    parser.add_argument(
        "--mineral_epochs",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--mineral_lr",
        type=float,
        default=1e-3,
    )

    # Deposit classification
    parser.add_argument(
        "--classification_epochs",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--classification_lr",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--freeze_backbone",
        default=True
    )

    return parser.parse_args()


def run_command(
    command: List[str],
    log_file: Path,
) -> None:
    print("\n" + "=" * 80)
    print(" ".join(command))
    print("=" * 80)

    with log_file.open(
        "a",
        encoding="utf-8",
    ) as file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in process.stdout:
            print(line, end="")
            file.write(line)
            file.flush()

        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"Command failed with return code {return_code}"
        )


def main():
    args = parse_args()

    project_dir = Path(__file__).resolve().parent

    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        experiment_name = (
            f"seed_{args.seed}_{timestamp}"
        )

    result_dir = (
        Path(args.result_root)
        / experiment_name
    )
    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = (
        result_dir
        / "pretrained.pt"
    )

    log_file = (
        result_dir
        / "pipeline.log"
    )

    python = sys.executable

    # --------------------------------------------------
    # 1. Pretrain
    # --------------------------------------------------
    pretrain_command = [
        python,
        str(project_dir / "pretrain.py"),
        "--data_dir",
        args.data_dir,
        "--output_dir",
        str(result_dir),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--epochs",
        str(args.pretrain_epochs),
        "--lr",
        str(args.pretrain_lr),
        "--rec_weight",
        str(0.1),
        "--contrastive_weight",
        str(0.0),
        "--ib_weight",
        str(1e-3),
        "--scatter_weight",
        str(1e-6),
    ]

    run_command(
        pretrain_command,
        log_file,
    )

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {checkpoint}"
        )

    # --------------------------------------------------
    # 2. Potential mineral prediction
    # --------------------------------------------------
    mineral_command = [
        python,
        str(
            project_dir
            / "finetune_mineral_prediction.py"
        ),
        "--data_dir",
        args.data_dir,
        "--checkpoint",
        str(checkpoint),
        "--result_dir",
        str(result_dir),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--epochs",
        str(args.mineral_epochs),
        "--lr",
        str(args.mineral_lr),
    ]

    if args.freeze_backbone:
            mineral_command.append(
                "--eval_only"
            )

    run_command(
        mineral_command,
        log_file,
    )
    

    # --------------------------------------------------
    # 3. Deposit classification
    # --------------------------------------------------
    classification_command = [
        python,
        str(
            project_dir
            / "finetune_deposit_classification.py"
        ),
        "--data_dir",
        args.data_dir,
        "--checkpoint",
        str(checkpoint),
        "--result_dir",
        str(result_dir),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--epochs",
        str(args.classification_epochs),
        "--lr",
        str(args.classification_lr),
        "--dropout",
        str(args.dropout),
    ]

    if args.freeze_backbone:
        classification_command.append(
            "--freeze_backbone"
        )

    run_command(
        classification_command,
        log_file,
    )

    # --------------------------------------------------
    # Finished
    # --------------------------------------------------
    print("\n" + "=" * 80)
    print("Pipeline completed.")
    print(f"Result directory: {result_dir}")
    print(f"Checkpoint:       {checkpoint}")
    print(f"Summary:          {result_dir / 'summary.csv'}")
    print(f"Log:              {log_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()