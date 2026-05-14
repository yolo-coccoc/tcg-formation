import json
import os
from statistics import mode
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from unittest.util import strclass

import numpy as np
import torch
import wandb

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger

sys.path.insert(0, "/N/slate/tnn3/DucHGA/TC-formation/Src_model")

from Dataset.Merra2_dataset import LData, Merra2Full
from Model.ResNet import ResNet18_Classification
from Progress.Progress import ClassificationModule
from Progress.Callback import save_checkpoint_callback
from Utils.Seed import set_all_seeds


def compute_class_weights(pos_weight: float) -> np.ndarray:
    """Compute class weights for imbalanced dataset."""
    if pos_weight <= 0:
        return None
    return np.array([(pos_weight + 1) / (2 * pos_weight),
                     (pos_weight + 1) / 2])


def setup_output_directory(args) -> str:
    """Set up output directory using formal version counting mode.
    
    Args:
        args: Command-line arguments
        out_dir: Base output directory
        project: Project name
        mode: 0 for new training, >0 for load existing trained model version
    """

    if args.mode < 0:
        version = 0
        while True:
            out_dir = os.path.join(args.out_dir, 
                                    f"{args.project}_r{args.ratio}_w{args.weight}_s{args.seed}",
                                    f"Step_{args.step}_v{version}")
            if os.path.isdir(out_dir):
                version += 1
            else:
                break

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        return out_dir
    elif args.mode >= 0:
        return os.path.join(args.out_dir, 
                            f"{args.project}_r{args.ratio}_w{args.weight}_s{args.seed}",
                            f"Step_{args.step}_v{version}")
    else:
        raise ValueError(f"Invalid mode: {args.mode}. Use 0 for new training or a positive version number for loading.")


def save_config(args, out_dir: str):
    """Save arguments to config.json with creation timestamp."""
    config = vars(args).copy()
    config["time_creation"] = datetime.now().isoformat()
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)


def setup_wandb(args, out_dir: str):
    """Initialize WandB logger."""
    os.environ["WANDB_API_KEY"] = '3b59eddf5201c6c82ed66a6f97c3b2a813ba8929'
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if not wandb_api_key:
        raise ValueError("WANDB_API_KEY environment variable not set")
    wandb.login(key=wandb_api_key)
    
    # Extract timestamp from output directory name
    run_name = os.path.basename(out_dir)
    return WandbLogger(project=args.project,
                       name=f"{run_name}_r{args.ratio}_w{args.weight}_s{args.seed}")


def main(args):
    # Set seed for reproducibility
    set_all_seeds(args.seed)

    # Setup output directory
    out_dir = setup_output_directory(args)
    print(f"Output directory: {out_dir}")

    # Prepare dataset
    ds = LData(dataset_class=Merra2Full,
               train_path=os.path.join(args.inp_dir, f"Step_{args.step}", "train.csv"),
               val_path=os.path.join(args.inp_dir, f"Step_{args.step}", "val.csv"),
               test_path=os.path.join(args.inp_dir, f"Step_{args.step}", "testFull.csv"),
               batch_size=args.batch_size,
               num_workers=args.num_workers,)
    print("Dataset prepared")

    # Prepare model
    model = ResNet18_Classification(inp_channels=278, num_classes=2)
    print("Model prepared")

    # Prepare loss function
    class_weight = compute_class_weights(args.weight)
    print(f"Class weights: {class_weight}")

    # Training phase (only if mode == 0)
    lightning_model = None
    if args.mode < 0:
        save_config(args, out_dir)
        
        wandb_logger = setup_wandb(args.project, out_dir, args.weight, args.seed)

        trainer = Trainer(
            logger=wandb_logger,
            log_every_n_steps=100,
            max_epochs=args.max_epochs,
            callbacks=save_checkpoint_callback(out_dir),
            accelerator="auto",
            devices=1,
        )
        print(class_weight)
        lightning_model = ClassificationModule(
            model,
            class_weight=class_weight,
            export_result=args.export_result,
            optimizer_kwargs={
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay,
            },
            out_dir=out_dir,
        )

        trainer.fit(lightning_model, datamodule=ds)
        print("Training completed")
    
    # Testing phase (always run all checkpoints)
    print("Testing all checkpoints...")
    test_results = {}
    
    for checkpoint_name in args.checkpoint:
        checkpoint_path = os.path.join(out_dir, "checkpoints", f"{checkpoint_name}.ckpt")
        print(f"Loading model from: {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found: {checkpoint_path}, skipping...")
            continue
        
        # Load lightning module with checkpoint
        lightning_model = ClassificationModule.load_from_checkpoint(
            checkpoint_path,
            model=model,
            class_weight=class_weight,
            export_result=f"{args.export_result}_{checkpoint_name}" if args.export_result else None,
            optimizer_kwargs={
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay,
            },
            out_dir=out_dir,
        )
        print(f"Model loaded from checkpoint: {checkpoint_name}")
        
        # Run testing
        test_trainer = Trainer(
            accelerator="auto",
            devices=1,
        )
        results = test_trainer.test(lightning_model, datamodule=ds)
        test_results[checkpoint_name] = results
        print(f"Testing completed for {checkpoint_name}")
    
    print(f"All checkpoints tested. Results saved to {out_dir}")
    return test_results


if __name__ == "__main__":
    parser = ArgumentParser(description="Train MERRA-2 classification model")

    # Project settings
    parser.add_argument("--project", type=str, default="test", help="WandB project name")

    # Dataset arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--step", type=int, default=2, help="Step size")
    # parser.add_argument("--ratio", type=int, default=30, help="Dataset ratio")
    parser.add_argument("--weight", type=float, default=6.0, help="Positive class weight")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_epochs", type=int, default=2, help="Maximum epochs")

    # Mode and paths
    parser.add_argument("--mode", type=int, default=0, help="Mode: 0 for new training, >0 for load existing model version")
    parser.add_argument("--checkpoint", type=str, nargs="*", default=("last", "best_f1s", "best_loss"), help="Checkpoints to save during training and test during evaluation")
    parser.add_argument("--inp_dir", type=str, default="./", help="Input directory")
    parser.add_argument("--out_dir", type=str, default="./", help="Output directory")
    parser.add_argument("--export_result", type=str, default=None, help="Export test results to file (xlsx)")

    args = parser.parse_args()
    
    main(args)
