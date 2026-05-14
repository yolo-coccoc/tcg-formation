import sys
sys.path.insert(0, '../lib')
#import wandb
import os
import torch
import json
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from Progress._Progress import CrossEntropyLoss_base
from Progress.Callback import save_checkpoint_callback
from Dataset.Merra2_dataset import LData, Merra2Full
from Model.ResNet import ResNet18_Classification
from Utils.Seed import set_all_seeds
from Utils.Metrics import *


def main(args):
    project = args.project

    seed = args.seed
    step = args.step
    ratio = args.ratio
    pos_weight = args.weight

    batch_size = args.batch_size
    num_workers = args.num_workers
    learning_rate = args.learning_rate
    max_epochs = args.max_epochs

    mode = args.mode
    checkpoint = args.checkpoint
    inp_dir = args.inp_dir
    out_dir = args.out_dir
    
    set_all_seeds(seed)
    out_dir = os.path.join(out_dir, f'{project}_r{ratio}_w{pos_weight}', f'Step_{step}')
    if mode < 0:
        version = 0
        while True:
            if os.path.isdir(out_dir + f'_v{version}'):
                version += 1
            else:
                # version = 5
                out_dir = out_dir + f'_v{version}'
                break
        
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(out_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, indent=4, ensure_ascii=False)
        
    else:
        version = mode
        out_dir = out_dir + f'_v{version}'
    print(f'Output directory: {out_dir}')

    #===================
    # Prepare data phase
    #===================
    inp_dir = os.path.join(inp_dir.format(ratio), f'Step_{step}')
    ds = LData(dataset_class=Merra2Full,
            # train_path = os.path.join(inp_dir, 'train.csv'),
            # val_path = os.path.join(inp_dir, 'val.csv'),
            # test_path = os.path.join(inp_dir, 'testRus.csv'),
            train_path = '/N/slate/tnn3/DucHGA/TC-formation/Data/Merra/data_path_check.csv',
            val_path = '/N/slate/tnn3/DucHGA/TC-formation/Data/Merra/data_path_check.csv',
            test_path = '/N/slate/tnn3/DucHGA/TC-formation/Data/Merra/data_path_check.csv',
            predict_path = '/N/slate/tnn3/DucHGA/TC-formation/Data/Merra/data_path_check.csv',
            # ratio = ratio,
            batch_size = batch_size,
            num_workers = num_workers,)
    print(f'Dataset prepared')

    #====================
    # Model phase
    #====================
    model = ResNet18_Classification(inp_channels=278,
                num_classes=2,)
    print(f'Model prepared')

    #====================
    # Loss function phase
    #====================
    if pos_weight > 0:
        class_weight = np.array([
            (pos_weight + 1) / (2 * pos_weight),
            (pos_weight + 1) / 2
        ])
    print(f'Loss function prepared with class weights: {class_weight}')

    #====================
    # Train phase
    #====================
    if mode < 0:
        os.environ["WANDB_API_KEY"] = '3b59eddf5201c6c82ed66a6f97c3b2a813ba8929'
        #wandb.login()
        #wandb_logger = WandbLogger(project=project,
        #                           name=f'Ver{version}_Baseline_r{ratio}_w{pos_weight}_s{seed}')
        
        trainer = Trainer(#logger = wandb_logger,
                    log_every_n_steps = 100,
                    max_epochs = max_epochs, 
                    callbacks = save_checkpoint_callback(out_dir),
                    accelerator='auto',
                    devices=1,)
        
        L_model = CrossEntropyLoss_base(model,
                                        class_weight = class_weight,
                                        learning_rate = learning_rate,
                                        out_dir = out_dir,)
        trainer.fit(L_model, datamodule=ds)
        del trainer

if __name__ == "__main__":
    #=========================
    # Path and config phase
    #=========================

    parser = ArgumentParser()
    parser.add_argument("--project", type=str, default = 'test')

    # dataset argument
    parser.add_argument("--seed", type=int, default = 42)
    parser.add_argument("--step", type=int, default = 2)
    parser.add_argument("--ratio", type=int, default = 30)
    parser.add_argument("--weight", type=int, default = 6)

    # trainer argument
    parser.add_argument("--batch_size", type=int, default = 64)
    parser.add_argument("--num_workers", type=int, default = 2)
    parser.add_argument("--learning_rate", type=float, default = 1e-4)
    parser.add_argument("--weight_decay", type=int, default = 0.01)
    parser.add_argument("--max_epochs", type=int, default = 2)

    # mode argument
    parser.add_argument("--mode", type=int, default = -1)
    parser.add_argument("--checkpoint", type=str, nargs='*', default=('last', 'best_f1s', 'best_loss'))

    # input and output paths
    parser.add_argument("--inp_dir", type=str, default = '/N/slate/tnn3/DucHGA/TC-formation/Data/Merra/Dataset/Rus{}')
    parser.add_argument("--out_dir", type=str, default = '/N/slate/tnn3/DucHGA/TC-formation/OutTemp')

    args = parser.parse_args()
    
    main(args)
