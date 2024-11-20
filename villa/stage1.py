import os

import hydra
import pyrootutils
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from rich import print
from torch.utils.data import DataLoader

from villa.utils.mapping import mapping
from villa.utils.train import train
from villa.utils.utils import set_seed

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)
config_dir = os.path.join(root, "villa", "configs")


@hydra.main(version_base="1.2", config_path=config_dir, config_name="default.yaml")
def main(cfg: DictConfig):
    cfg = instantiate(cfg)
    print(f"=> Starting (experiment={cfg.task_name})")

    # Set seed
    seed = cfg.get("seed", None)
    if seed is not None:
        set_seed(seed)

    # Initialize model, loss, and optimizer
    model = cfg.model.to("cuda")
    loss = cfg.loss
    opt = cfg.optimizer(model.parameters())
    print(
        f"=> Using model {type(model).__name__} with loss {type(loss).__name__} on {torch.cuda.device_count()} GPUs"
    )

    # Initialize dataloaders
    train_loader = DataLoader(**cfg.dataloader.train)
    val_loader = DataLoader(**cfg.dataloader.val)

    # Create checkpoint directory
    checkpoint_dir = os.path.join(cfg.checkpoint_dir, cfg.task_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Support for multi-GPU training
    model = torch.nn.DataParallel(model)

    # Train Stage 1 model
    print("=> Training ViLLA: Stage 1")
    train(
        model,
        loss,
        opt,
        None,
        train_loader,
        val_loader,
        cfg.epochs,
        cfg.batch_size,
        cfg.task_name,
        checkpoint_dir,
        os.path.join(config_dir, "experiment"),
        False,
    )

    # Update dataloader parameters for inference
    cfg.dataloader.train.batch_size = 1
    cfg.dataloader.train.shuffle = False
    cfg.dataloader.train.drop_last = False
    cfg.dataloader.val.batch_size = 1
    train_loader = DataLoader(**cfg.dataloader.train)
    val_loader = DataLoader(**cfg.dataloader.val)

    # Generate mappings between regions and attributes
    mapping(
        model,
        train_loader,
        "train",
        cfg.model.one_proj,
        cfg.data_dir,
        checkpoint_dir,
        cfg.mapping.epsilon,
    )
    mapping(
        model,
        val_loader,
        "val",
        cfg.model.one_proj,
        cfg.data_dir,
        checkpoint_dir,
        cfg.mapping.epsilon,
    )


if __name__ == "__main__":
    main()
