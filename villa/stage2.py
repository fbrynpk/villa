import os

import hydra
import pyrootutils
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from rich import print
from torch.utils.data import DataLoader

from villa.utils.train import train
from villa.utils.utils import set_seed

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)
config_dir = os.path.join(root, "villa", "configs")


@hydra.main(version_base="1.2", config_path=config_dir, config_name="default.yaml")
def main(cfg: DictConfig):
    cfg = instantiate(cfg)
    print(cfg)
    print(f"=> Starting (experiment={cfg.task_name})")

    # Set seed
    seed = cfg.get("seed", None)
    if seed is not None:
        set_seed(seed)

    # Initialize model, loss, optimizer, and scheduler
    model = cfg.model.to("cuda")
    loss = cfg.loss
    opt = cfg.optimizer(model.parameters())
    scheduler = cfg.scheduler(opt)

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

    # if os.path.isdir(checkpoint_dir):
    #     model.load_state_dict(
    #         torch.load(
    #             "C:/Users/DryLab/Desktop/villa/checkpoints/mimic_stage2/best.pkl"
    #         )
    #     )
    #     print("Last Best Model Load Succesfully!")

    # Train Stage 2 model
    print("=> Training ViLLA: Stage 2")
    train(
        model,
        loss,
        opt,
        scheduler,
        train_loader,
        val_loader,
        cfg.epochs,
        cfg.batch_size,
        cfg.task_name,
        checkpoint_dir,
        os.path.join(config_dir, "experiment"),
        True,
    )


if __name__ == "__main__":
    main()
