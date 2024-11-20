import os
import shutil
import time
from datetime import datetime

import numpy as np
import torch
from rich import print
from torch.utils.tensorboard import SummaryWriter


def _save_config_file(model_checkpoints_folder, config_file_path, task_name):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
    destination = os.path.join(model_checkpoints_folder, f"{task_name}.yaml")
    shutil.copy(os.path.join(config_file_path, f"{task_name}.yaml"), destination)


def train(
    model,
    loss_fn,
    opt,
    scheduler,
    train_loader,
    val_loader,
    epochs,
    batch_size,
    task_name,
    checkpoint_dir,
    config_file_path,
    early_stop=False,
):
    """
    Training loop.

    Parameters:
        model (torch.nn.Module): Model
        loss_fn (torch.nn.Module): Loss function
        opt (torch.optim): Optimizer
        scheduler (torch.optim.lr_scheduler): LR scheduler (set to None if no scheduler)
        train_loader (torch.utils.data.DataLoader): Dataloader for training data
        val_loader (torch.utils.data.DataLoader): Dataloader for validation data
        epochs (int): Number of training epochs
        batch_size (int): Batch size
        checkpoint_dir (str): Directory for storing model weights
        early_stop (bool): True if early stopping based on validation loss
    """
    # Initialize TensorBoard Writer
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(
        os.path.join(checkpoint_dir, "runs"), f"{task_name}_{timestamp}"
    )
    writer = SummaryWriter(log_dir=log_dir)

    scaler = torch.cuda.amp.GradScaler()
    epochs_no_improvements = 0
    best_val_loss = np.inf

    # Checkpoint folder
    model_checkpoints_folder = os.path.join(checkpoint_dir, writer.log_dir)

    # Save config file
    _save_config_file(
        model_checkpoints_folder,
        config_file_path,
        task_name,
    )

    for epoch in range(0, epochs):
        model.train()
        time_start = time.time()

        for step, sample in enumerate(train_loader):
            opt.zero_grad()

            with torch.cuda.amp.autocast(enabled=True):
                pred = model(sample)

            loss = loss_fn(pred, sample)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            # Log training metric to tensorboard
            writer.add_scalar(
                "Train Loss", loss.item(), epoch * len(train_loader) + step
            )
            writer.add_scalar(
                "Learning Rate",
                opt.param_groups[0]["lr"],
                epoch * len(train_loader) + step,
            )

            summary = (
                "\r[Epoch {}][Step {}/{}] Loss: {}, Lr: {} - {:.2f} m remaining".format(
                    epoch + 1,
                    step,
                    int(len(train_loader.dataset) / batch_size),
                    "{}: {:.2f}".format(
                        type(loss_fn).__name__, loss_fn.mean_running_loss
                    ),
                    *[group["lr"] for group in opt.param_groups],
                    ((time.time() - time_start) / (step + 1))
                    * ((len(train_loader.dataset) / batch_size) - step)
                    / 60,
                )
            )
            print(summary)

        # Log the average loss for the epoch
        writer.add_scalar("Average Train Loss", loss_fn.mean_running_loss, epoch)

        time_end = time.time()
        elapse_time = time_end - time_start
        print("Finished in {}s".format(int(elapse_time)))

        torch.save(
            model.state_dict(), os.path.join(model_checkpoints_folder, "last.pkl")
        )

        if early_stop:
            val_loss = evaluate(model, loss_fn, val_loader)
            if val_loss < best_val_loss:
                epochs_no_improvements = 0
                print("Saving best model")
                torch.save(
                    model.state_dict(),
                    os.path.join(model_checkpoints_folder, "best.pkl"),
                )
                best_val_loss = val_loss
            else:
                epochs_no_improvements += 1
                print(f"Validation With No Improvements: {epochs_no_improvements}")

            if scheduler:
                scheduler.step(val_loss)

            if epochs_no_improvements == 5:
                print("Early stop reached")
                print(f"Validation With No Improvements: {epochs_no_improvements}")
                return

    # Close TensorBoard Writer
    writer.close()


def evaluate(model, loss_fn, val_loader, split="val", writer=None, epoch=None):
    """
    Validation loop.

    Parameters:
        model (torch.nn.Module): Model
        loss_fn (torch.nn.Module): Loss function
        val_loader (torch.utils.data.DataLoader): Dataloader for validation data
        split (str): Evaluation split
    Returns:
        loss (torch.Tensor): Validation loss
    """
    print(f"Evaluating on {split}")
    model.eval()

    running_loss = 0
    num_batches = 0
    with torch.no_grad():
        for step, sample in enumerate(val_loader):
            pred = model(sample)

            running_loss += loss_fn(pred, sample)
            num_batches += 1

            if writer is not None and epoch is not None:
                writer.add_scalar(
                    "Val Loss", running_loss.item(), epoch * len(val_loader) + step
                )

    loss = running_loss / num_batches
    print(f"Loss = {loss}")

    # Log the average validation loss to TensorBoard
    if writer is not None and epoch is not None:
        writer.add_scalar("Average Val Loss", loss, epoch)

    return loss
