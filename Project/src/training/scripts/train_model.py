import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import pprint
import wandb

wandb.login()

from src.data_pipeline.preprocessing.preprocessing import preprocessing
from src.training.utils.get_loss_function import get_loss_function
from src.training.utils.get_lr_scheduler import get_lr_scheduler
from src.training.utils.get_model import get_model
from src.training.utils.get_optimizer import get_optimizer
from src.training.utils.load_config import load_config
from src.training.utils.mixed_precision import get_mixed_precision

scaler = get_mixed_precision()


def train_per_epoch(epoch):
    global model, optimizer, loss_function, scheduler, writer, train_loader

    model.train()
    train_loss = 0.0

    for batch_index, (train_images, train_labels) in enumerate(
        tqdm(train_loader, desc=f"Epoch {epoch}")
    ):
        train_images, train_labels = train_images.to(device), train_labels.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda'):
            outputs = model(train_images)
            loss = loss_function(outputs, train_labels)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        n_iter = (epoch - 1) * len(train_loader) + batch_index + 1
        writer.add_scalar("Train/Loss", loss.item(), n_iter)

        # W&B
        wandb.log({"train_loss": loss.item()})

        train_loss += loss.item()

    scheduler.step()

    return train_loss / len(train_loader)


@torch.no_grad()
def eval_training(epoch=0):
    global model, loss_function, test_loader, writer

    model.eval()
    test_loss = 0.0
    correct = 0

    for test_images, test_labels in test_loader:
        test_images, test_labels = test_images.to(device), test_labels.to(device)

        outputs = model(test_images)
        test_loss += loss_function(outputs, test_labels).item()

        correct += (outputs.argmax(1) == test_labels).sum().item()

    avg_loss = test_loss / len(test_loader)
    accuracy = 100.0 * correct / len(test_loader.dataset)

    print(f"Epoch {epoch}: Test Loss {avg_loss:.3f}, Accuracy {accuracy:.3f}%")

    writer.add_scalar("Test/Loss", avg_loss, epoch)
    writer.add_scalar("Test/Accuracy", accuracy, epoch)

    wandb.log({"val_loss": avg_loss, "val_acc": accuracy})

    return accuracy


def sweep_train():
    global model, train_loader, test_loader, writer, loss_function, optimizer, scheduler, device

    run = wandb.init(project="ACNN-project")
    config_wb = wandb.config

    device = config['experiment']['device']

    experiment_number = config['experiment']['number']

    # override with sweep hyperparameters
    config['training']['learning_rate'] = config_wb.learning_rate
    config['training']['weight_decay'] = config_wb.weight_decay
    config['dataset']['batch_size'] = config_wb.batch_size
    config['training']['optimizer'] = config_wb.optimizer

    train_loader, test_loader = preprocessing(config)
    model = get_model(config['model']['name'], config).to(device)
    loss_function = get_loss_function(config["training"]["loss_function"])
    optimizer = get_optimizer(config, model.parameters())
    scheduler = get_lr_scheduler(config, optimizer)

    log_dir = f'../experiments/experiment{experiment_number}/results'
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(
        os.path.join(log_dir, datetime.now().strftime(config["experiment"]["date_format"]))
    )

    best_acc = 0
    for epoch in range(1, config['training']['epochs'] + 1):
        train_per_epoch(epoch)
        acc = eval_training(epoch)

        if acc > best_acc:
            best_acc = acc
            checkpoint_dir = f'../experiments/experiment{experiment_number}/checkpoints'
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(
                checkpoint_dir, f'best_model_{best_acc}.pth'
            ))

    writer.close()
    run.finish()


if __name__ == '__main__':
    print("Type the number of the experiment you want to run:")
    experiment_number = int(input())
    config = load_config(f"../experiments/experiment{experiment_number}/config.yml")

    pprint.pprint(f"Sweep configuration: {config['sweep']}")

    sweep_id = wandb.sweep(config['sweep'], project="ACNN-project")
    wandb.agent(sweep_id, function=sweep_train, count=7)
