import os
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import wandb
import numpy as np

#from Project.src.data_pipeline.preprocessing.preprocessing import preprocessing
#from Project.src.training.utils.get_loss_function import get_loss_function
#from Project.src.training.utils.get_lr_scheduler import get_lr_scheduler, get_current_lr
#from Project.src.training.utils.get_model import get_model
#from Project.src.training.utils.get_optimizer import get_optimizer
#from Project.src.training.utils.mixed_precision import get_mixed_precision


from src.data_pipeline.preprocessing.preprocessing import preprocessing
from src.training.utils.get_loss_function import get_loss_function
from src.training.utils.get_lr_scheduler import get_lr_scheduler, get_current_lr
from src.training.utils.get_model import get_model
from src.training.utils.get_optimizer import get_optimizer
from src.training.utils.mixed_precision import get_mixed_precision


class EarlyStopping:
    """
    Early stopping utility to stop training when a metric stops improving.
    
    Args:
        patience: Number of epochs to wait after last improvement before stopping
        min_delta: Minimum change to qualify as an improvement
        mode: 'max' for metrics to maximize (e.g., accuracy), 'min' for metrics to minimize (e.g., loss)
        restore_best: Whether to restore the best model state when stopping
    """

    def __init__(self, patience=7, min_delta=0.0, mode='max', restore_best=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best
        self.counter = 0
        self.best_score = None
        self.best_state = None
        self.early_stop = False

    def __call__(self, score, model):
        """
        Check if training should stop based on the current score.
        
        Args:
            score: Current metric value (e.g., accuracy or loss)
            model: Model to potentially save/restore state (may be wrapped with DataParallel)
        
        Returns:
            bool: True if training should stop, False otherwise
        """
        # Get the underlying model if wrapped with DataParallel
        model_to_save = model.module if isinstance(model, nn.DataParallel) else model
        
        if self.best_score is None:
            self.best_score = score
            if self.restore_best:
                self.best_state = model_to_save.state_dict().copy()
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            if self.restore_best:
                self.best_state = model_to_save.state_dict().copy()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best and self.best_state is not None:
                    model_to_save.load_state_dict(self.best_state)
                    print(f"Early stopping: Restored best model with score {self.best_score:.4f}")

        return self.early_stop

    def _is_better(self, current, best):
        """Check if current score is better than best score."""
        if self.mode == 'max':
            return current > best + self.min_delta
        else:  # mode == 'min'
            return current < best - self.min_delta

    def get_best_score(self):
        """Get the best score seen so far."""
        return self.best_score


def evaluate_model(model, loss_function, test_loader, device, epoch, writer):
    """
    Evaluate a model on the test dataset.
    
    Args:
        model: The model to evaluate (may be wrapped with DataParallel)
        loss_function: Loss function to compute test loss
        test_loader: DataLoader for test dataset
        device: Device to run evaluation on
    
    Returns:
        tuple: (average_loss, accuracy) where accuracy is a percentage
    """
    model.eval()
    test_loss = 0.0
    correct = 0

    # Use autocast for consistency with training and to avoid DataParallel alignment issues
    use_autocast = device.type == 'cuda'
    
    with torch.no_grad():
        for test_images, test_labels in test_loader:
            test_images, test_labels = test_images.to(device), test_labels.to(device)

            # Use autocast during evaluation for consistency and to avoid alignment issues
            with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', enabled=use_autocast):
                outputs = model(test_images)
                loss = loss_function(outputs, test_labels)
            
            test_loss += loss.item()
            correct += (outputs.argmax(1) == test_labels).sum().item()

    avg_loss = test_loss / len(test_loader)
    accuracy = 100.0 * correct / len(test_loader.dataset)

    print(f"Epoch {epoch}: Test Loss {avg_loss:.3f}, Accuracy {accuracy:.3f}%")

    writer.add_scalar("Test/Loss", avg_loss, epoch)
    writer.add_scalar("Test/Accuracy", accuracy, epoch)

    wandb.log({"val_loss": avg_loss, "val_acc": accuracy})

    return avg_loss, accuracy


def initialize_training(config, train_dataset=None, test_dataset=None):
    """
    Initialize common training components.
    
    Args:
        config: Main configuration dictionary
        train_dataset: Optional pre-loaded train dataset (if provided, will skip dataset loading)
        test_dataset: Optional pre-loaded test dataset (if provided, will skip dataset loading)

    Returns:
        dict: Dictionary containing initialized components:
            - 'run': wandb run object
            - 'device': device string
            - 'experiment_number': experiment number
            - 'train_loader': training DataLoader
            - 'test_loader': test DataLoader
            - 'model': initialized model
            - 'loss_function': loss function
            - 'optimizer': optimizer
            - 'scheduler': learning rate scheduler
            - 'scaler': mixed precision scaler
            - 'writer': TensorBoard SummaryWriter
    """
    run = wandb.init(project="ACNN-project", resume="allow")

    config_wb = wandb.config

    device_str = config['experiment']['device']
    device = torch.device(device_str) if isinstance(device_str, str) else device_str
    experiment_number = config['experiment']['number']
    
    # Verify GPU is available if using CUDA
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is not available!")
        
        # Check for multiple GPUs and enable DataParallel if available
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print(f"Detected {num_gpus} GPUs. Enabling DataParallel for multi-GPU training.")
            print(f"GPUs: {[torch.cuda.get_device_name(i) for i in range(num_gpus)]}")
            # Use the first GPU as the main device for DataParallel
            device = torch.device('cuda:0')
        else:
            print(f"Using single GPU: {torch.cuda.get_device_name(0)}")

    # Override with sweep hyperparameters
    config['training']['learning_rate'] = config_wb.learning_rate
    config['training']['weight_decay'] = config_wb.weight_decay
    config['dataset']['batch_size'] = config_wb.batch_size
    config['training']['optimizer'] = config_wb.optimizer
    if hasattr(config_wb, 'dropout'):
        config['model']['drop_rate'] = config_wb.dropout
    if hasattr(config_wb, 'warmup_epochs'):
        config['training']['warmup_epochs'] = config_wb.warmup_epochs
    if hasattr(config_wb, 'T_0'):
        config['training']['T_0'] = config_wb.T_0
    if hasattr(config_wb, 'T_mult'):
        config['training']['T_mult'] = config_wb.T_mult
    if hasattr(config_wb, 'eta_min'):
        config['training']['eta_min'] = config_wb.eta_min
    if hasattr(config_wb, 'randaugment_magnitude'):
        config['augmentation']['randaugment_magnitude'] = config_wb.randaugment_magnitude

    # Use pre-loaded datasets if provided, otherwise load them
    if train_dataset is not None and test_dataset is not None:
        train_loader, test_loader = preprocessing(config, train_dataset=train_dataset, test_dataset=test_dataset)
    else:
        train_loader, test_loader = preprocessing(config)
    model = get_model(config['model']['name'], config).to(device)
    
    # Wrap model with DataParallel if multiple GPUs are available
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Model wrapped with DataParallel to use {torch.cuda.device_count()} GPUs")
    
    loss_function = get_loss_function(config["training"]["loss_function"], config)
    optimizer = get_optimizer(config, model.parameters())
    scheduler = get_lr_scheduler(config, optimizer)
    scaler = get_mixed_precision()

    log_dir = f'../experiments/experiment{experiment_number}/results'
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(
        os.path.join(log_dir, datetime.now().strftime(config["experiment"]["date_format"]))
    )

    return {
        'run': run,
        'device': device,
        'experiment_number': experiment_number,
        'train_loader': train_loader,
        'test_loader': test_loader,
        'model': model,
        'loss_function': loss_function,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'scaler': scaler,
        'writer': writer
    }


def save_checkpoint_if_best(model, accuracy, best_acc, experiment_number):
    """
    Save model checkpoint if current accuracy is better than best accuracy.
    
    Args:
        model: The model to save (may be wrapped with DataParallel)
        accuracy: Current accuracy (percentage)
        best_acc: Best accuracy seen so far (percentage)
        experiment_number: Experiment number for checkpoint directory
    
    Returns:
        float: Updated best accuracy (accuracy if it's better, otherwise best_acc)
    """
    if accuracy > best_acc:
        checkpoint_dir = f'../experiments/experiment{experiment_number}/checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Handle DataParallel: save the underlying model's state dict
        if isinstance(model, nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        torch.save(state_dict, os.path.join(
            checkpoint_dir, f'best_model_{accuracy}.pth'
        ))
        return accuracy
    return best_acc


def finalize_training(writer, run):
    """
    Finalize training by closing resources.
    
    Args:
        writer: TensorBoard SummaryWriter to close
        run: Wandb run object to finish
    """
    writer.close()
    run.finish()


def get_training_logic(name):
    """
    Get a training logic function by name.
    
    Args:
        name: Name of the training logic (e.g., "baseline")
    
    Returns:
        A training function that can be used with wandb sweeps.
        The function signature should be: train_function(config, wandb_config)
    """
    if name == "baseline":
        return baseline_training_logic
    else:
        raise ValueError(f"Unknown training logic: {name}")


def baseline_training_logic(config, train_dataset=None, test_dataset=None):
    """
    Baseline training logic: standard supervised training with mixed precision.
    
    Args:
        config: Main configuration dictionary
        train_dataset: Optional pre-loaded train dataset
        test_dataset: Optional pre-loaded test dataset
    """
    components = initialize_training(config, train_dataset=train_dataset, test_dataset=test_dataset)
    run = components['run']
    device = components['device']
    experiment_number = components['experiment_number']
    train_loader = components['train_loader']
    test_loader = components['test_loader']
    model = components['model']
    loss_function = components['loss_function']
    optimizer = components['optimizer']
    scheduler = components['scheduler']
    scaler = components['scaler']
    writer = components['writer']

    best_acc = 0

    # Initialize early stopping if enabled
    early_stopping = None
    if config.get('training', {}).get('early_stopping', {}).get('enabled', False):
        patience = config['training']['early_stopping'].get('patience', 7)
        min_delta = config['training']['early_stopping'].get('min_delta', 0.0)
        mode = config['training']['early_stopping'].get('mode', 'max')  # 'max' for accuracy, 'min' for loss
        restore_best = config['training']['early_stopping'].get('restore_best', True)
        early_stopping = EarlyStopping(patience=patience, min_delta=min_delta, mode=mode, restore_best=restore_best)
        print(f"Early stopping enabled: patience={patience}, mode={mode}, min_delta={min_delta}")

    for epoch in range(1, config['training']['epochs'] + 1):
        # Training phase
        model.train()
        train_loss = 0.0

        for batch_index, (train_images, train_labels) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch}")
        ):
            train_images = train_images.to(device)
            train_labels = train_labels.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
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

        # Update learning rate with warmup support
        current_lr = get_current_lr(epoch, optimizer, scheduler, config)
        writer.add_scalar("Train/LearningRate", current_lr, epoch)
        wandb.log({"learning_rate": current_lr})

        avg_loss, accuracy = evaluate_model(model, loss_function, test_loader, device, epoch, writer)

        # Step ReduceLROnPlateau with validation loss if needed
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_loss)

        best_acc = save_checkpoint_if_best(model, accuracy, best_acc, experiment_number)

        # Check early stopping
        if early_stopping is not None:
            # Use accuracy for early stopping (mode='max') or loss (mode='min')
            metric = accuracy if early_stopping.mode == 'max' else avg_loss
            if early_stopping(metric, model):
                print(f"Early stopping triggered at epoch {epoch}")
                print(f"Best {early_stopping.mode} score: {early_stopping.get_best_score():.4f}")
                wandb.log({"early_stopped": True, "early_stop_epoch": epoch})
                break

    finalize_training(writer, run)
