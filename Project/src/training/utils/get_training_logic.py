import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import wandb
import numpy as np

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
            model: Model to potentially save/restore state
        
        Returns:
            bool: True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            if self.restore_best:
                self.best_state = model.state_dict().copy()
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            if self.restore_best:
                self.best_state = model.state_dict().copy()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best and self.best_state is not None:
                    model.load_state_dict(self.best_state)
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
        model: The model to evaluate
        loss_function: Loss function to compute test loss
        test_loader: DataLoader for test dataset
        device: Device to run evaluation on
    
    Returns:
        tuple: (average_loss, accuracy) where accuracy is a percentage
    """
    model.eval()
    test_loss = 0.0
    correct = 0

    with torch.no_grad():
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
    
    return avg_loss, accuracy


def initialize_training(config, config_wb):
    """
    Initialize common training components.
    
    Args:
        config: Main configuration dictionary
        config_wb: Wandb config object with sweep hyperparameters
    
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
    # Reuse existing wandb run if already initialized (e.g., by wandb.agent)
    run = wandb.init(project="ACNN-project", resume="allow")
    
    device = config['experiment']['device']
    experiment_number = config['experiment']['number']

    # Override with sweep hyperparameters
    config['training']['learning_rate'] = config_wb.learning_rate
    config['training']['weight_decay'] = config_wb.weight_decay
    config['dataset']['batch_size'] = config_wb.batch_size
    config['training']['optimizer'] = config_wb.optimizer
    if hasattr(config_wb, 'dropout'):
        config['model']['drop_rate'] = config_wb.dropout
    if hasattr(config_wb, 'warmup_epochs'):
        config['training']['warmup_epochs'] = config_wb.warmup_epochs

    train_loader, test_loader = preprocessing(config)
    model = get_model(config['model']['name'], config).to(device)
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
        model: The model to save
        accuracy: Current accuracy (percentage)
        best_acc: Best accuracy seen so far (percentage)
        experiment_number: Experiment number for checkpoint directory
    
    Returns:
        float: Updated best accuracy (accuracy if it's better, otherwise best_acc)
    """
    if accuracy > best_acc:
        checkpoint_dir = f'../experiments/experiment{experiment_number}/checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(
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


def generate_pseudo_labels(model, unlabeled_loader, device, confidence_threshold=0.0):
    """
    Generate pseudo-labels for unlabeled data using a teacher model.
    
    Args:
        model: Teacher model to generate pseudo-labels
        unlabeled_loader: DataLoader for unlabeled data
        device: Device to run inference on
        confidence_threshold: Minimum confidence to keep a pseudo-label (0.0 = keep all)
    
    Returns:
        tuple: (pseudo_labeled_data, pseudo_labeled_targets) as lists
    """
    model.eval()
    pseudo_data = []
    pseudo_targets = []
    
    with torch.no_grad():
        for images, _ in tqdm(unlabeled_loader, desc="Generating pseudo-labels"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confidences, predicted = torch.max(probs, dim=1)
            
            # Filter by confidence threshold
            mask = confidences >= confidence_threshold
            if mask.any():
                pseudo_data.append(images[mask].cpu())
                pseudo_targets.append(predicted[mask].cpu())
    
    if pseudo_data:
        pseudo_data = torch.cat(pseudo_data, dim=0)
        pseudo_targets = torch.cat(pseudo_targets, dim=0)
    else:
        pseudo_data = torch.empty(0)
        pseudo_targets = torch.empty(0, dtype=torch.long)
    
    return pseudo_data, pseudo_targets


class PseudoLabeledDataset(torch.utils.data.Dataset):
    """Dataset wrapper for pseudo-labeled data."""
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def train_epoch_with_noise(model, labeled_loader, unlabeled_loader, loss_function, 
                          optimizer, scheduler, scaler, device, writer, epoch, config,
                          lambda_u=1.0, drop_rate=0.0):
    """
    Train one epoch with noise injection (Noisy Student approach).
    
    Args:
        model: Model to train
        labeled_loader: DataLoader for labeled data
        unlabeled_loader: DataLoader for unlabeled/pseudo-labeled data (can be None)
        loss_function: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        scaler: Mixed precision scaler
        device: Device to train on
        writer: TensorBoard writer
        epoch: Current epoch number
        lambda_u: Weight for unlabeled loss
        drop_rate: Dropout rate for additional noise
    
    Returns:
        float: Average training loss
    """
    model.train()
    
    # Enable dropout for additional noise
    if drop_rate > 0:
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = drop_rate
    
    train_loss = 0.0
    n_iter = 0
    
    # Combine labeled and unlabeled loaders
    if unlabeled_loader is not None:
        # Use zip to iterate over both loaders
        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)
        
        max_iter = max(len(labeled_loader), len(unlabeled_loader))
        
        for batch_idx in range(max_iter):
            # Get labeled batch
            try:
                labeled_images, labeled_targets = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                labeled_images, labeled_targets = next(labeled_iter)
            
            # Get unlabeled batch
            try:
                unlabeled_images, unlabeled_targets = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                unlabeled_images, unlabeled_targets = next(unlabeled_iter)
            
            labeled_images = labeled_images.to(device)
            labeled_targets = labeled_targets.to(device)
            unlabeled_images = unlabeled_images.to(device)
            unlabeled_targets = unlabeled_targets.to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda'):
                # Labeled loss
                labeled_outputs = model(labeled_images)
                labeled_loss = loss_function(labeled_outputs, labeled_targets)
                
                # Unlabeled loss (pseudo-labels)
                unlabeled_outputs = model(unlabeled_images)
                unlabeled_loss = loss_function(unlabeled_outputs, unlabeled_targets)
                
                # Combined loss
                loss = labeled_loss + lambda_u * unlabeled_loss
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            n_iter = (epoch - 1) * max_iter + batch_idx + 1
            writer.add_scalar("Train/Loss", loss.item(), n_iter)
            writer.add_scalar("Train/LabeledLoss", labeled_loss.item(), n_iter)
            writer.add_scalar("Train/UnlabeledLoss", unlabeled_loss.item(), n_iter)
            
            wandb.log({
                "train_loss": loss.item(),
                "train_labeled_loss": labeled_loss.item(),
                "train_unlabeled_loss": unlabeled_loss.item()
            })
            
            train_loss += loss.item()
    else:
        # Only labeled data
        for batch_idx, (images, targets) in enumerate(labeled_loader):
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(images)
                loss = loss_function(outputs, targets)
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            n_iter = (epoch - 1) * len(labeled_loader) + batch_idx + 1
            writer.add_scalar("Train/Loss", loss.item(), n_iter)
            wandb.log({"train_loss": loss.item()})
            
            train_loss += loss.item()
    
    # Update learning rate with warmup support
    current_lr = get_current_lr(epoch, optimizer, scheduler, config)
    writer.add_scalar("Train/LearningRate", current_lr, epoch)
    wandb.log({"learning_rate": current_lr})
    
    # Reset dropout
    if drop_rate > 0:
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
    
    return train_loss / max(len(labeled_loader), len(unlabeled_loader) if unlabeled_loader else len(labeled_loader))


def get_training_logic(name):
    """
    Get a training logic function by name.
    
    Args:
        name: Name of the training logic (e.g., "baseline", "noisy_student")
    
    Returns:
        A training function that can be used with wandb sweeps.
        The function signature should be: train_function(config, wandb_config)
    """
    if name == "baseline":
        return baseline_training_logic
    elif name == "noisy_student":
        return noisy_student_training_logic
    else:
        raise ValueError(f"Unknown training logic: {name}")


def baseline_training_logic(config, config_wb):
    """
    Baseline training logic: standard supervised training with mixed precision.
    
    Args:
        config: Main configuration dictionary
        config_wb: Wandb config object with sweep hyperparameters
    """
    components = initialize_training(config, config_wb)
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


def noisy_student_training_logic(config, config_wb):
    """
    Noisy Student training logic: semi-supervised learning with teacher-student approach.
    
    Steps:
    1. Train teacher model on labeled data (subset of training data)
    2. Generate pseudo-labels for unlabeled data using teacher
    3. Train student model (possibly larger) on labeled + pseudo-labeled data with noise
    4. Optionally iterate (student becomes teacher)
    
    Args:
        config: Main configuration dictionary
        config_wb: Wandb config object with sweep hyperparameters
    """
    # Use common initialization logic
    components = initialize_training(config, config_wb)
    run = components['run']
    device = components['device']
    experiment_number = components['experiment_number']
    test_loader = components['test_loader']
    loss_function = components['loss_function']
    scaler = components['scaler']
    writer = components['writer']
    
    # Get the train dataset from the train_loader (we'll split it ourselves)
    train_loader = components['train_loader']
    train_dataset = train_loader.dataset
    
    # Get Noisy Student specific parameters
    labeled_ratio = config.get('noisy_student', {}).get('labeled_ratio', 0.1)  # 10% labeled by default
    num_iterations = config.get('noisy_student', {}).get('num_iterations', 1)  # Number of teacher-student iterations
    lambda_u = config.get('noisy_student', {}).get('lambda_u', 1.0)  # Weight for unlabeled loss
    drop_rate = config.get('noisy_student', {}).get('drop_rate', 0.1)  # Additional dropout noise
    confidence_threshold = config.get('noisy_student', {}).get('confidence_threshold', 0.0)
    student_larger = config.get('noisy_student', {}).get('student_larger', False)  # Use larger student model
    
    # Split into labeled and unlabeled subsets
    dataset_size = len(train_dataset)
    labeled_size = int(dataset_size * labeled_ratio)
    indices = np.random.permutation(dataset_size)
    labeled_indices = indices[:labeled_size]
    unlabeled_indices = indices[labeled_size:]
    
    labeled_dataset = Subset(train_dataset, labeled_indices)
    unlabeled_dataset = Subset(train_dataset, unlabeled_indices)
    
    labeled_loader = DataLoader(
        labeled_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=True,
        pin_memory=True
    )
    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=True,
        pin_memory=True
    )
    
    # Teacher-Student iterations
    teacher_model = None
    best_acc = 0
    
    for iteration in range(num_iterations):
        print(f"\n{'='*50}")
        print(f"Noisy Student Iteration {iteration + 1}/{num_iterations}")
        print(f"{'='*50}")
        
        # Step 1: Train teacher model (or use previous student as teacher)
        if teacher_model is None:
            # First iteration: train teacher on labeled data only
            print("Training teacher model on labeled data...")
            teacher_model = get_model(config['model']['name'], config).to(device)
            teacher_optimizer = get_optimizer(config, teacher_model.parameters())
            teacher_scheduler = get_lr_scheduler(config, teacher_optimizer)
            
            teacher_epochs = config.get('noisy_student', {}).get('teacher_epochs', config['training']['epochs'])
            
            # Initialize early stopping for teacher if enabled
            teacher_early_stopping = None
            if config.get('training', {}).get('early_stopping', {}).get('enabled', False):
                patience = config['training']['early_stopping'].get('patience', 7)
                min_delta = config['training']['early_stopping'].get('min_delta', 0.0)
                mode = config['training']['early_stopping'].get('mode', 'max')
                restore_best = config['training']['early_stopping'].get('restore_best', True)
                teacher_early_stopping = EarlyStopping(patience=patience, min_delta=min_delta, mode=mode, restore_best=restore_best)
            
            for epoch in range(1, teacher_epochs + 1):
                train_loss = train_epoch_with_noise(
                    teacher_model, labeled_loader, None, loss_function,
                    teacher_optimizer, teacher_scheduler, scaler, device,
                    writer, epoch, config, lambda_u=0.0, drop_rate=0.0
                )
                
                avg_loss, accuracy = evaluate_model(teacher_model, loss_function, test_loader, device, epoch, writer)
                
                # Step ReduceLROnPlateau with validation loss if needed
                if isinstance(teacher_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    teacher_scheduler.step(avg_loss)
                
                best_acc = save_checkpoint_if_best(teacher_model, accuracy, best_acc, experiment_number)
                
                # Check early stopping for teacher
                if teacher_early_stopping is not None:
                    metric = accuracy if teacher_early_stopping.mode == 'max' else avg_loss
                    if teacher_early_stopping(metric, teacher_model):
                        print(f"Early stopping triggered for teacher at epoch {epoch}")
                        break
        
        # Step 2: Generate pseudo-labels for unlabeled data
        print("Generating pseudo-labels for unlabeled data...")
        pseudo_data, pseudo_targets = generate_pseudo_labels(
            teacher_model, unlabeled_loader, device, confidence_threshold
        )
        
        if len(pseudo_data) == 0:
            print("Warning: No pseudo-labels generated. Skipping student training.")
            break
        
        # Create pseudo-labeled dataset
        pseudo_dataset = PseudoLabeledDataset(pseudo_data, pseudo_targets)
        pseudo_loader = DataLoader(
            pseudo_dataset,
            batch_size=config['dataset']['batch_size'],
            shuffle=True,
            pin_memory=True
        )
        
        print(f"Generated {len(pseudo_data)} pseudo-labels")
        
        # Step 3: Train student model on labeled + pseudo-labeled data with noise
        print("Training student model with noise...")
        
        # Optionally use a larger model for student
        if student_larger and iteration == 0:
            # Try to get a larger variant (e.g., EfficientNet B1 instead of B0)
            student_model_name = config.get('noisy_student', {}).get('student_model_name', config['model']['name'])
            student_config = config.copy()
            student_config['model']['name'] = student_model_name
            student_model = get_model(student_model_name, student_config).to(device)
        else:
            student_model = get_model(config['model']['name'], config).to(device)
        
        student_optimizer = get_optimizer(config, student_model.parameters())
        student_scheduler = get_lr_scheduler(config, student_optimizer)
        
        student_epochs = config.get('noisy_student', {}).get('student_epochs', config['training']['epochs'])
        
        # Initialize early stopping for student if enabled
        student_early_stopping = None
        if config.get('training', {}).get('early_stopping', {}).get('enabled', False):
            patience = config['training']['early_stopping'].get('patience', 7)
            min_delta = config['training']['early_stopping'].get('min_delta', 0.0)
            mode = config['training']['early_stopping'].get('mode', 'max')
            restore_best = config['training']['early_stopping'].get('restore_best', True)
            student_early_stopping = EarlyStopping(patience=patience, min_delta=min_delta, mode=mode, restore_best=restore_best)
        
        for epoch in range(1, student_epochs + 1):
            train_loss = train_epoch_with_noise(
                student_model, labeled_loader, pseudo_loader, loss_function,
                student_optimizer, student_scheduler, scaler, device,
                writer, epoch, config, lambda_u=lambda_u, drop_rate=drop_rate
            )
            
            avg_loss, accuracy = evaluate_model(student_model, loss_function, test_loader, device, epoch, writer)
            
            # Step ReduceLROnPlateau with validation loss if needed
            if isinstance(student_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                student_scheduler.step(avg_loss)
            
            best_acc = save_checkpoint_if_best(student_model, accuracy, best_acc, experiment_number)
            
            # Check early stopping for student
            if student_early_stopping is not None:
                metric = accuracy if student_early_stopping.mode == 'max' else avg_loss
                if student_early_stopping(metric, student_model):
                    print(f"Early stopping triggered for student at epoch {epoch}")
                    break
        
        # Step 4: Student becomes teacher for next iteration
        teacher_model = student_model
        print(f"Iteration {iteration + 1} complete. Best accuracy so far: {best_acc:.2f}%")
    
    finalize_training(writer, run)
