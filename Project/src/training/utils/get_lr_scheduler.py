import sys
import torch


def get_current_lr(epoch, optimizer, scheduler, config):
    """
    Get the current learning rate with warmup support.
    
    During warmup: linearly increases from initial_lr to target learning_rate.
    After warmup: uses the scheduler to get the current learning rate.
    
    Args:
        epoch: Current epoch number (1-indexed)
        optimizer: Optimizer object
        scheduler: Learning rate scheduler
        config: Configuration dictionary
    
    Returns:
        float: Current learning rate
    """
    warmup_epochs = config.get('training', {}).get('warmup_epochs', 0)
    target_lr = config['training']['learning_rate']
    initial_lr = config.get('training', {}).get('warmup_initial_lr', 0.0)
    
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        # Warmup phase: linear increase from initial_lr to target_lr
        warmup_lr = initial_lr + (target_lr - initial_lr) * (epoch / warmup_epochs)
        
        # Set the learning rate manually (don't step scheduler during warmup)
        for param_group in optimizer.param_groups:
            param_group['lr'] = warmup_lr
        
        return warmup_lr
    else:
        # After warmup: use the scheduler
        # For ReduceLROnPlateau, we don't step here (it's stepped with a metric)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            # ReduceLROnPlateau is stepped separately with a metric
            current_lr = optimizer.param_groups[0]['lr']
        else:
            # Step the scheduler and get the new LR
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
        
        return current_lr


def get_lr_scheduler(configs, optimizer):
    name = configs["training"]["scheduler"].lower()
    if name == 'steplr':
        print('StepLR Learning Rate Scheduler loaded!')
        return torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=configs["training"]["step_size"],
                                               gamma=configs["training"]["gamma"])
    elif name == 'reducelronplateau':
        print('ReduceLROnPlateau Learning Rate Scheduler loaded!')
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                          mode=configs["training"]["mode"],
                                                          factor=configs["training"]["factor"],
                                                          patience=configs["training"]["patience"],
                                                          threshold=configs["training"]["threshold"])
    elif name == 'cosineannealingwarmrestarts' or name == 'cosinewarmrestarts':
        print('CosineAnnealingWarmRestarts Learning Rate Scheduler loaded!')
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                    T_0=configs["training"]["T_0"],
                                                                    T_mult=configs["training"].get("T_mult", 1),
                                                                    eta_min=configs["training"].get("eta_min", 0))
    else:
        print('The Learning Rate Scheduler name you have entered is not supported!')
        sys.exit()
