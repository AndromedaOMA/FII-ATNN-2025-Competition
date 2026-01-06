import sys
import torch


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
