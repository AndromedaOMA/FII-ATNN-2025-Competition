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
    else:
        print('The Learning Rate Scheduler name you have entered is not supported!')
        sys.exit()
