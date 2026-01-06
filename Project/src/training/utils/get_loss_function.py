import sys

import torch


def get_loss_function(name):
    name = name.lower()
    if name == 'crossentropyloss':
        print('CrossEntropyLoss Loss Function loaded!')
        return torch.nn.CrossEntropyLoss()
    elif name == 'mseloss':
        print('MSELoss Loss Function loaded!')
        return torch.nn.MSELoss()
    else:
        print('The loss function name you have entered is not supported!')
        sys.exit()
