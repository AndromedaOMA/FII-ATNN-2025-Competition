import sys

import torch


def get_loss_function(name, config=None):
    """
    Get a loss function by name.
    
    Args:
        name: Name of the loss function (e.g., "CrossEntropyLoss")
        config: Optional configuration dictionary for label smoothing
    
    Returns:
        Loss function instance
    """
    name = name.lower()
    label_smoothing = 0.0
    if config is not None:
        label_smoothing = config.get('training', {}).get('label_smoothing', 0.0)
    
    if name == 'crossentropyloss':
        if label_smoothing > 0:
            print(f'CrossEntropyLoss Loss Function loaded with label_smoothing={label_smoothing}!')
        else:
            print('CrossEntropyLoss Loss Function loaded!')
        return torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    elif name == 'mseloss':
        print('MSELoss Loss Function loaded!')
        return torch.nn.MSELoss()
    else:
        print('The loss function name you have entered is not supported!')
        sys.exit()
