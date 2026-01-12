import sys
import torch
# from sam import SAM


def get_optimizer(configs, params):
    name = configs["training"]["optimizer"].lower()
    if name == 'sgd':
        print('SGD Optimizer loaded!')
        return torch.optim.SGD(params, lr=configs["training"]["learning_rate"], momentum=configs["training"]["momentum"], weight_decay=float(configs["training"]["weight_decay"]))
    elif name == 'adam':
        print('Adam Optimizer loaded!')
        return torch.optim.Adam(params, lr=configs["training"]["learning_rate"], weight_decay=float(configs["training"]["weight_decay"]))
    elif name == 'adamw':
        print('AdamW Optimizer loaded!')
        return torch.optim.AdamW(params, lr=configs["training"]["learning_rate"], weight_decay=float(configs["training"]["weight_decay"]))
    elif name == 'muon':
        print('Muon Optimizer loaded!')
        return torch.optim.Muon(params, lr=configs["training"]["learning_rate"], weight_decay=float(configs["training"]["weight_decay"]))
    # elif name == 'sam':
    #     print('Optimizer loaded!')
    #     # base_optim = torch.optim.SGD(params, lr=configs["training"]["learning_rate"], momentum=configs["training"]["momentum"], weight_decay=float(configs["training"]["weight_decay"]))
    #     base_optim = torch.optim.SGD
    #     return SAM(params, base_optim, lr=configs["training"]["learning_rate"], momentum=float(configs["training"]["momentum"]))
    else:
        print('The optimizer name you have entered is not supported!')
        sys.exit()
