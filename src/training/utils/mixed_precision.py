import torch


def get_mixed_precision():
    return torch.amp.GradScaler()
