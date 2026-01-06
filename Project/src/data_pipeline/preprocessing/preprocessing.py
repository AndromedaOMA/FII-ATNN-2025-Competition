import os
import sys
from typing import Optional, Callable

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.v2 as v2
from torch.utils.data import Dataset, DataLoader


class CIFAR100_noisy_fine(Dataset):
    """
    See https://github.com/UCSC-REAL/cifar-10-100n, https://www.noisylabels.com/ and `Learning with Noisy Labels
    Revisited: A Study Using Real-World Human Annotations`.
    """

    def __init__(
        self, root: str, train: bool, transform: Optional[Callable], download: bool
    ):
        cifar100 = datasets.CIFAR100(
            root=root, train=train, transform=transform, download=download
        )
        data, targets = tuple(zip(*cifar100))

        if train:
            noisy_label_file = os.path.join(root, "CIFAR-100-noisy.npz")
            if not os.path.isfile(noisy_label_file):
                raise FileNotFoundError(
                    f"{type(self).__name__} need {noisy_label_file} to be used!"
                )

            noise_file = np.load(noisy_label_file)
            if not np.array_equal(noise_file["clean_label"], targets):
                raise RuntimeError("Clean labels do not match!")
            targets = noise_file["noisy_label"]

        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i: int):
        return self.data[i], self.targets[i]


class SimpleCachedDataset(Dataset):
    def __init__(self, dataset):
        # Runtime transforms are not implemented in this simple cached dataset.
        self.data = tuple([x for x in dataset])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def create_cutmix_mixup_collate_fn(num_classes, cutmix_alpha=0.5, mixup_alpha=0.5):
    """
    Create a collate function that applies CutMix and MixUp at batch level.
    
    Args:
        cutmix_alpha: Alpha parameter for CutMix (0.0 = no CutMix)
        mixup_alpha: Alpha parameter for MixUp (0.0 = no MixUp)
    
    Returns:
        Collate function for DataLoader
    """
    # Create CutMix and MixUp transforms once (they are stateless)
    cutmix = v2.CutMix(alpha=cutmix_alpha, num_classes=num_classes) if cutmix_alpha > 0 else None
    mixup = v2.MixUp(alpha=mixup_alpha, num_classes=num_classes) if mixup_alpha > 0 else None
    
    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        
        # Apply CutMix if enabled
        if cutmix is not None:
            images, labels = cutmix(images, labels)
        
        # Apply MixUp if enabled
        if mixup is not None:
            images, labels = mixup(images, labels)
        
        return images, labels
    
    return collate_fn


def preprocessing(config):
    if config['dataset']['name'] == 'MNIST':
        train_transformer = transforms.Compose([
            transforms.RandomAffine(degrees=2, translate=[0.1, 0.1]),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ])
        test_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ])
        train_dataset = datasets.MNIST(root=f'{config["dataset"]["data_dir"]}/train', train=True, transform=train_transformer, download=True)
        test_dataset = datasets.MNIST(root=f'{config["dataset"]["data_dir"]}/test', train=False, transform=test_transformer, download=True)
        train_loader = DataLoader(train_dataset, batch_size=config['dataset']['batch_size'], shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=config['dataset']['batch_size'], shuffle=False, pin_memory=True)
        return train_loader, test_loader

    elif config['dataset']['name'] == 'CIFAR10':
        mean = list(map(float, config["dataset"]["mean"]))
        std = list(map(float, config["dataset"]["std"]))

        train_transformer = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_dataset = datasets.CIFAR10(root=f'{config["dataset"]["data_dir"]}/train', train=True, transform=train_transformer, download=True)
        test_dataset = datasets.CIFAR10(root=f'{config["dataset"]["data_dir"]}/test', train=False, transform=test_transformer, download=True)
        train_loader = DataLoader(train_dataset, batch_size=config['dataset']['batch_size'], shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=config['dataset']['batch_size'], shuffle=False, pin_memory=True)
        return train_loader, test_loader

    elif config['dataset']['name'] == 'CIFAR100':
        mean = list(map(float, config["dataset"]["mean"]))
        std = list(map(float, config["dataset"]["std"]))

        train_transformer = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_dataset = datasets.CIFAR100(root=f'{config["dataset"]["data_dir"]}/train', train=True, transform=train_transformer, download=True)
        test_dataset = datasets.CIFAR100(root=f'{config["dataset"]["data_dir"]}/test', train=False, transform=test_transformer, download=True)
        train_loader = DataLoader(train_dataset, batch_size=config['dataset']['batch_size'], shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=config['dataset']['batch_size'], shuffle=False, pin_memory=True)
        return train_loader, test_loader

    elif config['dataset']['name'] == 'CIFAR100-N':
        num_classes = 100

        mean = list(map(float, config["dataset"]["mean"]))
        std = list(map(float, config["dataset"]["std"]))
        
        # Get augmentation parameters from config
        cutmix_alpha = config.get('augmentation', {}).get('cutmix_alpha', 0.5)
        mixup_alpha = config.get('augmentation', {}).get('mixup_alpha', 0.5)
        randaugment_num_ops = config.get('augmentation', {}).get('randaugment_num_ops', 2)
        randaugment_magnitude = config.get('augmentation', {}).get('randaugment_magnitude', 9)
        
        # Training transforms with RandAugment (per-sample augmentation)
        train_transformer = v2.Compose([
            v2.ToImage(),
            v2.Resize((224, 224), antialias=True),
            v2.RandAugment(num_ops=randaugment_num_ops, magnitude=randaugment_magnitude),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean, std, inplace=True)
        ])
        test_transformer = v2.Compose([
            v2.ToImage(),
            v2.Resize((224, 224), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean, std, inplace=True)
        ])
        
        data_root = config["dataset"]["data_dir"]
        
        train_dataset = CIFAR100_noisy_fine(
            root=data_root, 
            train=True, 
            transform=train_transformer, 
            download=False
        )
        test_dataset = CIFAR100_noisy_fine(
            root=data_root, 
            train=False, 
            transform=test_transformer, 
            download=False
        )
        
        train_dataset = SimpleCachedDataset(train_dataset)
        test_dataset = SimpleCachedDataset(test_dataset)
        
        # Create collate function for CutMix/MixUp (batch-level augmentation)
        train_collate_fn = create_cutmix_mixup_collate_fn(num_classes, cutmix_alpha, mixup_alpha)
        
        pin_memory = config.get('dataset', {}).get('pin_memory', True)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config['dataset']['batch_size'], 
            shuffle=True, 
            pin_memory=pin_memory,
            collate_fn=train_collate_fn
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config['dataset']['batch_size'], 
            shuffle=False, 
            pin_memory=pin_memory
        )
        return train_loader, test_loader

    elif config['dataset']['name'] == 'OxfordIIITPet':
        mean = list(map(float, config["dataset"]["mean"]))
        std = list(map(float, config["dataset"]["std"]))

        train_transformer = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_dataset = datasets.OxfordIIITPet(root=f'{config["dataset"]["data_dir"]}/train', train=True, transform=train_transformer, download=True)
        test_dataset = datasets.OxfordIIITPet(root=f'{config["dataset"]["data_dir"]}/test', train=False, transform=test_transformer, download=True)
        train_loader = DataLoader(train_dataset, batch_size=config['dataset']['batch_size'], shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=config['dataset']['batch_size'], shuffle=False, pin_memory=True)
        return train_loader, test_loader
    else:
        print('The dataset name you have entered is not supported!')
        sys.exit()

