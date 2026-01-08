import os
from pathlib import Path
import sys
from typing import Optional, Callable, Dict, Any
import pickle

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.v2 as v2
from torch.utils.data import Dataset, DataLoader


class CIFAR100_noisy_fine(Dataset):
    """
    Lazy loader for CIFAR-100 with noisy labels.
    See https://github.com/UCSC-REAL/cifar-10-100n, https://www.noisylabels.com/ and `Learning with Noisy Labels
    Revisited: A Study Using Real-World Human Annotations`.
    
    This class loads images lazily, only accessing the underlying dataset when items are requested.
    Noisy labels are loaded upfront (they're just integers, very small memory footprint).
    """

    def __init__(
        self, root: str, train: bool, transform: Optional[Callable], download: bool
    ):
        self.root = root
        self.train = train
        self.transform = transform
        self.download = download
        
        # Lazy initialization - will be created on first access
        self._cifar100 = None
        
        # Load noisy labels upfront (small memory footprint, just integers)
        if self.train:
            noisy_label_file = os.path.join(root, "CIFAR-100-noisy.npz")
            if not os.path.isfile(noisy_label_file):
                raise FileNotFoundError(
                    f"{type(self).__name__} need {noisy_label_file} to be used!"
                )

            noise_file = np.load(noisy_label_file)
            self._noisy_labels = noise_file["noisy_label"]
            self._clean_labels_ref = noise_file["clean_label"]  # For validation
        else:
            self._noisy_labels = None
            self._clean_labels_ref = None

    def _get_cifar100(self):
        """Lazy initialization of CIFAR100 dataset."""
        if self._cifar100 is None:
            self._cifar100 = datasets.CIFAR100(
                root=self.root, train=self.train, transform=self.transform, download=self.download
            )
            
            # Validate labels on first access (only for train set)
            if self.train and self._clean_labels_ref is not None:
                # Get clean labels from the dataset (lazy, but we need to validate)
                # We'll validate by checking length and a sample of indices
                dataset_len = len(self._cifar100)
                if len(self._clean_labels_ref) != dataset_len:
                    raise RuntimeError(f"Clean labels length mismatch: {len(self._clean_labels_ref)} != {dataset_len}")
                
                # Validate a sample of labels to ensure they match
                sample_indices = [0, len(self._clean_labels_ref) // 2, len(self._clean_labels_ref) - 1]
                for idx in sample_indices:
                    _, clean_label = self._cifar100[idx]
                    if clean_label != self._clean_labels_ref[idx]:
                        raise RuntimeError(f"Clean labels do not match at index {idx}!")
        
        return self._cifar100

    def __len__(self):
        """Get dataset length lazily."""
        return len(self._get_cifar100())

    def __getitem__(self, i: int):
        """Get item lazily - only loads image data when accessed."""
        cifar100 = self._get_cifar100()
        image, clean_label = cifar100[i]
        
        if self.train and self._noisy_labels is not None:
            return image, self._noisy_labels[i]
        
        return image, clean_label


class HybridCachedDataset(Dataset):
    """
    Hybrid RAM/SSD caching dataset.
    Caches dataset elements in RAM until a memory limit is reached,
    then caches remaining elements to SSD.
    
    The RAM limit is shared across all instances (train + test together).
    
    Args:
        dataset: Original dataset to cache
        config: Configuration dictionary with caching parameters:
            - cache.ram_limit_gb: RAM limit in GB (default: 16) - shared across train+test
            - cache.ssd_path: Path to SSD cache directory (required)
            - cache.prefix: Optional prefix for cache files (useful to distinguish train/test)
    """
    
    # Class-level variables to track shared RAM usage across all instances
    _shared_ram_usage = 0
    _shared_ram_limit = None
    _lock = None
    
    def __init__(self, dataset: Dataset, config: Dict[str, Any], cache_prefix: str = ""):
        self.dataset = dataset
        self.dataset_len = len(dataset)
        self.cache_prefix = cache_prefix
        
        # Get cache configuration
        cache_config = config.get('cache', {})
        ram_limit_gb = cache_config.get('ram_limit_gb', 16)
        ram_limit_bytes = ram_limit_gb * 1024 * 1024 * 1024  # Convert GB to bytes
        
        # Initialize shared RAM limit on first instance
        if HybridCachedDataset._shared_ram_limit is None:
            HybridCachedDataset._shared_ram_limit = ram_limit_bytes
            HybridCachedDataset._shared_ram_usage = 0
            # Import threading for lock (lazy import to avoid circular dependencies)
            import threading
            HybridCachedDataset._lock = threading.Lock()
        elif HybridCachedDataset._shared_ram_limit != ram_limit_bytes:
            # Warn if different limits are specified (shouldn't happen, but handle gracefully)
            print(f"Warning: RAM limit mismatch. Using {HybridCachedDataset._shared_ram_limit / (1024**3):.2f} GB (first instance)")
        
        ssd_path_str = cache_config.get('ssd_path', '')
        
        if not ssd_path_str:
            raise ValueError("cache.ssd_path must be specified in config for HybridCachedDataset")
        
        # Convert to Path if it's a string
        if isinstance(ssd_path_str, str):
            base_ssd_path = Path(ssd_path_str)
        else:
            base_ssd_path = ssd_path_str
        
        # Append train_cache or test_cache to create separate directories
        if cache_prefix == "train":
            self.ssd_path = base_ssd_path / "train_cache"
        elif cache_prefix == "test":
            self.ssd_path = base_ssd_path / "test_cache"
        else:
            # Fallback: use prefix as directory name if provided
            if cache_prefix:
                self.ssd_path = base_ssd_path / f"{cache_prefix}_cache"
            else:
                self.ssd_path = base_ssd_path
        
        # Create SSD cache directory if it doesn't exist
        self.ssd_path.mkdir(parents=True, exist_ok=True)
        
        # Track which indices are in RAM vs SSD (instance-specific)
        self.ram_cache: Dict[int, tuple] = {}
        self.ram_memory_usage = 0  # Instance-specific usage
        self.ssd_indices = set()
        
        # Cache all elements
        if HybridCachedDataset._lock:
            with HybridCachedDataset._lock:
                shared_limit_gb = HybridCachedDataset._shared_ram_limit / (1024**3)
                print(f"[{self.cache_prefix}] Caching dataset to RAM/SSD (Shared RAM limit: {shared_limit_gb:.2f} GB)...")
        else:
            shared_limit_gb = ram_limit_bytes / (1024**3)
            print(f"[{self.cache_prefix}] Caching dataset to RAM/SSD (Shared RAM limit: {shared_limit_gb:.2f} GB)...")
        
        self._cache_dataset()
        
        ram_count = len(self.ram_cache)
        ssd_count = len(self.ssd_indices)
        if HybridCachedDataset._lock:
            with HybridCachedDataset._lock:
                shared_usage_gb = HybridCachedDataset._shared_ram_usage / (1024**3)
        else:
            shared_usage_gb = 0
        print(f"[{self.cache_prefix}] Caching complete: {ram_count} items in RAM ({self.ram_memory_usage / (1024**3):.2f} GB), "
              f"{ssd_count} items on SSD (Total shared RAM: {shared_usage_gb:.2f} GB)")
    
    def _get_item_size_bytes(self, item: tuple) -> int:
        """Estimate memory size of an item in bytes."""
        size = 0
        for element in item:
            if isinstance(element, torch.Tensor):
                size += element.element_size() * element.nelement()
            elif isinstance(element, (int, float)):
                size += 8  # Approximate
            elif isinstance(element, np.ndarray):
                size += element.nbytes
        return size
    
    def _get_ssd_file_path(self, index: int) -> Path:
        """Get the file path for a cached item on SSD."""
        prefix_str = f"{self.cache_prefix}_" if self.cache_prefix else ""
        return self.ssd_path / f"{prefix_str}item_{index:08d}.pkl"
    
    def _save_to_ssd(self, index: int, item: tuple):
        """Save an item to SSD cache."""
        file_path = self._get_ssd_file_path(index)
        
        # Check if file already exists to avoid redundant writes
        if file_path.exists():
            self.ssd_indices.add(index)
            return
        
        # Save to SSD
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(item, f)
            self.ssd_indices.add(index)
        except Exception as e:
            print(f"Warning: Failed to save item {index} to SSD: {e}")
            raise
    
    def _load_from_ssd(self, index: int) -> tuple:
        """Load an item from SSD cache."""
        file_path = self._get_ssd_file_path(index)
        if not file_path.exists():
            raise FileNotFoundError(f"Cached item {index} not found on SSD: {file_path}")
        
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    def _cache_dataset(self):
        """Iterate through dataset and cache items to RAM or SSD."""
        # First, check which items are already cached on SSD
        print(f"  [{self.cache_prefix}] Checking for existing SSD cache files...")
        for i in range(self.dataset_len):
            file_path = self._get_ssd_file_path(i)
            if file_path.exists():
                self.ssd_indices.add(i)
        
        existing_ssd_count = len(self.ssd_indices)
        if existing_ssd_count > 0:
            print(f"  [{self.cache_prefix}] Found {existing_ssd_count} items already cached on SSD")
        
        # Cache remaining items
        for i in range(self.dataset_len):
            # Skip if already on SSD
            if i in self.ssd_indices:
                continue
            
            # Load item from original dataset
            item = self.dataset[i]
            item_size = self._get_item_size_bytes(item)
            
            # Check if we can fit in shared RAM (across all instances)
            with HybridCachedDataset._lock:
                shared_usage = HybridCachedDataset._shared_ram_usage
                if shared_usage + item_size <= HybridCachedDataset._shared_ram_limit:
                    # Cache in RAM and update shared usage
                    self.ram_cache[i] = item
                    self.ram_memory_usage += item_size
                    HybridCachedDataset._shared_ram_usage += item_size
                else:
                    # Not enough shared RAM, will cache to SSD below
                    pass
            
            # If not in RAM cache, save to SSD
            if i not in self.ram_cache:
                self._save_to_ssd(i, item)
            
            # Progress indicator
            if (i + 1) % 1000 == 0:
                cached_count = len(self.ram_cache) + len(self.ssd_indices)
                with HybridCachedDataset._lock:
                    shared_usage_gb = HybridCachedDataset._shared_ram_usage / (1024**3)
                    shared_limit_gb = HybridCachedDataset._shared_ram_limit / (1024**3)
                print(f"  [{self.cache_prefix}] Cached {cached_count}/{self.dataset_len} items "
                      f"(Shared RAM: {shared_usage_gb:.2f}/{shared_limit_gb:.2f} GB)...", end='\r')
        
        print()  # New line after progress
    
    def __len__(self):
        return self.dataset_len
    
    def __getitem__(self, i: int):
        """Get item from RAM cache or SSD cache."""
        if i in self.ram_cache:
            return self.ram_cache[i]
        elif i in self.ssd_indices:
            return self._load_from_ssd(i)
        else:
            # Fallback: load from original dataset (shouldn't happen if caching worked)
            print(f"Warning: Item {i} not found in cache, loading from original dataset")
            return self.dataset[i]


class SimpleCachedDataset(Dataset):
    """
    Simple cached dataset (kept for backward compatibility).
    Caches all items in RAM.
    """
    def __init__(self, dataset):
        # Runtime transforms are not implemented in this simple cached dataset.
        self.data = tuple([x for x in dataset])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def load_datasets(config):
    """
    Load datasets without creating DataLoaders.
    This is useful for loading datasets once before a sweep.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        tuple: (train_dataset, test_dataset)
    """
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
        return train_dataset, test_dataset
    
    elif config['dataset']['name'] == 'CIFAR10':
        mean = list(map(float, config["dataset"]["mean"]))
        std = list(map(float, config["dataset"]["std"]))
        train_transformer = transforms.Compose([
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
        return train_dataset, test_dataset
    
    elif config['dataset']['name'] == 'CIFAR100':
        mean = list(map(float, config["dataset"]["mean"]))
        std = list(map(float, config["dataset"]["std"]))
        train_transformer = transforms.Compose([
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
        return train_dataset, test_dataset
    
    elif config['dataset']['name'] == 'CIFAR100-N':
        mean = list(map(float, config["dataset"]["mean"]))
        std = list(map(float, config["dataset"]["std"]))
        randaugment_num_ops = config.get('augmentation', {}).get('randaugment_num_ops', 2)
        # Use a default magnitude for initial loading (will be overridden if needed)
        randaugment_magnitude = config.get('augmentation', {}).get('randaugment_magnitude', 9)
        
        # Get resize resolution from config (default: 128)
        resize_resolution = config.get('dataset', {}).get('resize_resolution', 128)
        if isinstance(resize_resolution, (int, float)):
            resize_resolution = (int(resize_resolution), int(resize_resolution))
        elif isinstance(resize_resolution, (list, tuple)) and len(resize_resolution) == 2:
            resize_resolution = (int(resize_resolution[0]), int(resize_resolution[1]))
        else:
            resize_resolution = (128, 128)  # Default fallback
        
        train_transformer = v2.Compose([
            v2.ToImage(),
            v2.Resize(resize_resolution, antialias=True),
            v2.RandAugment(num_ops=randaugment_num_ops, magnitude=randaugment_magnitude),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean, std, inplace=True)
        ])
        test_transformer = v2.Compose([
            v2.ToImage(),
            v2.Resize(resize_resolution, antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean, std, inplace=True)
        ])
        
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        data_root = str((project_root / config["dataset"]["data_dir"]).resolve())
        
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
        
        train_dataset = HybridCachedDataset(train_dataset, config, cache_prefix="train")
        test_dataset = HybridCachedDataset(test_dataset, config, cache_prefix="test")
        return train_dataset, test_dataset
    
    elif config['dataset']['name'] == 'OxfordIIITPet':
        mean = list(map(float, config["dataset"]["mean"]))
        std = list(map(float, config["dataset"]["std"]))
        train_transformer = transforms.Compose([
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
        return train_dataset, test_dataset
    else:
        print('The dataset name you have entered is not supported!')
        sys.exit()


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


def preprocessing(config, train_dataset=None, test_dataset=None):
    """
    Preprocess dataset and create DataLoaders.
    
    Args:
        config: Configuration dictionary
        train_dataset: Optional pre-loaded train dataset (if provided, will skip dataset loading)
        test_dataset: Optional pre-loaded test dataset (if provided, will skip dataset loading)
    
    Returns:
        tuple: (train_loader, test_loader) or (train_dataset, test_dataset) if return_datasets=True
    """
    # If datasets are provided, just create DataLoaders
    if train_dataset is not None and test_dataset is not None:
        pin_memory = config.get('dataset', {}).get('pin_memory', True)
        batch_size = config['dataset']['batch_size']
        num_workers = config.get('dataset', {}).get('num_workers', 0)
        
        # For cached datasets, reduce workers to avoid memory duplication
        # Each worker process gets its own copy of the dataset in memory
        if isinstance(train_dataset, HybridCachedDataset) and num_workers > 0:
            # Use persistent_workers to avoid recreating workers each epoch
            # This reduces memory overhead
            persistent_workers = True
            # Reduce workers for cached datasets to save memory
            # With cached data, workers are less critical since data is already loaded
            if num_workers > 2:
                num_workers = 2
        else:
            persistent_workers = num_workers > 0
        
        # Check if we need a collate function (for CIFAR100-N with CutMix/MixUp)
        collate_fn = None
        if config['dataset']['name'] == 'CIFAR100-N':
            num_classes = 100
            cutmix_alpha = config.get('augmentation', {}).get('cutmix_alpha', 0.5)
            mixup_alpha = config.get('augmentation', {}).get('mixup_alpha', 0.5)
            collate_fn = create_cutmix_mixup_collate_fn(num_classes, cutmix_alpha, mixup_alpha)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            persistent_workers=persistent_workers
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers
        )
        return train_loader, test_loader
    
    # Otherwise, load datasets as before
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
        
        # Get resize resolution from config (default: 128)
        resize_resolution = config.get('dataset', {}).get('resize_resolution', 128)
        if isinstance(resize_resolution, (int, float)):
            resize_resolution = (int(resize_resolution), int(resize_resolution))
        elif isinstance(resize_resolution, (list, tuple)) and len(resize_resolution) == 2:
            resize_resolution = (int(resize_resolution[0]), int(resize_resolution[1]))
        else:
            resize_resolution = (128, 128)  # Default fallback
        
        # Training transforms with RandAugment (per-sample augmentation)
        train_transformer = v2.Compose([
            v2.ToImage(),
            v2.Resize(resize_resolution, antialias=True),
            v2.RandAugment(num_ops=randaugment_num_ops, magnitude=randaugment_magnitude),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean, std, inplace=True)
        ])
        test_transformer = v2.Compose([
            v2.ToImage(),
            v2.Resize(resize_resolution, antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean, std, inplace=True)
        ])
        
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        data_root = str((project_root / config["dataset"]["data_dir"]).resolve())

        print(str(data_root))
        
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
        
        train_dataset = HybridCachedDataset(train_dataset, config, cache_prefix="train")
        test_dataset = HybridCachedDataset(test_dataset, config, cache_prefix="test")
        
        # Create collate function for CutMix/MixUp (batch-level augmentation)
        train_collate_fn = create_cutmix_mixup_collate_fn(num_classes, cutmix_alpha, mixup_alpha)
        
        pin_memory = config.get('dataset', {}).get('pin_memory', True)
        num_workers = config.get('dataset', {}).get('num_workers', 0)
        
        # For cached datasets, reduce workers to avoid memory duplication
        # Each worker process gets its own copy of the dataset in memory
        if isinstance(train_dataset, HybridCachedDataset) and num_workers > 0:
            persistent_workers = True
            # Reduce workers for cached datasets to save memory
            if num_workers > 2:
                num_workers = 2
        else:
            persistent_workers = num_workers > 0
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config['dataset']['batch_size'], 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=train_collate_fn,
            persistent_workers=persistent_workers
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config['dataset']['batch_size'], 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers
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

