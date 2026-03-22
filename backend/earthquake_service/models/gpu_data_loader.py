# earthquake_service/training/gpu_data_loader.py
"""
Memory-efficient data loader for GPU training with large datasets.
"""
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import h5py
from pathlib import Path

class EfficientEarthquakeDataset(Dataset):
    """Memory-efficient dataset for large-scale earthquake data"""
    
    def __init__(self, data_path, chunk_size=10000):
        self.data_path = Path(data_path)
        self.chunk_size = chunk_size
        
        # Load metadata
        with h5py.File(self.data_path, 'r') as f:
            self.total_samples = f['features'].shape[0]
            self.feature_dim = f['features'].shape[1]
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        # Load data on-demand to save memory
        with h5py.File(self.data_path, 'r') as f:
            features = f['features'][idx]
            labels = f['labels'][idx]
        
        return torch.FloatTensor(features), torch.LongTensor([labels])[0]

def create_gpu_dataloaders(data_path, batch_size=256, val_split=0.2):
    """Create GPU-optimized dataloaders with prefetching"""
    
    dataset = EfficientEarthquakeDataset(data_path)
    
    # Split dataset
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create dataloaders with GPU optimization
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    return train_loader, val_loader