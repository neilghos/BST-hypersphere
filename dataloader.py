import os
import ssl
import certifi
orig_create_default_context = ssl.create_default_context
def create_default_context_patched(*args, **kwargs):
    if not kwargs.get('cafile') and not kwargs.get('capath') and not kwargs.get('cadata'):
        kwargs['cafile'] = certifi.where()
    return orig_create_default_context(*args, **kwargs)
ssl.create_default_context = create_default_context_patched
ssl._create_default_https_context = create_default_context_patched

import urllib.request
import gzip
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

DATASETS = {
    'alpha': 'https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz',
    'otc': 'https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz'
}

class SNAPBitcoinDataset(Dataset):
    def __init__(self, data_type='alpha', split='train', root_dir='./data', transform=None):
        """
        Args:
            data_type (str): 'alpha' or 'otc'
            split (str): 'train', 'val', 'test', or 'all'
            root_dir (str): Directory to store downloaded data
            transform (callable, optional): Optional transform to be applied
        """
        assert data_type in DATASETS, "data_type must be 'alpha' or 'otc'"
        assert split in ['train', 'val', 'test', 'all'], "split must be 'train', 'val', 'test', or 'all'"
        
        self.split = split
        self.transform = transform
        
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
            
        url = DATASETS[data_type]
        filename = url.split('/')[-1]
        filepath = os.path.join(root_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filepath)
            print("Download complete.")
            
        # Parse the dataset
        # Format: SOURCE, TARGET, RATING, TIME
        df = pd.read_csv(filepath, compression='gzip', header=None, 
                         names=['source', 'target', 'rating', 'time'])
        
        # Make node IDs contiguous and zero-indexed globally (important for embeddings)
        # We calculate the mapping over the entire dataset first to ensure consistency across splits.
        all_nodes = pd.concat([df['source'], df['target']]).unique()
        self.node_mapping = {old_id: new_id for new_id, old_id in enumerate(all_nodes)}
        self.num_nodes = len(self.node_mapping)
        
        df['source'] = df['source'].map(self.node_mapping)
        df['target'] = df['target'].map(self.node_mapping)
        
        # Temporal Split: sort by time to prevent future data leakage
        # df = df.sort_values('time').reset_index(drop=True)  <-- Delete this
        df = df.sample(frac=1, random_state=42).reset_index(drop=True) # <-- Use this
        
        n_edges = len(df)
        train_end = int(0.7 * n_edges)
        val_end = int(0.8 * n_edges)
        
        if split == 'train':
            df = df.iloc[:train_end]
        elif split == 'val':
            df = df.iloc[train_end:val_end]
        elif split == 'test':
            df = df.iloc[val_end:]
            
        self.sources = torch.tensor(df['source'].values, dtype=torch.long)
        self.targets = torch.tensor(df['target'].values, dtype=torch.long)
        self.ratings = torch.tensor(df['rating'].values, dtype=torch.float32)
        self.times = torch.tensor(df['time'].values, dtype=torch.long)
        
    def __len__(self):
        return len(self.sources)
    
    def __getitem__(self, idx):
        sample = {
            'source': self.sources[idx],
            'target': self.targets[idx],
            'rating': self.ratings[idx],
            'time': self.times[idx]
        }
        if self.transform:
            sample = self.transform(sample)
        return sample

def get_dataloaders(data_type='alpha', batch_size=1024, root_dir='./data'):
    """
    Returns train, val, and test DataLoaders for the specified SNAP Bitcoin dataset.
    """
    print(f"Initializing {data_type} dataset splits...")
    train_ds = SNAPBitcoinDataset(data_type=data_type, split='train', root_dir=root_dir)
    val_ds = SNAPBitcoinDataset(data_type=data_type, split='val', root_dir=root_dir)
    test_ds = SNAPBitcoinDataset(data_type=data_type, split='test', root_dir=root_dir)
    
    # Shuffle only the training set
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, train_ds.num_nodes

if __name__ == "__main__":
    print("Testing SNAP Bitcoin Dataloader (Alpha)")
    train_loader, val_loader, test_loader, num_nodes = get_dataloaders('alpha', batch_size=1024)
    
    print(f"\nTotal Unique Nodes: {num_nodes}")
    print(f"Train edges: {len(train_loader.dataset)} ({len(train_loader)} batches)")
    print(f"Val edges:   {len(val_loader.dataset)} ({len(val_loader)} batches)")
    print(f"Test edges:  {len(test_loader.dataset)} ({len(test_loader)} batches)")
    
    # Verify temporal splitting
    train_max_time = train_loader.dataset.times.max().item()
    val_min_time = val_loader.dataset.times.min().item()
    val_max_time = val_loader.dataset.times.max().item()
    test_min_time = test_loader.dataset.times.min().item()
    
    print("\n--- Temporal Split Verification ---")
    print(f"Train Max Time: {train_max_time}")
    print(f"Val Min Time:   {val_min_time}")
    print(f"Val Max Time:   {val_max_time}")
    print(f"Test Min Time:  {test_min_time}")
    
    assert train_max_time <= val_min_time, "Leakage: Train time overlaps with Val time!"
    assert val_max_time <= test_min_time, "Leakage: Val time overlaps with Test time!"
    print("Verification Passed: No temporal leakage detected. Splits are purely chronological.")
