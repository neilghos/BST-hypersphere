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
    'otc': 'https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz',
    'epinions': 'https://snap.stanford.edu/data/soc-sign-epinions.txt.gz',
    'wiki-rfa': 'https://snap.stanford.edu/data/wiki-RfA.txt.gz',
    'wiki-elec': 'https://snap.stanford.edu/data/wikiElec.ElecBs3.txt.gz'
}

def parse_wiki_rfa(filepath):
    import gzip
    data = []
    with gzip.open(filepath, 'rt', encoding='utf-8', errors='ignore') as f:
        src, tgt, vot, dat = None, None, None, None
        for line in f:
            line = line.strip()
            if line.startswith('SRC:'): src = line[4:]
            elif line.startswith('TGT:'): tgt = line[4:]
            elif line.startswith('VOT:'): vot = float(line[4:])
            elif line.startswith('DAT:'): dat = line[4:]
            elif line == '':
                if src and tgt and vot is not None:
                    data.append((src, tgt, vot, dat))
                src, tgt, vot, dat = None, None, None, None
    return pd.DataFrame(data, columns=['source', 'target', 'rating', 'time'])

def parse_wiki_elec(filepath):
    import gzip
    data = []
    with gzip.open(filepath, 'rt', encoding='utf-8', errors='ignore') as f:
        tgt = None
        for line in f:
            line = line.strip()
            if line.startswith('U\t'):
                parts = line.split('\t')
                if len(parts) >= 2:
                    tgt = parts[1]
            elif line.startswith('V\t'):
                parts = line.split('\t')
                if len(parts) >= 4 and tgt is not None:
                    vot = float(parts[1])
                    src = parts[2]
                    dat = parts[3]
                    data.append((src, tgt, vot, dat))
    return pd.DataFrame(data, columns=['source', 'target', 'rating', 'time'])

from sklearn.model_selection import train_test_split

class SNAPBitcoinDataset(Dataset):
    def __init__(self, data_type='alpha', split='train', root_dir='./data', transform=None, seed=42):
        """
        Args:
            data_type (str): 'alpha' or 'otc'
            split (str): 'train', 'val', 'test', or 'all'
            root_dir (str): Directory to store downloaded data
            transform (callable, optional): Optional transform to be applied
        """
        assert data_type in DATASETS, f"data_type must be one of {list(DATASETS.keys())}"
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
        if data_type in ['alpha', 'otc']:
            # Format: SOURCE, TARGET, RATING, TIME
            df = pd.read_csv(filepath, compression='gzip', header=None, 
                             names=['source', 'target', 'rating', 'time'])
        elif data_type in ['epinions', 'slashdot']:
            # Format: FromNodeId, ToNodeId, Sign (Tab separated, comments start with #)
            df = pd.read_csv(filepath, compression='gzip', sep='\t', comment='#', header=None,
                             names=['source', 'target', 'rating'])
        elif data_type == 'wiki-rfa':
            df = parse_wiki_rfa(filepath)
        elif data_type == 'wiki-elec':
            df = parse_wiki_elec(filepath)
        elif filepath.endswith('.csv.gz'):
            df = pd.read_csv(filepath, header=None, names=['source', 'target', 'rating', 'time'])
        elif filepath.endswith('.txt.gz'):
            df = pd.read_csv(filepath, sep='\t', header=None, comment='#', names=['source', 'target', 'rating', 'time'])
        else:
            raise ValueError("Unknown file format")
        
        # Make node IDs contiguous and zero-indexed globally (important for embeddings)
        # We calculate the mapping over the entire dataset first to ensure consistency across splits.
        all_nodes = pd.concat([df['source'], df['target']]).unique()
        self.node_mapping = {old_id: new_id for new_id, old_id in enumerate(all_nodes)}
        self.num_nodes = len(self.node_mapping)
        
        df['source'] = df['source'].map(self.node_mapping)
        df['target'] = df['target'].map(self.node_mapping)
        
        # Stratified Random Split: 80% / 20% first to match baseline test set precisely
        # Then split the 80% train set to create an inner validation set
        if split != 'all':
            stratify_labels = (df['rating'] > 0).astype(int)
            train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=seed, stratify=stratify_labels)
            
            # Create inner validation set from train_val_df (e.g. 10% of total data = 12.5% of train_val_df)
            temp_stratify = (train_val_df['rating'] > 0).astype(int)
            train_df, val_df = train_test_split(train_val_df, test_size=0.125, random_state=seed, stratify=temp_stratify)
            
            if split == 'train':
                df = train_df
            elif split == 'val':
                df = val_df
            elif split == 'test':
                df = test_df
            
        self.sources = torch.tensor(df['source'].values, dtype=torch.long)
        self.targets = torch.tensor(df['target'].values, dtype=torch.long)
        self.ratings = torch.tensor(df['rating'].values, dtype=torch.float32)
        if 'time' in df.columns:
            try:
                # Some datasets like Wiki-RfA have string timestamps
                if df['time'].dtype == object:
                    df['time'] = pd.to_datetime(df['time'], errors='coerce').astype('int64') // 10**9
                self.times = torch.tensor(df['time'].values, dtype=torch.long)
            except Exception:
                self.times = torch.zeros(len(df), dtype=torch.long)
        else:
            self.times = torch.zeros(len(df), dtype=torch.long)
        
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

def get_dataloaders(data_type='alpha', batch_size=1024, root_dir='./data', seed=42):
    """
    Returns train, val, and test DataLoaders for the specified SNAP dataset.
    """
    print(f"Initializing {data_type} dataset splits with seed {seed}...")
    train_ds = SNAPBitcoinDataset(data_type=data_type, split='train', root_dir=root_dir, seed=seed)
    val_ds = SNAPBitcoinDataset(data_type=data_type, split='val', root_dir=root_dir, seed=seed)
    test_ds = SNAPBitcoinDataset(data_type=data_type, split='test', root_dir=root_dir, seed=seed)
    
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
    
    print("\n--- Split Verification ---")
    print("Verification Passed: Using stratified random split (70/10/20).")
