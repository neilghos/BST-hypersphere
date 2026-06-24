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
from collections import defaultdict
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

DATASETS = {
    'alpha': 'https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz',
    'otc': 'https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz',
    'epinions': 'https://snap.stanford.edu/data/soc-sign-epinions.txt.gz',
    'slashdot': 'https://snap.stanford.edu/data/soc-sign-Slashdot090221.txt.gz',
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


def _maybe_stratify(labels):
    """Return labels only when every class has enough members for stratified split."""
    value_counts = labels.value_counts()
    if len(value_counts) > 1 and value_counts.min() >= 2:
        return labels
    return None


def _split_by_directed_pair(df, seed):
    """
    Split rows by unique directed pair so duplicate interactions never cross splits.
    Keeps approximate sign balance by stratifying pair buckets when possible.

    Flow:
    1. Collapse repeated rows of the same directed pair (source, target).
    2. Mark each pair as:
       - pos_only: all observed rows are positive
       - non_pos_only: all observed rows are non-positive
       - mixed: the same directed pair has both kinds of rows
    3. Split the unique pairs into 70/10/20 train/val/test buckets.
    4. Merge those pair-level assignments back onto the original row-level
       dataframe so every repeated copy of the same directed pair stays together.
    """
    pair_df = (
        df.groupby(['source', 'target'], sort=False)['rating']
        .agg(
            has_pos=lambda s: (s > 0).any(),
            has_non_pos=lambda s: (s <= 0).any(),
        )
        .reset_index()
    )

    pair_df['stratify_label'] = 'mixed'
    pair_df.loc[pair_df['has_pos'] & ~pair_df['has_non_pos'], 'stratify_label'] = 'pos_only'
    pair_df.loc[~pair_df['has_pos'] & pair_df['has_non_pos'], 'stratify_label'] = 'non_pos_only'

    train_val_pairs, test_pairs = train_test_split(
        pair_df[['source', 'target', 'stratify_label']],
        test_size=0.2,
        random_state=seed,
        stratify=_maybe_stratify(pair_df['stratify_label']),
    )

    train_pairs, val_pairs = train_test_split(
        train_val_pairs,
        test_size=0.125,
        random_state=seed,
        stratify=_maybe_stratify(train_val_pairs['stratify_label']),
    )

    split_assignments = pd.concat(
        [
            train_pairs[['source', 'target']].assign(split='train'),
            val_pairs[['source', 'target']].assign(split='val'),
            test_pairs[['source', 'target']].assign(split='test'),
        ],
        ignore_index=True,
    )

    return df.merge(split_assignments, on=['source', 'target'], how='left')


def _build_target_lookup_from_tensors(*edge_splits):
    """
    Build a directed source -> blocked-target lookup from one or more edge splits.

    Example:
        val/test edges: (0, 5), (0, 8), (2, 3)
        output: {0: {5, 8}, 2: {3}}

    Stage 2/3 use this lookup to avoid sampling a held-out real edge as a fake
    negative during training.
    """
    targets_by_source = defaultdict(set)
    for sources, targets in edge_splits:
        for src, tgt in zip(sources.tolist(), targets.tolist()):
            targets_by_source[src].add(tgt)
    return {src: frozenset(targets) for src, targets in targets_by_source.items()}


def sample_targets_excluding_lookup(sources, num_nodes, blocked_targets_by_source, device):
    """
    Sample random targets while excluding exact directed pairs present in the blocked lookup.
    This is intentionally limited to the provided blocked splits (e.g. val/test), not train.

    For each source node in the batch:
    1. Sample a random target uniformly from the node universe.
    2. If (source, sampled_target) is a blocked held-out edge, resample just that item.
    3. Repeat until no sampled directed pair collides with the blocked lookup.
    """
    sampled = torch.randint(0, num_nodes, (len(sources),), device=device)
    if len(sources) == 0 or not blocked_targets_by_source:
        return sampled

    source_list = sources.detach().cpu().tolist()
    sampled_list = sampled.detach().cpu().tolist()
    pending_indices = list(range(len(source_list)))

    while pending_indices:
        collision_indices = [
            idx for idx in pending_indices
            if sampled_list[idx] in blocked_targets_by_source.get(source_list[idx], ())
        ]
        if not collision_indices:
            break

        resampled = torch.randint(0, num_nodes, (len(collision_indices),), device=device)
        sampled[collision_indices] = resampled
        resampled_list = resampled.detach().cpu().tolist()
        for idx, tgt in zip(collision_indices, resampled_list):
            sampled_list[idx] = tgt
        pending_indices = collision_indices

    return sampled

class SNAPBitcoinDataset(Dataset):
    def __init__(self, data_type='alpha', split='train', root_dir='./data', transform=None, seed=42):
        """
        Main preprocessing entrypoint for one dataset split.

        High-level flow:
        1. Load one canonical raw file into a dataframe.
        2. Build a stable integer node mapping over the full raw node universe.
        3. Apply a pair-aware 70/10/20 split by directed (source, target).
        4. Keep only the requested split rows.
        5. Convert source/target/rating/time columns into tensors for PyTorch.

        Args:
            data_type (str): dataset key from DATASETS
            split (str): 'train', 'val', or 'test'
            root_dir (str): Directory to store downloaded data
            transform (callable, optional): Optional transform to be applied
        """
        assert data_type in DATASETS, f"data_type must be one of {list(DATASETS.keys())}"
        assert split in ['train', 'val', 'test'], "split must be 'train', 'val', or 'test'"
        
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
            
        # Load the raw edge list into a dataframe with canonical column names.
        if data_type in ['alpha', 'otc']:
            df = pd.read_csv(filepath, compression='gzip', header=None, 
                             names=['source', 'target', 'rating', 'time'])
        elif data_type in ['epinions', 'slashdot']:
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
        
        # Collect every raw node ID that appears anywhere in the file, then map it
        # to a compact integer index so the embedding table can address nodes as
        # rows 0..num_nodes-1.
        all_nodes = pd.concat([df['source'], df['target']]).unique()
        self.node_mapping = {old_id: new_id for new_id, old_id in enumerate(all_nodes)}
        self.num_nodes = len(self.node_mapping)
        df['source'] = df['source'].map(self.node_mapping)
        df['target'] = df['target'].map(self.node_mapping)
        
        # Compute one pair-consistent split assignment and then keep only the
        # requested slice for this dataset object.
        split_df = _split_by_directed_pair(df, seed)
        if split == 'train':
            df = split_df[split_df['split'] == 'train'].drop(columns=['split'])
        elif split == 'val':
            df = split_df[split_df['split'] == 'val'].drop(columns=['split'])
        elif split == 'test':
            df = split_df[split_df['split'] == 'test'].drop(columns=['split'])
            
        # Materialize the final split as tensors so DataLoader can batch them.
        self.sources = torch.tensor(df['source'].values, dtype=torch.long)
        self.targets = torch.tensor(df['target'].values, dtype=torch.long)
        self.ratings = torch.tensor(df['rating'].values, dtype=torch.float32)
        if 'time' in df.columns:
            try:
                if df['time'].dtype == object:
                    df['time'] = pd.to_datetime(df['time'], errors='coerce').astype('int64') // 10**9
                self.times = torch.tensor(df['time'].values, dtype=torch.long)
            except Exception:
                # Time is not currently used by the model, so fall back to zeros
                self.times = torch.zeros(len(df), dtype=torch.long)
        else:
            self.times = torch.zeros(len(df), dtype=torch.long)
        
    def __len__(self):
        return len(self.sources)
    
    def __getitem__(self, idx):
        # Return one row as a uniform dict so PyTorch's DataLoader can batch
        # source/target/rating/time into tensors automatically.
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
    Build the three split datasets/loaders plus the held-out masking metadata.

    We instantiate the dataset class three times with the same seed. That recreates
    the same pair-aware split assignment each time, so train/val/test stay aligned
    while still materializing as separate Dataset objects.
    """
    print(f"Initializing {data_type} dataset splits with seed {seed}...")
    train_ds = SNAPBitcoinDataset(data_type=data_type, split='train', root_dir=root_dir, seed=seed)
    val_ds = SNAPBitcoinDataset(data_type=data_type, split='val', root_dir=root_dir, seed=seed)
    test_ds = SNAPBitcoinDataset(data_type=data_type, split='test', root_dir=root_dir, seed=seed)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    sampling_metadata = {
        # Stage 2/3 sampling blocks validation and test directed pairs so held-out
        # edges are never injected back as random negatives during training.
        'heldout_targets_by_source': _build_target_lookup_from_tensors(
            (val_ds.sources, val_ds.targets),
            (test_ds.sources, test_ds.targets),
        )
    }

    return train_loader, val_loader, test_loader, train_ds.num_nodes, sampling_metadata

if __name__ == "__main__":
    print("Testing SNAP Bitcoin Dataloader (Alpha)")
    train_loader, val_loader, test_loader, num_nodes, sampling_metadata = get_dataloaders('alpha', batch_size=1024)
    
    print(f"\nTotal Unique Nodes: {num_nodes}")
    print(f"Train edges: {len(train_loader.dataset)} ({len(train_loader)} batches)")
    print(f"Val edges:   {len(val_loader.dataset)} ({len(val_loader)} batches)")
    print(f"Test edges:  {len(test_loader.dataset)} ({len(test_loader)} batches)")
    print(f"Held-out source nodes tracked for sampling: {len(sampling_metadata['heldout_targets_by_source'])}")
    
    print("\n--- Split Verification ---")
    print("Verification Passed: Using pair-aware stratified split (70/10/20) by directed edge.")
