"""
Multi-seed benchmark runner for BST pipeline.
Runs the full pipeline N times with different random seeds and logs results to CSV.

Usage:
    python benchmark.py --dataset otc --runs 10
    python benchmark.py --dataset alpha --runs 10
    python benchmark.py --dataset epinions --runs 5
    python benchmark.py --dataset slashdot --runs 5
"""

import argparse
import csv
import os
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from anchor import initialize_hypersphere_anchors
from BSTspace import Stage1_BST_Optimizer
from dataloader import get_dataloaders, sample_targets_excluding_lookup
from nodealligner import Stage2_NodeAligner, stage2_pairwise_auc_loss, stage2_signed_bst_loss, HierarchicalPredictor
from evaluator import SNAPEval, evaluate_pipeline


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def snapshot_state_dict(module):
    """Clone a module state dict onto CPU for safe checkpoint selection."""
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def run_single_dataset(dataset, seed, device, frozen_anchors):
    """Run full Stage 2 + Stage 3 experiment for a dataset and return test metrics."""
    set_seed(seed)
    
    zero_positive = dataset in {'wiki-rfa', 'wiki-elec'}
    
    train_loader, val_loader, test_loader, num_nodes, sampling_metadata = get_dataloaders(
        dataset, batch_size=1024, seed=seed
    )
    heldout_targets_by_source = sampling_metadata['heldout_targets_by_source']
    
    aligner = Stage2_NodeAligner(num_nodes=num_nodes, raw_embed_dim=128, hypersphere_dim=384).to(device)
    optimizer_s2 = torch.optim.Adam(aligner.parameters(), lr=0.005)
    
    aligner.train()
    epochs_s2 = 10
    pbar_s2 = tqdm(range(epochs_s2), desc=f"Stage 2 (Seed {seed})", leave=False, dynamic_ncols=True)
    for epoch in pbar_s2:
        epoch_loss = 0.0
        num_batches = 0
        for batch in train_loader:
            sources = batch['source'].to(device)
            targets = batch['target'].to(device)
            ratings = batch['rating'].to(device)
            
            # Do NOT filter out negative edges!
            v_neg = sample_targets_excluding_lookup(
                sources, num_nodes, heldout_targets_by_source, device
            )
            
            optimizer_s2.zero_grad()
            u_embeds = aligner(sources)
            v_embeds = aligner(targets)
            v_neg_embeds = aligner(v_neg)
            loss_s2 = stage2_signed_bst_loss(
                u_embeds, v_embeds, v_neg_embeds, ratings, frozen_anchors, zero_positive=zero_positive
            )
            loss_s2.backward()
            optimizer_s2.step()
            
            epoch_loss += loss_s2.item()
            num_batches += 1
            
        pbar_s2.set_postfix({'AUC Loss': f"{epoch_loss/max(1, num_batches):.4f}"})
    pbar_s2.close()
    
    predictor = HierarchicalPredictor(embed_dim=384).to(device)
    optimizer_s3 = torch.optim.Adam(list(aligner.parameters()) + list(predictor.parameters()), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    zero_label_policy = 'positive' if zero_positive else 'negative'
    evaluator = SNAPEval(zero_label_policy=zero_label_policy)
    
    epochs_s3 = 10
    best_val_acc = float('-inf')
    best_val_threshold = 0.0
    best_epoch = 0
    best_aligner_state = snapshot_state_dict(aligner)
    best_predictor_state = snapshot_state_dict(predictor)
    pbar_s3 = tqdm(range(epochs_s3), desc=f"Stage 3 (Seed {seed})", leave=False, dynamic_ncols=True)
    for epoch in pbar_s3:
        epoch_exist_loss = 0.0
        epoch_sign_loss = 0.0
        num_batches = 0
        predictor.train()
        aligner.train()
        for batch in train_loader:
            sources = batch['source'].to(device)
            targets = batch['target'].to(device)
            ratings = batch['rating'].to(device)
            
            optimizer_s3.zero_grad()
            
            num_edges = len(sources)
            fake_targets = sample_targets_excluding_lookup(
                sources, num_nodes, heldout_targets_by_source, device
            )
            
            all_u = torch.cat([sources, sources], dim=0)
            all_v = torch.cat([targets, fake_targets], dim=0)
            exist_labels = torch.cat([torch.ones(num_edges, device=device), torch.zeros(num_edges, device=device)], dim=0)
            
            u_embeds = aligner(all_u)
            v_embeds = aligner(all_v)
            exist_logits, sign_logits = predictor(u_embeds, v_embeds)
            
            loss_exist = criterion(exist_logits, exist_labels)
            real_sign_logits = sign_logits[:num_edges]
            sign_labels = (ratings >= 0).float() if zero_positive else (ratings > 0).float()
            loss_sign = criterion(real_sign_logits, sign_labels)
            
            loss_s3 = loss_exist + loss_sign
            loss_s3.backward()
            optimizer_s3.step()
            
            epoch_exist_loss += loss_exist.item()
            epoch_sign_loss += loss_sign.item()
            num_batches += 1

        _, val_labels, val_preds = evaluate_pipeline(
            aligner,
            val_loader,
            device,
            evaluator,
            predictor=predictor,
            split_name="Validation",
            return_raw=True,
            verbose=False,
        )
        val_threshold = evaluator.find_best_threshold(val_labels, val_preds, metric='acc')
        val_metrics = evaluator.eval(
            {
                'y_true': val_labels,
                'y_pred': val_preds,
            },
            threshold=val_threshold,
        )

        if val_metrics['acc'] > best_val_acc:
            best_val_acc = val_metrics['acc']
            best_val_threshold = val_threshold
            best_epoch = epoch + 1
            best_aligner_state = snapshot_state_dict(aligner)
            best_predictor_state = snapshot_state_dict(predictor)
            
        pbar_s3.set_postfix({
            'Exist': f"{epoch_exist_loss/max(1, num_batches):.4f}", 
            'Sign': f"{epoch_sign_loss/max(1, num_batches):.4f}",
            'ValAcc': f"{val_metrics['acc']:.4f}",
            'BestAcc': f"{best_val_acc:.4f}",
        })
    pbar_s3.close()
    aligner.load_state_dict(best_aligner_state)
    predictor.load_state_dict(best_predictor_state)
    print(
        f"Restored best Stage 3 checkpoint from epoch {best_epoch} "
        f"(Val Acc: {best_val_acc:.4f}, Threshold: {best_val_threshold:.4f})"
    )
    
    test_metrics = evaluate_pipeline(
        aligner, test_loader, device, evaluator, predictor=predictor, 
        split_name="Test", threshold=best_val_threshold
    )
    
    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="BST Multi-Seed Benchmark Runner")
    parser.add_argument('--dataset', type=str, default='otc', choices=['alpha', 'otc', 'epinions', 'slashdot', 'wiki-rfa', 'wiki-elec', 'all'])
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='./results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, f"benchmark_{args.dataset}_{args.runs}runs.csv")
    
    datasets_to_run = ['alpha', 'otc', 'slashdot', 'wiki-elec', 'wiki-rfa', 'epinions'] if args.dataset == 'all' else [args.dataset]
    
    print("=" * 60)
    print(f"  BST BENCHMARK: {args.dataset.upper()} | {args.runs} RUNS")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seeds = list(range(42, 42 + args.runs))
    all_results = {ds: [] for ds in datasets_to_run}
    
    fieldnames = ['dataset', 'run', 'seed', 'acc', 'f1_macro', 'f1_pos', 'f1_neg', 'auc']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    
    for run_idx, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"  RUN {run_idx + 1}/{args.runs} | Seed: {seed}")
        print(f"{'='*60}")
        
        # --- Stage 1 ---
        set_seed(seed)
        anchor_embeddings = initialize_hypersphere_anchors()
        bst_model = Stage1_BST_Optimizer(anchor_embeddings, neg_margin=0.0)
        optimizer_s1 = torch.optim.Adam(bst_model.parameters(), lr=0.01)
        
        for epoch in range(500):
            optimizer_s1.zero_grad()
            loss = bst_model()
            loss.backward()
            optimizer_s1.step()
            
        final_anchors = bst_model.get_normalized_anchors()
        frozen_anchors = {name: tensor.detach().to(device) for name, tensor in final_anchors.items()}
        
        for ds in datasets_to_run:
            start_time = time.time()
            metrics = run_single_dataset(ds, seed, device, frozen_anchors)
            elapsed = time.time() - start_time
            
            row = {
                'dataset': ds,
                'run': run_idx + 1,
                'seed': seed,
                'acc': metrics['acc'],
                'f1_macro': metrics['f1_macro'],
                'f1_pos': metrics['f1_pos'],
                'f1_neg': metrics['f1_neg'],
                'auc': metrics['auc']
            }
            all_results[ds].append(row)
            
            with open(csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(row)
            
            print(f"  [{ds}] Run {run_idx + 1} done in {elapsed:.1f}s | AUC: {metrics['auc']:.4f} | F1_NEG: {metrics['f1_neg']:.4f}")
    
    print("\n" + "=" * 60)
    print("  FINAL RESULTS (mean ± std)")
    print("=" * 60)
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        for ds in datasets_to_run:
            ds_results = all_results[ds]
            print(f"--- {ds.upper()} ---")
            summary_mean = {'dataset': ds, 'run': 'MEAN', 'seed': '-'}
            summary_std = {'dataset': ds, 'run': 'STD', 'seed': '-'}
            for metric_name in ['acc', 'f1_macro', 'f1_pos', 'f1_neg', 'auc']:
                values = [r[metric_name] for r in ds_results]
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"  {metric_name.upper():10s}: {mean_val:.4f} ± {std_val:.4f}")
                summary_mean[metric_name] = f"{mean_val:.4f}"
                summary_std[metric_name] = f"{std_val:.4f}"
            
            writer.writerow(summary_mean)
            writer.writerow(summary_std)
    
    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
