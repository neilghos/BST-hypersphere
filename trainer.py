import torch
import torch.nn.functional as F
from anchor import initialize_hypersphere_anchors
from BSTspace import Stage1_BST_Optimizer

import torch.nn as nn
from dataloader import get_dataloaders, sample_targets_excluding_lookup
from nodealligner import Stage2_NodeAligner, stage2_signed_bst_loss, HierarchicalPredictor
from evaluator import SNAPEval, evaluate_pipeline

import numpy as np

def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def snapshot_state_dict(module):
    """Clone a module state dict onto CPU for safe checkpoint selection."""
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='otc', choices=['alpha', 'otc', 'epinions', 'slashdot', 'wiki-rfa', 'wiki-elec', 'all'])
    args = parser.parse_args()
    
    datasets_to_run = ['alpha', 'otc', 'slashdot', 'wiki-elec', 'wiki-rfa', 'epinions'] if args.dataset == 'all' else [args.dataset]

    set_seed(42)
    
    anchor_embeddings = initialize_hypersphere_anchors()
    bst_model = Stage1_BST_Optimizer(anchor_embeddings, neg_margin=0.0)
    optimizer = torch.optim.Adam(bst_model.parameters(), lr=0.01)
    
    print("Stage 1")
    
    epochs = 500
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = bst_model()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 40 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:3d}/{epochs} | BST Loss: {loss.item():.4f}")

    print("\n Final Hypersphere Cosine Similarities ")
    final_anchors = bst_model.get_normalized_anchors()
     
    with torch.no_grad():
        sim_P1_P2 = F.cosine_similarity(final_anchors["P1"], final_anchors["P2"], dim=0)
        sim_P1_P3 = F.cosine_similarity(final_anchors["P1"], final_anchors["P3"], dim=0)
        
        print(f"P1 (Trust) vs P2 (Malicious): {sim_P1_P2.item():.4f}")
        print(f"P1 (Trust) vs P3 (Neutral):   {sim_P1_P3.item():.4f}")
        
        print("\nPseudo-Anchor Relationships with P1 and P2:")
        print(f"Enemy of both vs P1 (Enemy): {F.cosine_similarity(final_anchors['A_enemy_1_enemy_2'], final_anchors['P1'], dim=0):.4f}")
        print(f"Enemy of both vs P2 (Enemy): {F.cosine_similarity(final_anchors['A_enemy_1_enemy_2'], final_anchors['P2'], dim=0):.4f}")
        print(f"Friend of both vs P1 (Friend): {F.cosine_similarity(final_anchors['A_friend_1_friend_2'], final_anchors['P1'], dim=0):.4f}")
        print(f"Friend of both vs P2 (Friend): {F.cosine_similarity(final_anchors['A_friend_1_friend_2'], final_anchors['P2'], dim=0):.4f}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    frozen_anchors = {name: tensor.detach().to(device) for name, tensor in final_anchors.items()}


    import os
    import csv
    
    os.makedirs('results', exist_ok=True)
    if args.dataset == 'all':
        csv_path = 'results/trainer_results_all.csv'
    else:
        csv_path = f'results/trainer_results_{args.dataset}.csv'
        
    fieldnames = ['dataset', 'acc', 'f1_macro', 'f1_pos', 'f1_neg', 'auc']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    for ds_name in datasets_to_run:
        print(f"\n{'='*60}\n   RUNNING DATASET: {ds_name.upper()}\n{'='*60}")
        zero_positive = ds_name in {'wiki-rfa', 'wiki-elec'}


        print("\nSTAGE 2")


        train_loader, val_loader, test_loader, num_nodes, sampling_metadata = get_dataloaders(ds_name, batch_size=1024)
        heldout_targets_by_source = sampling_metadata['heldout_targets_by_source']

        aligner = Stage2_NodeAligner(num_nodes=num_nodes, raw_embed_dim=128, hypersphere_dim=384).to(device)
        optimizer_s2 = torch.optim.Adam(aligner.parameters(), lr=0.005)

        aligner.train()

        epochs_s2 = 10
        for epoch in range(epochs_s2):
            epoch_loss = 0.0

            for batch in train_loader:
                sources = batch['source'].to(device)
                targets = batch['target'].to(device)
                ratings = batch['rating'].to(device)


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

            print(f"Stage 2 | Epoch {epoch + 1:2d}/{epochs_s2} | AUC Loss: {epoch_loss / len(train_loader):.4f}")

        print("\nPipeline Complete: The hypersphere is fully populated.")

        print("\nStarting Stage 3")
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

        for epoch in range(epochs_s3):
            epoch_exist_loss = 0.0
            epoch_sign_loss = 0.0
            aligner.train()
            predictor.train()

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

            print(
                f"Stage 3 | Epoch {epoch + 1:2d}/{epochs_s3} | "
                f"Exist Loss: {epoch_exist_loss / len(train_loader):.4f} | "
                f"Sign Loss: {epoch_sign_loss / len(train_loader):.4f} | "
                f"Val Acc: {val_metrics['acc']:.4f} | "
                f"Val T: {val_threshold:.4f} | "
                f"Best Val Acc: {best_val_acc:.4f}"
            )

        print("\n Stage 3 complete, giving predictions...")
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
        
        from interpretability import run_interpretability_module
        run_interpretability_module(
            aligner, predictor, test_loader, device, frozen_anchors, best_val_threshold, zero_positive
        )
        
        row = {
            'dataset': ds_name,
            'acc': test_metrics['acc'],
            'f1_macro': test_metrics['f1_macro'],
            'f1_pos': test_metrics['f1_pos'],
            'f1_neg': test_metrics['f1_neg'],
            'auc': test_metrics['auc']
        }
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)
