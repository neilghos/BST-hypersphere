import torch
import torch.nn.functional as F
from anchor import initialize_hypersphere_anchors
from BSTspace import Stage1_BST_Optimizer

import torch.nn as nn
from dataloader import get_dataloaders, sample_targets_excluding_lookup
from nodealligner import Stage2_NodeAligner, stage2_pure_positive_loss, stage2_pairwise_auc_loss, HierarchicalPredictor
from evaluator import SNAPEval, evaluate_pipeline

import numpy as np

def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='otc', choices=['alpha', 'otc', 'epinions', 'wiki-rfa', 'wiki-elec', 'all'])
    args = parser.parse_args()
    
    datasets_to_run = ['alpha', 'otc', 'wiki-elec', 'wiki-rfa','epinions' ] if args.dataset == 'all' else [args.dataset]

    set_seed(42)
    
    anchor_embeddings = initialize_hypersphere_anchors()
    bst_model = Stage1_BST_Optimizer(anchor_embeddings, neg_margin=0.0)
    optimizer = torch.optim.Adam(bst_model.parameters(), lr=0.01)
    
    print("\n--- Starting Stage 1 BST Optimization ---")
    
    epochs = 500
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = bst_model()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 40 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:3d}/{epochs} | BST Loss: {loss.item():.4f}")

    print("\n--- Final Hypersphere Cosine Similarities ---")
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
    print(f"\nMoving computation to: {device}")

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


        print("\n=============================================")
        print("   STAGE 2: TRANSDUCTIVE NODE ALIGNMENT")
        print("=============================================")

        train_loader, val_loader, test_loader, num_nodes, sampling_metadata = get_dataloaders(ds_name, batch_size=1024)
        heldout_targets_by_source = sampling_metadata['heldout_targets_by_source']

        aligner = Stage2_NodeAligner(num_nodes=num_nodes, raw_embed_dim=128, hypersphere_dim=384).to(device)
        optimizer_s2 = torch.optim.Adam(aligner.parameters(), lr=0.005)

        print("\n--- Starting Stage 2: Pairwise AUC Optimization ---")
        aligner.train()

        epochs_s2 = 100
        for epoch in range(epochs_s2):
            epoch_loss = 0.0

            for batch in train_loader:
                sources = batch['source'].to(device)
                targets = batch['target'].to(device)
                ratings = batch['rating'].to(device)

                pos_mask = ratings > 0
                if not pos_mask.any():
                    continue

                u_pos = sources[pos_mask]
                v_pos = targets[pos_mask]

                v_neg = sample_targets_excluding_lookup(
                    u_pos, num_nodes, heldout_targets_by_source, device
                )

                optimizer_s2.zero_grad()

                u_embeds = aligner(u_pos)
                v_pos_embeds = aligner(v_pos)
                v_neg_embeds = aligner(v_neg)

                loss_s2 = stage2_pairwise_auc_loss(u_embeds, v_pos_embeds, v_neg_embeds, frozen_anchors)

                loss_s2.backward()
                optimizer_s2.step()

                epoch_loss += loss_s2.item()

            print(f"Stage 2 | Epoch {epoch + 1:2d}/{epochs_s2} | AUC Loss: {epoch_loss / len(train_loader):.4f}")

        print("\nPipeline Complete: The hypersphere is fully populated.")

        print("\n--- Starting Stage 3: Supervised Blame Game Finetuning ---")
        predictor = HierarchicalPredictor(embed_dim=384).to(device)

        optimizer_s3 = torch.optim.Adam(list(aligner.parameters()) + list(predictor.parameters()), lr=0.001)

        criterion = nn.BCEWithLogitsLoss()

        epochs_s3 = 50
        predictor.train()
        aligner.train()

        for epoch in range(epochs_s3):
            epoch_exist_loss = 0.0
            epoch_sign_loss = 0.0

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

                sign_labels = (ratings > 0).float()

                loss_sign = criterion(real_sign_logits, sign_labels)

                loss_s3 = loss_exist + loss_sign

                loss_s3.backward()
                optimizer_s3.step()

                epoch_exist_loss += loss_exist.item()
                epoch_sign_loss += loss_sign.item()

            print(f"Stage 3 | Epoch {epoch + 1:2d}/{epochs_s3} | Exist Loss: {epoch_exist_loss / len(train_loader):.4f} | Sign Loss: {epoch_sign_loss / len(train_loader):.4f}")

        print("\nPipeline Complete: Predictor fully trained.")

        evaluator = SNAPEval(task_type='sign_prediction')

        _, val_labels, val_preds = evaluate_pipeline(
            aligner, val_loader, device, evaluator, predictor=predictor, 
            split_name="Validation (Untuned)", return_raw=True
        )
        best_t = evaluator.find_best_threshold(val_labels, val_preds, metric='acc')
        print(f"Tuned Threshold for Accuracy: {best_t:.4f}")

        test_metrics = evaluate_pipeline(
            aligner, test_loader, device, evaluator, predictor=predictor, 
            split_name="Test", threshold=best_t
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
