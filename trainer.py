import torch
import torch.nn.functional as F
from anchor import initialize_hypersphere_anchors
from BSTspace import Stage1_BST_Optimizer

# Assuming you saved the previous blocks in these files:
# pyrefly: ignore [missing-import]
from dataloader import get_dataloaders
from nodealligner import Stage2_NodeAligner, stage2_pure_positive_loss, stage2_pairwise_auc_loss
from evaluator import SNAPEval, evaluate_pipeline

if __name__ == "__main__":
    # ==========================================
    # STAGE 1: THE PHYSICS (ANCHOR OPTIMIZATION)
    # ==========================================
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

    # ==========================================
    # THE HAND-OFF (FREEZING THE UNIVERSE)
    # ==========================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nMoving computation to: {device}")

    # CRITICAL: Detach from Stage 1 gradient graph and move to GPU
    frozen_anchors = {name: tensor.detach().to(device) for name, tensor in final_anchors.items()}

    print("\n=============================================")
    print("   STAGE 2: TRANSDUCTIVE NODE ALIGNMENT")
    print("=============================================")

    # 1. Load the SNAP Dataset (This handles the download, mapping, and temporal split)
    train_loader, val_loader, test_loader, num_nodes = get_dataloaders('alpha', batch_size=1024)

    # 2. Initialize the Stage 2 Projector Network
    aligner = Stage2_NodeAligner(num_nodes=num_nodes, raw_embed_dim=128, hypersphere_dim=384).to(device)
    optimizer_s2 = torch.optim.Adam(aligner.parameters(), lr=0.005)

    # 3. Stage 2 Training Loop (Pure Positive Pull)
    print("\n--- Starting Stage 2: Pairwise AUC Optimization ---")
    aligner.train()

    epochs_s2 = 100
    for epoch in range(epochs_s2):
        epoch_loss = 0.0
        
        for batch in train_loader:
            sources = batch['source'].to(device)
            targets = batch['target'].to(device)
            ratings = batch['rating'].to(device)
            
            # Grab the Trust Backbone
            pos_mask = ratings > 0
            if not pos_mask.any():
                continue
                
            u_pos = sources[pos_mask]
            v_pos = targets[pos_mask]
            
            # --- THE MAGIC: Dynamic Negative Edge Augmentation ---
            # For every real target, generate a random "fake" target
            v_neg = torch.randint(0, num_nodes, (len(u_pos),), device=device)
            
            optimizer_s2.zero_grad()
            
            # Project all three sets of nodes
            u_embeds = aligner(u_pos)
            v_pos_embeds = aligner(v_pos)
            v_neg_embeds = aligner(v_neg)
            
            # Apply the Pairwise AUC math
            loss_s2 = stage2_pairwise_auc_loss(u_embeds, v_pos_embeds, v_neg_embeds, frozen_anchors)
            
            loss_s2.backward()
            optimizer_s2.step()
            
            epoch_loss += loss_s2.item()
            
        print(f"Stage 2 | Epoch {epoch + 1:2d}/{epochs_s2} | AUC Loss: {epoch_loss / len(train_loader):.4f}")

    print("\nPipeline Complete: The hypersphere is fully populated.")

    # ==========================================
    # STAGE 3: EVALUATION
    # ==========================================
    evaluator = SNAPEval(task_type='sign_prediction')
    
    # Evaluate on Validation Set
    evaluate_pipeline(aligner, val_loader, device, evaluator, split_name="Validation")
    
    # Evaluate on Test Set
    evaluate_pipeline(aligner, test_loader, device, evaluator, split_name="Test")