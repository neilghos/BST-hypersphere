import torch
import torch.nn as nn
import torch.nn.functional as F

class Stage2_NodeAligner(nn.Module):
    def __init__(self, num_nodes, raw_embed_dim=128, hypersphere_dim=384):
        super().__init__()
        
        # Base node representations
        self.node_embeds = nn.Embedding(num_nodes, raw_embed_dim)
        
        # The MLP Projector: Maps raw node features to the Stage 1 physics space
        self.projector = nn.Sequential(
            nn.Linear(raw_embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, hypersphere_dim)
        )
        
    def forward(self, node_ids):
        # 1. Get raw embeddings
        raw_x = self.node_embeds(node_ids)
        
        # 2. Project to hypersphere dimension
        projected_x = self.projector(raw_x)
        
        # 3. CRITICAL: Snap to the S^d surface
        return F.normalize(projected_x, p=2, dim=-1)
    

def stage2_pure_positive_loss(u_embeds, v_embeds, ratings, anchors):
    """
    u_embeds: [batch_size, 384] - Source node embeddings
    v_embeds: [batch_size, 384] - Target node embeddings
    ratings: [batch_size] - Edge weights (-10 to 10)
    anchors: dict of frozen Stage 1 tensors
    """
    P1 = anchors["P1"].to(u_embeds.device) # Trust Anchor
    P2 = anchors["P2"].to(u_embeds.device) # Malicious Anchor
    
    # Scale ratings to act as gravity weights (0.0 to 1.0)
    weights = torch.abs(ratings) / 10.0
    
    # Masks for topological routing
    pos_mask = ratings > 0
    neg_mask = ratings < 0
    
    total_loss = 0.0
    
    # --- Positive Edges (Trust) ---
    if pos_mask.any():
        u_pos = u_embeds[pos_mask]
        v_pos = v_embeds[pos_mask]
        w_pos = weights[pos_mask]
        
        # Pull both to P1. Loss = 1 - cos(theta)
        pull_u_P1 = 1.0 - F.cosine_similarity(u_pos, P1.unsqueeze(0), dim=-1)
        pull_v_P1 = 1.0 - F.cosine_similarity(v_pos, P1.unsqueeze(0), dim=-1)
        
        # Apply the gravitational weight
        total_loss += torch.sum(w_pos * (pull_u_P1 + pull_v_P1))
        
    # --- Negative Edges (Distrust) ---
    if neg_mask.any():
        u_neg = u_embeds[neg_mask]
        v_neg = v_embeds[neg_mask]
        w_neg = weights[neg_mask]
        
        # Rater (u) is pulled to P1, Rated (v) is pulled to P2
        pull_u_P1 = 1.0 - F.cosine_similarity(u_neg, P1.unsqueeze(0), dim=-1)
        pull_v_P2 = 1.0 - F.cosine_similarity(v_neg, P2.unsqueeze(0), dim=-1)
        
        total_loss += torch.sum(w_neg * (pull_u_P1 + pull_v_P2))
        
    # Average over the batch
    return total_loss / len(ratings)


def stage2_pairwise_auc_loss(u_embeds, v_pos_embeds, v_neg_embeds, anchors, margin=0.2):
    # 1. The Direct AUC Optimizer (Pairwise Ranking)
    pos_scores = F.cosine_similarity(u_embeds, v_pos_embeds, dim=-1)
    neg_scores = F.cosine_similarity(u_embeds, v_neg_embeds, dim=-1)
    
    # BPR / Margin Loss: We want pos_score to be at least 'margin' higher than neg_score
    auc_loss = torch.relu(margin - (pos_scores - neg_scores)).mean()
    
    # 2. Soft Transductive Gravity (Keep the Stage 1 physics alive)
    # Gently pull the sources to the P1 Trust Anchor so the space doesn't arbitrarily rotate
    P1 = anchors["P1"].to(u_embeds.device)
    gravity_loss = (1.0 - F.cosine_similarity(u_embeds, P1.unsqueeze(0), dim=-1)).mean()
    
    return auc_loss + (0.1 * gravity_loss)

class HierarchicalPredictor(nn.Module):
    def __init__(self, embed_dim=384):
        super().__init__()
        
        # Input features: Concat(u, v) [dim*2] + Dot Product (u * v) [dim] -> Total: dim * 3
        in_features = embed_dim * 3
        
        # 1. Existence Head (Predicts if edge exists)
        self.exist_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # 2. Sign Head (Predicts Trust vs Distrust)
        self.sign_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, u_embeds, v_embeds):
        # Construct the features
        concat_feats = torch.cat([u_embeds, v_embeds], dim=-1)
        dot_feats = u_embeds * v_embeds
        x = torch.cat([concat_feats, dot_feats], dim=-1)
        
        exist_logits = self.exist_head(x).squeeze(-1)
        sign_logits = self.sign_head(x).squeeze(-1)
        
        return exist_logits, sign_logits