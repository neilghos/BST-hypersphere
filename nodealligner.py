import torch
import torch.nn as nn
import torch.nn.functional as F

class Stage2_NodeAligner(nn.Module):
    def __init__(self, num_nodes, raw_embed_dim=128, hypersphere_dim=384):
        super().__init__()
        
        self.node_embeds = nn.Embedding(num_nodes, raw_embed_dim)
        
        self.projector = nn.Sequential(
            nn.Linear(raw_embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, hypersphere_dim)
        )
        
    def forward(self, node_ids):
        raw_x = self.node_embeds(node_ids)
        
        projected_x = self.projector(raw_x)
        
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
    
    weights = torch.abs(ratings) / 10.0
    
    pos_mask = ratings > 0
    neg_mask = ratings < 0
    
    total_loss = 0.0
    
    if pos_mask.any():
        u_pos = u_embeds[pos_mask]
        v_pos = v_embeds[pos_mask]
        w_pos = weights[pos_mask]
        
        pull_u_P1 = 1.0 - F.cosine_similarity(u_pos, P1.unsqueeze(0), dim=-1)
        pull_v_P1 = 1.0 - F.cosine_similarity(v_pos, P1.unsqueeze(0), dim=-1)
        
        total_loss += torch.sum(w_pos * (pull_u_P1 + pull_v_P1))
        
    if neg_mask.any():
        u_neg = u_embeds[neg_mask]
        v_neg = v_embeds[neg_mask]
        w_neg = weights[neg_mask]
        
        pull_u_P1 = 1.0 - F.cosine_similarity(u_neg, P1.unsqueeze(0), dim=-1)
        pull_v_P2 = 1.0 - F.cosine_similarity(v_neg, P2.unsqueeze(0), dim=-1)
        
        total_loss += torch.sum(w_neg * (pull_u_P1 + pull_v_P2))
        
    return total_loss / len(ratings)


def stage2_pairwise_auc_loss(u_embeds, v_pos_embeds, v_neg_embeds, anchors, margin=0.2):
    pos_scores = F.cosine_similarity(u_embeds, v_pos_embeds, dim=-1)
    neg_scores = F.cosine_similarity(u_embeds, v_neg_embeds, dim=-1)
    
    auc_loss = torch.relu(margin - (pos_scores - neg_scores)).mean()
    
    P1 = anchors["P1"].to(u_embeds.device)
    gravity_loss = (1.0 - F.cosine_similarity(u_embeds, P1.unsqueeze(0), dim=-1)).mean()
    
    return auc_loss + (0.1 * gravity_loss)

class HierarchicalPredictor(nn.Module):
    def __init__(self, embed_dim=384):
        super().__init__()
        
        in_features = embed_dim * 3
        
        self.exist_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.sign_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, u_embeds, v_embeds):
        concat_feats = torch.cat([u_embeds, v_embeds], dim=-1)
        dot_feats = u_embeds * v_embeds
        x = torch.cat([concat_feats, dot_feats], dim=-1)
        
        exist_logits = self.exist_head(x).squeeze(-1)
        sign_logits = self.sign_head(x).squeeze(-1)
        
        return exist_logits, sign_logits