import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from anchor import initialize_hypersphere_anchors

class Stage1_BST_Optimizer(nn.Module):
    def __init__(self, initial_anchors, neg_margin=0.0):
        super().__init__()
        
        self.anchors = nn.ParameterDict({
            name: nn.Parameter(tensor) for name, tensor in initial_anchors.items()
        })
        
        # neg_margin = 0.0 means enemies are pushed until they are 90 degrees apart.
        # If you want them strictly opposite, set this to -0.5 or -1.0.
        self.neg_margin = neg_margin 
        
    def get_normalized_anchors(self):
        return {name: F.normalize(param, p=2, dim=0) for name, param in self.anchors.items()}

    def forward(self):
        sphere_anchors = self.get_normalized_anchors()
        
        P1 = sphere_anchors["P1"]  
        P2 = sphere_anchors["P2"]  
        
        # --- NEGATIVE REPULSION (Enemies should be pushed apart) ---
        # Formula: relu(cos - neg_margin)
        cos_P1_P2 = F.cosine_similarity(P1, P2, dim=0)
        imbalance_loss_1 = torch.relu(cos_P1_P2 - self.neg_margin) 
        
        A_enemy_1_enemy_2 = sphere_anchors["A_enemy_1_enemy_2"]
        cos_P1_A = F.cosine_similarity(P1, A_enemy_1_enemy_2, dim=0)
        loss_enemy_P1 = torch.relu(cos_P1_A - self.neg_margin)

        cos_P2_A = F.cosine_similarity(P2, A_enemy_1_enemy_2, dim=0)
        loss_enemy_P2 = torch.relu(cos_P2_A - self.neg_margin)
        
        # --- POSITIVE ATTRACTION (Friends should overlap at 1.0) ---
        # Formula: 1.0 - cos
        A_friend_1_friend_2 = sphere_anchors["A_friend_1_friend_2"]
        cos_P1_F = F.cosine_similarity(P1, A_friend_1_friend_2, dim=0)
        loss_friend_P1 = 1.0 - cos_P1_F 
        
        cos_P2_F = F.cosine_similarity(P2, A_friend_1_friend_2, dim=0)
        loss_friend_P2 = 1.0 - cos_P2_F 

        # --- MIXED POLARITY ---
        A_friend_1_enemy_2 = sphere_anchors["A_friend_1_enemy_2"]
        cos_P1_F1E2 = F.cosine_similarity(P1, A_friend_1_enemy_2, dim=0)
        loss_F1E2_P1 = 1.0 - cos_P1_F1E2 # Pull to P1
        
        cos_P2_F1E2 = F.cosine_similarity(P2, A_friend_1_enemy_2, dim=0)
        loss_F1E2_P2 = torch.relu(cos_P2_F1E2 - self.neg_margin) # Push from P2

        A_enemy_1_friend_2 = sphere_anchors["A_enemy_1_friend_2"]
        cos_P1_E1F2 = F.cosine_similarity(P1, A_enemy_1_friend_2, dim=0)
        loss_E1F2_P1 = torch.relu(cos_P1_E1F2 - self.neg_margin) # Push from P1
        
        cos_P2_E1F2 = F.cosine_similarity(P2, A_enemy_1_friend_2, dim=0)
        loss_E1F2_P2 = 1.0 - cos_P2_E1F2 # Pull to P2

        # Sum it all up
        total_bst_loss = (imbalance_loss_1 + 
                          loss_enemy_P1 + loss_enemy_P2 + 
                          loss_friend_P1 + loss_friend_P2 + 
                          loss_F1E2_P1 + loss_F1E2_P2 + 
                          loss_E1F2_P1 + loss_E1F2_P2)
        
        return total_bst_loss