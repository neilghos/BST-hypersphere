"""
Stage 1 optimization of the BST hypersphere geometry.

This module is intentionally minimal: it does not define a multilayer neural
network that maps inputs to outputs. Instead, the anchor vectors themselves are
the learnable parameters. The role of Stage 1 is not feature extraction from
data, but direct geometric refinement of a small semantic anchor system.

Why there are no layers:
    The anchors already come from `anchor.py` as initialized semantic vectors.
    Stage 1 only needs to move those vectors so their relative cosine geometry
    satisfies the desired signed-balance constraints. A deep network would add
    unnecessary function approximation when the actual object being optimized is
    the anchor arrangement itself.

What optimization is trying to achieve:
    - keep antagonistic poles separated (`P1` vs `P2`)
    - push "enemy of both" away from both poles
    - pull "friend of both" toward both poles
    - pull/push the polarized pseudo-anchors toward one pole and away from the
      opposite pole

The final output of this stage is a refined set of normalized anchor directions
on the unit hypersphere. Later stages use those directions as the signed
semantic reference frame for node alignment and downstream edge prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Stage1_BST_Optimizer(nn.Module):
    """
    Directly optimize the Stage 1 anchor geometry on the unit hypersphere.

    Args:
        initial_anchors (dict[str, torch.Tensor]):
            Anchor vectors from `anchor.py`.
        neg_margin (float):
            Maximum allowed cosine similarity for pairs that should remain
            separated. With the default 0.0, "push apart" means "do not let the
            cosine become positive."

    Implementation detail:
        The model stores one learnable parameter per anchor inside a
        `nn.ParameterDict`. There are no hidden layers because the anchor
        coordinates themselves are the object of optimization.
    """
    def __init__(self, initial_anchors, neg_margin=0.0):
        super().__init__()
        
        self.anchors = nn.ParameterDict({
            name: nn.Parameter(tensor) for name, tensor in initial_anchors.items()
        })
        
        self.neg_margin = neg_margin 
        
    def get_normalized_anchors(self):
        """
        Project every learnable anchor back onto the unit hypersphere.

        This is done every forward pass so all geometric comparisons remain pure
        directional cosine comparisons rather than being influenced by vector
        magnitude.
        """
        return {name: F.normalize(param, p=2, dim=0) for name, param in self.anchors.items()}

    def forward(self):
        """
        Compute the Stage 1 BST geometry loss.

        Loss components:
            1. `P1` vs `P2`:
               Keep the trust and malicious poles separated.
            2. `A_enemy_1_enemy_2`:
               Push the "enemy of both" anchor away from both poles.
            3. `A_friend_1_friend_2`:
               Pull the "friend of both" anchor toward both poles.
            4. `A_friend_1_enemy_2`:
               Pull toward `P1`, push away from `P2`.
            5. `A_enemy_1_friend_2`:
               Push away from `P1`, pull toward `P2`.

        Goal of the full objective:
            Arrange the anchor system so its cosine geometry encodes the intended
            signed-balance structure before any graph-dependent learning begins.
            The returned scalar is minimized by gradient descent during Stage 1.
        """
        sphere_anchors = self.get_normalized_anchors()
        
        P1 = sphere_anchors["P1"]  
        P2 = sphere_anchors["P2"]  
        
        cos_P1_P2 = F.cosine_similarity(P1, P2, dim=0)
        imbalance_loss_1 = torch.relu(cos_P1_P2 - self.neg_margin) 
        
        A_enemy_1_enemy_2 = sphere_anchors["A_enemy_1_enemy_2"]
        cos_P1_A = F.cosine_similarity(P1, A_enemy_1_enemy_2, dim=0)
        loss_enemy_P1 = torch.relu(cos_P1_A - self.neg_margin)

        cos_P2_A = F.cosine_similarity(P2, A_enemy_1_enemy_2, dim=0)
        loss_enemy_P2 = torch.relu(cos_P2_A - self.neg_margin)
        
        A_friend_1_friend_2 = sphere_anchors["A_friend_1_friend_2"]
        cos_P1_F = F.cosine_similarity(P1, A_friend_1_friend_2, dim=0)
        loss_friend_P1 = 1.0 - cos_P1_F 
        
        cos_P2_F = F.cosine_similarity(P2, A_friend_1_friend_2, dim=0)
        loss_friend_P2 = 1.0 - cos_P2_F 

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

        total_bst_loss = (imbalance_loss_1 + 
                          loss_enemy_P1 + loss_enemy_P2 + 
                          loss_friend_P1 + loss_friend_P2 + 
                          loss_F1E2_P1 + loss_F1E2_P2 + 
                          loss_E1F2_P1 + loss_E1F2_P2)
        
        return total_bst_loss
