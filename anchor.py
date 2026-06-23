"""
Semantic anchor initialization for Stage 1 BST geometry.

This module does not use graph edges or labels. It builds a small set of
language-defined prototype vectors that serve as the starting coordinates for
the signed hypersphere used later by `Stage1_BST_Optimizer`.

Anchor IDs:
    P1:
        Trust / reputable prototype. This is the positive reference point.
    P2:
        Malicious / adversarial prototype. This is the negative reference point.
    P3:
        Neutral / inactive prototype. It provides a semantic midpoint-style
        reference even though the current Stage 1 loss does not directly
        constrain it.
    A_enemy_1_enemy_2:
        A pseudo-anchor representing an entity that is hostile to both P1 and
        P2. In balance-theoretic terms, this is a "common enemy" style node
        that should not collapse onto either prototype.
    A_friend_1_friend_2:
        A pseudo-anchor representing an entity allied with both P1 and P2. This
        encodes a "common friend" relation and is used to pull the geometry
        toward mutual agreement with both reference poles.
    A_friend_1_enemy_2:
        A pseudo-anchor aligned with P1 but opposed to P2.
    A_enemy_1_friend_2:
        A pseudo-anchor opposed to P1 but aligned with P2.

Geometric intent:
    The frozen sentence encoder first maps these textual descriptions into a
    generic semantic vector space. We then mean-center the prompt embeddings and
    L2-normalize them so they lie on a shared unit hypersphere. The result is a
    directional semantic scaffold: trusted, malicious, neutral, and four
    balance-constraint pseudo-anchors already occupy distinct regions before any
    graph training begins.

    Stage 1 optimization later refines this scaffold so cosine relations better
    satisfy the intended signed-balance structure. In other words, this module
    establishes the initial semantic layout; `BSTspace.py` is what sharpens that
    layout into the final BST geometry.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['HF_HUB_OFFLINE'] = '1'


#baggage 
import ssl
import certifi
orig_create_default_context = ssl.create_default_context
def create_default_context_patched(*args, **kwargs):
    if not kwargs.get('cafile') and not kwargs.get('capath') and not kwargs.get('cadata'):
        kwargs['cafile'] = certifi.where()
    return orig_create_default_context(*args, **kwargs)
ssl.create_default_context = create_default_context_patched

from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F

def initialize_hypersphere_anchors():
    """
    Encode the handcrafted anchor prompts into a mean-centered unit hypersphere.

    Returns:
        dict[str, torch.Tensor]:
            Mapping from anchor ID to a normalized embedding vector.

    Processing steps:
        1. Encode each semantic prompt with a frozen sentence encoder.
        2. Stack all prompt embeddings into one matrix.
        3. Subtract the global centroid so the anchor system is centered rather
           than inheriting the encoder's common-language bias.
        4. L2-normalize each row so every anchor lies on the same hypersphere
           and can later be compared using cosine geometry.
    """

    model_name = 'all-MiniLM-L6-v2'
    print(f"Loading frozen text tower: {model_name}...")
    text_tower = SentenceTransformer(model_name)

    anchor_prompts = {
        "P1": "Archetype Trust: A highly trusted, reputable sovereign node with significant positive algebraic credit and strong consensus alliance.",
        "P2": "Archetype Malicious: A malicious, adversarial actor characterized by systemic risk, fraudulent behavior, and high structural distrust.",
        "P3": "Archetype Neutral: A neutral, passive observer node with inert transactional background and no strong polar network alignment.",
        
        "A_enemy_1_enemy_2": "Pseudo-Anchor Constraint: A structural adversary that exhibits explicit hostility, distrust, and negative links to both the highly trusted node and the malicious adversarial node.",
        "A_friend_1_friend_2": "Pseudo-Anchor Constraint: A mutual structural ally that exhibits strong positive consensus and trust with both the highly trusted node and the malicious adversarial node.",
        "A_friend_1_enemy_2": "Pseudo-Anchor Constraint: A polarized node that is deeply allied with the highly trusted node but structurally hostile and distrustful to the malicious adversarial node.",
        "A_enemy_1_friend_2": "Pseudo-Anchor Constraint: A polarized node that is structurally hostile to the highly trusted node but deeply allied and trusted by the malicious adversarial node."
    }

    print("Encoding anchors and projecting to hypersphere...")
    anchor_embeddings = {}
    raw_vectors = []
    keys = []

    with torch.no_grad():
            for anchor_id, prompt in anchor_prompts.items():
                raw_vectors.append(text_tower.encode(prompt, convert_to_tensor=True))
                keys.append(anchor_id)
                
            raw_tensor = torch.stack(raw_vectors)
            

            centroid = raw_tensor.mean(dim=0)
            centered_tensor = raw_tensor - centroid
            

            normalized_tensor = F.normalize(centered_tensor, p=2, dim=1)
            
            for i, key in enumerate(keys):
                anchor_embeddings[key] = normalized_tensor[i]

    return anchor_embeddings

if __name__ == "__main__":
    anchors = initialize_hypersphere_anchors()
    
    print("\nInitial Hypersphere Cosine Similarities (Before Stage 1 BST Optimization):")
    
    sim_P1_P2 = F.cosine_similarity(anchors["P1"], anchors["P2"], dim=0)
    sim_P1_P3 = F.cosine_similarity(anchors["P1"], anchors["P3"], dim=0)
    
    print(f"P1 (Trust) vs P2 (Malicious): {sim_P1_P2.item():.4f}")
    print(f"P1 (Trust) vs P3 (Neutral):   {sim_P1_P3.item():.4f}")
    
