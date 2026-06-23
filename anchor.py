import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['HF_HUB_OFFLINE'] = '1'

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
    
