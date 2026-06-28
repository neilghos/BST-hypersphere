import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

def run_pca_alignment_analysis(node_embeds, anchors, dataset_name):
    """
    Runs PCA on the hypersphere embeddings and checks how well the primary 
    axes of variance align with the P1 and P2 anchors.
    """
    print(f"\n--- Running PCA Alignment Analysis ({dataset_name}) ---")
    
    if isinstance(node_embeds, torch.Tensor):
        X = node_embeds.detach().cpu().numpy()
    else:
        X = node_embeds
        
    p1 = anchors["P1"].detach().cpu().numpy()
    p2 = anchors["P2"].detach().cpu().numpy()
    
    # 1. Fit PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    
    components = pca.components_  # Shape: (3, 384)
    explained_variance = pca.explained_variance_ratio_ * 100
    
    report_lines = [
        f"PCA Alignment Analysis: {dataset_name.upper()}",
        "="*40,
        f"Explained Variance:",
        f"  PC1: {explained_variance[0]:.2f}%",
        f"  PC2: {explained_variance[1]:.2f}%",
        f"  PC3: {explained_variance[2]:.2f}%",
        f"  Total (Top 3): {sum(explained_variance):.2f}%\n",
        "Alignment with Anchors (Signed Cosine Similarity):"
    ]
    
    def get_alignment_str(anchor_name, anchor_vec):
        alignments = []
        for i, pc in enumerate(components):
            cos_sim = np.dot(anchor_vec, pc) / (np.linalg.norm(anchor_vec) * np.linalg.norm(pc))
            alignments.append(f"PC{i+1}: {cos_sim:+.4f}")
        return f"  {anchor_name} -> " + " | ".join(alignments)
        
    report_lines.append(get_alignment_str("P1", p1))
    report_lines.append(get_alignment_str("P2", p2))
    
    report_text = "\n".join(report_lines)
    print(report_text)
    
    os.makedirs("results", exist_ok=True)
    with open(f"results/pca_report_{dataset_name}.txt", "w") as f:
        f.write(report_text + "\n")
        
    # 2. 3D Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Downsample points for visual clarity (max 2000)
    num_points = X_pca.shape[0]
    if num_points > 2000:
        idx = np.random.choice(num_points, 2000, replace=False)
        X_plot = X_pca[idx]
    else:
        X_plot = X_pca
        
    # Plot node cloud
    ax.scatter(X_plot[:, 0], X_plot[:, 1], X_plot[:, 2], 
               c='#a6cee3', alpha=0.3, s=15, label="Nodes (Sampled)")
               
    # Transform anchors into the PCA space
    anchors_transformed = pca.transform(np.vstack([p1, p2]))
    p1_pca = anchors_transformed[0]
    p2_pca = anchors_transformed[1]
    
    # Plot Anchors
    ax.scatter(p1_pca[0], p1_pca[1], p1_pca[2], 
               c='green', marker='*', s=300, edgecolor='black', label="P1 (Pole 1)")
    ax.scatter(p2_pca[0], p2_pca[1], p2_pca[2], 
               c='red', marker='X', s=250, edgecolor='black', label="P2 (Pole 2)")
               
    ax.set_xlabel(f'PC1 ({explained_variance[0]:.1f}%)')
    ax.set_ylabel(f'PC2 ({explained_variance[1]:.1f}%)')
    ax.set_zlabel(f'PC3 ({explained_variance[2]:.1f}%)')
    ax.set_title(f"Hypersphere PCA Projection ({dataset_name.upper()})")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"results/pca_alignment_3d_{dataset_name}.png", dpi=300)
    plt.close()
    
    print(f"Saved PCA report and 3D plot to results/ directory.")
