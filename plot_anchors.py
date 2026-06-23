import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

from anchor import initialize_hypersphere_anchors
from BSTspace import Stage1_BST_Optimizer

def plot_tsne():
    print("Loading frozen text tower and encoding anchors...")
    anchor_embeddings = initialize_hypersphere_anchors()
    
    print("Running Stage 1 BST Optimization for 500 epochs...")
    bst_model = Stage1_BST_Optimizer(anchor_embeddings, neg_margin=0.0)
    optimizer = torch.optim.Adam(bst_model.parameters(), lr=0.01)
    
    for epoch in range(500):
        optimizer.zero_grad()
        loss = bst_model()
        loss.backward()
        optimizer.step()
        
    frozen_anchors = bst_model.get_normalized_anchors()
    
    names = list(frozen_anchors.keys())
    tensors = [frozen_anchors[n].detach().cpu().numpy() for n in names]
    X = np.stack(tensors)
    
    print(f"Plotting {len(names)} anchors using t-SNE...")
    
    perplexity = min(3, len(names) - 1)
    
    # 2D t-SNE
    tsne_2d = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_2d = tsne_2d.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c='blue', marker='o', s=100)
    for i, name in enumerate(names):
        plt.annotate(name, (X_2d[i, 0], X_2d[i, 1]), xytext=(5, 5), textcoords='offset points')
    plt.title('t-SNE 2D Projection of BST Hypersphere Anchors')
    plt.grid(True)
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/tsne_2d.png')
    plt.close()
    
    # 3D t-SNE
    tsne_3d = TSNE(n_components=3, perplexity=perplexity, random_state=42)
    X_3d = tsne_3d.fit_transform(X)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c='red', marker='o', s=100)
    for i, name in enumerate(names):
        ax.text(X_3d[i, 0], X_3d[i, 1], X_3d[i, 2], name)
    plt.title('t-SNE 3D Projection of BST Hypersphere Anchors')
    plt.savefig('results/tsne_3d.png')
    plt.close()
    
    print("Saved t-SNE plots to results/tsne_2d.png and results/tsne_3d.png")

if __name__ == '__main__':
    plot_tsne()
