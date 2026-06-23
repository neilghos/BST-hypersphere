import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D

def build_semantic_basis(anchor_dict):
    """Build a fixed 3D semantic basis directly from the optimized Stage 1 anchors."""
    p1 = anchor_dict["P1"].detach().cpu()
    p2 = anchor_dict["P2"].detach().cpu()
    friend_both = anchor_dict["A_friend_1_friend_2"].detach().cpu()
    enemy_both = anchor_dict["A_enemy_1_enemy_2"].detach().cpu()

    # X-axis: P1 - P2 (Trust vs Malicious)
    axis_x = p1 - p2
    axis_x = axis_x / axis_x.norm(p=2).clamp_min(1e-8)

    # Y-axis: Orthogonal component of Friend vs Enemy balance
    raw_y = friend_both - enemy_both
    raw_y = raw_y - torch.dot(raw_y, axis_x) * axis_x
    if raw_y.norm(p=2) < 1e-8:
        fallback = (
            anchor_dict["A_friend_1_enemy_2"].detach().cpu()
            - anchor_dict["A_enemy_1_friend_2"].detach().cpu()
        )
        raw_y = fallback - torch.dot(fallback, axis_x) * axis_x
    axis_y = raw_y / raw_y.norm(p=2).clamp_min(1e-8)

    # Z-axis: Orthogonal component using P3 (Neutral)
    p3 = anchor_dict.get("P3", anchor_dict["A_friend_1_enemy_2"]).detach().cpu()
    raw_z = p3 - torch.dot(p3, axis_x) * axis_x - torch.dot(p3, axis_y) * axis_y
    if raw_z.norm(p=2) < 1e-8:
        # Fallback to random orthogonal vector if P3 is purely in XY plane
        fallback = torch.randn_like(p3)
        raw_z = fallback - torch.dot(fallback, axis_x) * axis_x - torch.dot(fallback, axis_y) * axis_y
    axis_z = raw_z / raw_z.norm(p=2).clamp_min(1e-8)

    return axis_x.numpy(), axis_y.numpy(), axis_z.numpy()

def project_to_semantic_space(points, axis_x, axis_y, axis_z):
    """Project hypersphere points onto the fixed 3D semantic space."""
    x_coords = points @ axis_x
    y_coords = points @ axis_y
    z_coords = points @ axis_z
    return np.column_stack([x_coords, y_coords, z_coords])

def draw_panel_3d(ax, title, anchor_coords, anchor_names, node_coords=None):
    """Draw one evolution panel in 3D with anchors in red and nodes in blue."""
    if node_coords is not None and len(node_coords) > 0:
        ax.scatter(node_coords[:, 0], node_coords[:, 1], node_coords[:, 2], 
                   c="royalblue", s=12, alpha=0.35, label="Train nodes")
        
    ax.scatter(anchor_coords[:, 0], anchor_coords[:, 1], anchor_coords[:, 2], 
               c="crimson", s=120, edgecolors="black", linewidths=0.5, label="Anchors", zorder=3)
    
    for idx, name in enumerate(anchor_names):
        ax.text(anchor_coords[idx, 0], anchor_coords[idx, 1], anchor_coords[idx, 2], 
                name, fontsize=9, color="darkred", fontweight="bold")
        
    ax.set_title(title)
    # Remove axis tick labels for cleaner look
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

def plot_stage(dataset_name, stage_name, anchor_dict, node_embeds=None, output_dir="results"):
    """
    Plots the current state of the hypersphere in 3D.
    - anchor_dict: Dictionary of anchor tensors.
    - node_embeds: Optional tensor of node embeddings to plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    axis_x, axis_y, axis_z = build_semantic_basis(anchor_dict)
    
    anchor_names = list(anchor_dict.keys())
    anchor_points = torch.stack([anchor_dict[name].detach().cpu() for name in anchor_names], dim=0).numpy()
    anchor_coords = project_to_semantic_space(anchor_points, axis_x, axis_y, axis_z)
    
    node_coords = None
    if node_embeds is not None:
        if len(node_embeds) > 1200:
            indices = torch.randperm(len(node_embeds))[:1200]
            sampled_nodes = node_embeds[indices]
        else:
            sampled_nodes = node_embeds
        node_coords = project_to_semantic_space(sampled_nodes.detach().cpu().numpy(), axis_x, axis_y, axis_z)
        
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    draw_panel_3d(ax, f"{stage_name} Geometry ({dataset_name})", anchor_coords, anchor_names, node_coords)
    
    all_coords = anchor_coords
    if node_coords is not None:
        all_coords = np.vstack([anchor_coords, node_coords])
    
    x_min, y_min, z_min = all_coords.min(axis=0)
    x_max, y_max, z_max = all_coords.max(axis=0)
    x_pad = 0.1 * max(1e-6, x_max - x_min)
    y_pad = 0.1 * max(1e-6, y_max - y_min)
    z_pad = 0.1 * max(1e-6, z_max - z_min)
    
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_zlim(z_min - z_pad, z_max + z_pad)
    
    if node_coords is not None:
        ax.legend(loc="upper right")
        
    fig.tight_layout()
    clean_stage_name = stage_name.replace(" ", "_").lower()
    output_path = os.path.join(output_dir, f"plot_{dataset_name}_{clean_stage_name}_3d.png")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved 3D interpretability plot: {output_path}")
