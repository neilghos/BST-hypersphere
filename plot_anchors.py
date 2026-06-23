import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from anchor import initialize_hypersphere_anchors
from BSTspace import Stage1_BST_Optimizer
from dataloader import get_dataloaders, sample_targets_excluding_lookup
from evaluator import SNAPEval, evaluate_pipeline
from nodealligner import Stage2_NodeAligner, stage2_pairwise_auc_loss, HierarchicalPredictor


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def snapshot_state_dict(module):
    """Clone a module state dict onto CPU for safe checkpoint selection."""
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def train_stage1(stage1_epochs):
    """Run Stage 1 and return the optimized anchor dictionary."""
    print("Loading frozen text tower and encoding anchors...")
    anchor_embeddings = initialize_hypersphere_anchors()

    print(f"Running Stage 1 BST Optimization for {stage1_epochs} epochs...")
    bst_model = Stage1_BST_Optimizer(anchor_embeddings, neg_margin=0.0)
    optimizer = torch.optim.Adam(bst_model.parameters(), lr=0.01)

    for epoch in range(stage1_epochs):
        optimizer.zero_grad()
        loss = bst_model()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Stage 1 | Epoch {epoch + 1:3d}/{stage1_epochs} | Loss: {loss.item():.4f}")

    return bst_model.get_normalized_anchors()


def train_stage2(dataset, frozen_anchors, stage2_epochs, batch_size, seed, device):
    """Train Stage 2 and return the split loaders plus the aligned node model."""
    zero_positive = dataset in {"wiki-rfa", "wiki-elec"}
    train_loader, val_loader, test_loader, num_nodes, sampling_metadata = get_dataloaders(
        dataset, batch_size=batch_size, seed=seed
    )
    heldout_targets_by_source = sampling_metadata["heldout_targets_by_source"]

    aligner = Stage2_NodeAligner(num_nodes=num_nodes, raw_embed_dim=128, hypersphere_dim=384).to(device)
    optimizer = torch.optim.Adam(aligner.parameters(), lr=0.005)

    print(f"Running Stage 2 for {stage2_epochs} epochs...")
    aligner.train()
    for epoch in range(stage2_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            sources = batch["source"].to(device)
            targets = batch["target"].to(device)
            ratings = batch["rating"].to(device)

            pos_mask = ratings >= 0 if zero_positive else ratings > 0
            if not pos_mask.any():
                continue

            u_pos = sources[pos_mask]
            v_pos = targets[pos_mask]
            v_neg = sample_targets_excluding_lookup(
                u_pos, num_nodes, heldout_targets_by_source, device
            )

            optimizer.zero_grad()
            u_embeds = aligner(u_pos)
            v_pos_embeds = aligner(v_pos)
            v_neg_embeds = aligner(v_neg)

            loss = stage2_pairwise_auc_loss(u_embeds, v_pos_embeds, v_neg_embeds, frozen_anchors)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        print(
            f"Stage 2 | Epoch {epoch + 1:3d}/{stage2_epochs} | "
            f"AUC Loss: {epoch_loss / max(1, num_batches):.4f}"
        )

    return train_loader, val_loader, test_loader, num_nodes, heldout_targets_by_source, aligner, zero_positive


def train_stage3(
    aligner,
    train_loader,
    val_loader,
    num_nodes,
    heldout_targets_by_source,
    zero_positive,
    stage3_epochs,
    device,
):
    """Jointly fine-tune the aligner and predictor, keeping the best val-acc checkpoint."""
    predictor = HierarchicalPredictor(embed_dim=384).to(device)
    optimizer = torch.optim.Adam(list(aligner.parameters()) + list(predictor.parameters()), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    zero_label_policy = "positive" if zero_positive else "negative"
    evaluator = SNAPEval(task_type="sign_prediction", zero_label_policy=zero_label_policy)

    best_val_acc = float("-inf")
    best_val_threshold = 0.0
    best_epoch = 0
    best_aligner_state = snapshot_state_dict(aligner)
    best_predictor_state = snapshot_state_dict(predictor)

    print(f"Running Stage 3 for {stage3_epochs} epochs...")
    for epoch in range(stage3_epochs):
        epoch_exist_loss = 0.0
        epoch_sign_loss = 0.0
        num_batches = 0

        aligner.train()
        predictor.train()
        for batch in train_loader:
            sources = batch["source"].to(device)
            targets = batch["target"].to(device)
            ratings = batch["rating"].to(device)

            optimizer.zero_grad()

            num_edges = len(sources)
            fake_targets = sample_targets_excluding_lookup(
                sources, num_nodes, heldout_targets_by_source, device
            )

            all_u = torch.cat([sources, sources], dim=0)
            all_v = torch.cat([targets, fake_targets], dim=0)
            exist_labels = torch.cat(
                [torch.ones(num_edges, device=device), torch.zeros(num_edges, device=device)],
                dim=0,
            )

            u_embeds = aligner(all_u)
            v_embeds = aligner(all_v)
            exist_logits, sign_logits = predictor(u_embeds, v_embeds)

            loss_exist = criterion(exist_logits, exist_labels)
            real_sign_logits = sign_logits[:num_edges]
            sign_labels = (ratings >= 0).float() if zero_positive else (ratings > 0).float()
            loss_sign = criterion(real_sign_logits, sign_labels)

            loss = loss_exist + loss_sign
            loss.backward()
            optimizer.step()

            epoch_exist_loss += loss_exist.item()
            epoch_sign_loss += loss_sign.item()
            num_batches += 1

        _, val_labels, val_preds = evaluate_pipeline(
            aligner,
            val_loader,
            device,
            evaluator,
            predictor=predictor,
            split_name="Validation",
            return_raw=True,
            verbose=False,
        )
        val_threshold = evaluator.find_best_threshold(val_labels, val_preds, metric="acc")
        val_metrics = evaluator.eval(
            {
                "y_true": val_labels,
                "y_pred": val_preds,
            },
            threshold=val_threshold,
        )

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_val_threshold = val_threshold
            best_epoch = epoch + 1
            best_aligner_state = snapshot_state_dict(aligner)
            best_predictor_state = snapshot_state_dict(predictor)

        print(
            f"Stage 3 | Epoch {epoch + 1:3d}/{stage3_epochs} | "
            f"Exist Loss: {epoch_exist_loss / max(1, num_batches):.4f} | "
            f"Sign Loss: {epoch_sign_loss / max(1, num_batches):.4f} | "
            f"Val Acc: {val_metrics['acc']:.4f} | "
            f"Best Val Acc: {best_val_acc:.4f}"
        )

    aligner.load_state_dict(best_aligner_state)
    predictor.load_state_dict(best_predictor_state)
    print(
        f"Restored best Stage 3 checkpoint from epoch {best_epoch} "
        f"(Val Acc: {best_val_acc:.4f}, Threshold: {best_val_threshold:.4f})"
    )

    return aligner


def get_train_node_ids(train_loader, max_nodes, seed):
    """Collect unique train nodes and optionally subsample them for cleaner plots."""
    node_ids = torch.unique(
        torch.cat([train_loader.dataset.sources, train_loader.dataset.targets], dim=0)
    )

    if max_nodes is None or len(node_ids) <= max_nodes:
        return node_ids

    generator = torch.Generator()
    generator.manual_seed(seed)
    keep = torch.randperm(len(node_ids), generator=generator)[:max_nodes]
    return node_ids[keep]


def collect_stage_points(anchor_dict, stage2_aligner, stage3_aligner, node_ids, device):
    """Collect anchor and node embeddings for Stage 1/2/3 plotting."""
    anchor_names = list(anchor_dict.keys())
    anchor_points = torch.stack([anchor_dict[name].detach().cpu() for name in anchor_names], dim=0).numpy()

    node_ids = node_ids.to(device)
    with torch.no_grad():
        stage2_nodes = stage2_aligner(node_ids).detach().cpu().numpy()
        stage3_nodes = stage3_aligner(node_ids).detach().cpu().numpy()

    return anchor_names, anchor_points, stage2_nodes, stage3_nodes


def build_semantic_basis(anchor_dict):
    """
    Build a fixed 2D semantic basis directly from the optimized Stage 1 anchors.

    x-axis:
        trust vs malicious polarity, using the direction P1 - P2

    y-axis:
        common-friend vs common-enemy balance, using the direction
        A_friend_1_friend_2 - A_enemy_1_enemy_2 and orthogonalizing it against x
    """
    p1 = anchor_dict["P1"].detach().cpu()
    p2 = anchor_dict["P2"].detach().cpu()
    friend_both = anchor_dict["A_friend_1_friend_2"].detach().cpu()
    enemy_both = anchor_dict["A_enemy_1_enemy_2"].detach().cpu()

    axis_x = p1 - p2
    axis_x = axis_x / axis_x.norm(p=2).clamp_min(1e-8)

    raw_y = friend_both - enemy_both
    raw_y = raw_y - torch.dot(raw_y, axis_x) * axis_x
    if raw_y.norm(p=2) < 1e-8:
        fallback = (
            anchor_dict["A_friend_1_enemy_2"].detach().cpu()
            - anchor_dict["A_enemy_1_friend_2"].detach().cpu()
        )
        raw_y = fallback - torch.dot(fallback, axis_x) * axis_x
    axis_y = raw_y / raw_y.norm(p=2).clamp_min(1e-8)

    return axis_x.numpy(), axis_y.numpy()


def project_to_semantic_plane(points, axis_x, axis_y):
    """Project hypersphere points onto the fixed semantic plane."""
    x_coords = points @ axis_x
    y_coords = points @ axis_y
    return np.column_stack([x_coords, y_coords])


def draw_panel(ax, title, anchor_coords, anchor_names, node_coords=None):
    """Draw one evolution panel with anchors in red and nodes in blue."""
    if node_coords is not None and len(node_coords) > 0:
        ax.scatter(
            node_coords[:, 0],
            node_coords[:, 1],
            c="royalblue",
            s=12,
            alpha=0.35,
            label="Train nodes",
        )

    ax.scatter(
        anchor_coords[:, 0],
        anchor_coords[:, 1],
        c="crimson",
        s=90,
        edgecolors="black",
        linewidths=0.4,
        label="Anchors",
        zorder=3,
    )

    for idx, name in enumerate(anchor_names):
        ax.annotate(
            name,
            (anchor_coords[idx, 0], anchor_coords[idx, 1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color="crimson",
            fontweight="bold",
        )

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_semantic_evolution(dataset, anchor_names, anchor_coords, stage2_coords, stage3_coords, output_path):
    """Render Stage 1/2/3 panels in one fixed semantic coordinate system."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    draw_panel(axes[0], "Stage 1: Anchor Geometry", anchor_coords, anchor_names)
    draw_panel(axes[1], "Stage 2: Contrastive Alignment", anchor_coords, anchor_names, stage2_coords)
    draw_panel(axes[2], "Stage 3: Supervised Refinement", anchor_coords, anchor_names, stage3_coords)

    # Keep all panels on the same visual scale for easier stage-to-stage comparison.
    all_coords = np.vstack([anchor_coords, stage2_coords, stage3_coords])
    x_min, y_min = all_coords.min(axis=0)
    x_max, y_max = all_coords.max(axis=0)
    x_pad = 0.05 * max(1e-6, x_max - x_min)
    y_pad = 0.05 * max(1e-6, y_max - y_min)

    for ax in axes:
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)

    fig.suptitle(f"BST Hypersphere Evolution in a Fixed Semantic Plane ({dataset})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot BST anchor/node evolution in a fixed semantic plane")
    parser.add_argument(
        "--dataset",
        type=str,
        default="otc",
        choices=["alpha", "otc", "epinions", "slashdot", "wiki-rfa", "wiki-elec"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--stage1-epochs", type=int, default=500)
    parser.add_argument("--stage2-epochs", type=int, default=10)
    parser.add_argument("--stage3-epochs", type=int, default=15)
    parser.add_argument("--max-nodes", type=int, default=1200)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frozen_anchors = train_stage1(args.stage1_epochs)
    train_loader, val_loader, _, num_nodes, heldout_targets_by_source, stage2_aligner, zero_positive = train_stage2(
        args.dataset,
        {name: tensor.detach().to(device) for name, tensor in frozen_anchors.items()},
        args.stage2_epochs,
        args.batch_size,
        args.seed,
        device,
    )
    stage2_state = snapshot_state_dict(stage2_aligner)
    stage3_aligner = train_stage3(
        stage2_aligner,
        train_loader,
        val_loader,
        num_nodes,
        heldout_targets_by_source,
        zero_positive,
        args.stage3_epochs,
        device,
    )
    stage2_aligner_for_plot = Stage2_NodeAligner(
        num_nodes=num_nodes, raw_embed_dim=128, hypersphere_dim=384
    ).to(device)
    stage2_aligner_for_plot.load_state_dict(stage2_state)

    node_ids = get_train_node_ids(train_loader, args.max_nodes, args.seed)
    anchor_names, anchor_points, stage2_nodes, stage3_nodes = collect_stage_points(
        frozen_anchors,
        stage2_aligner_for_plot,
        stage3_aligner,
        node_ids,
        device,
    )
    axis_x, axis_y = build_semantic_basis(frozen_anchors)
    stage1_coords = project_to_semantic_plane(anchor_points, axis_x, axis_y)
    stage2_coords = project_to_semantic_plane(stage2_nodes, axis_x, axis_y)
    stage3_coords = project_to_semantic_plane(stage3_nodes, axis_x, axis_y)

    output_path = args.output or os.path.join("results", f"semantic_plane_evolution_{args.dataset}.png")
    plot_semantic_evolution(
        args.dataset,
        anchor_names,
        stage1_coords,
        stage2_coords,
        stage3_coords,
        output_path,
    )
    print(f"Saved semantic-plane evolution plot to {output_path}")


if __name__ == "__main__":
    main()
