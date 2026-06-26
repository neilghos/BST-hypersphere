import os
import random
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


AMBIGUITY_EPS = 0.05


def _safe_mean(values):
    return sum(values) / len(values) if values else 0.0


def _safe_dataset_name(ds_name):
    return str(ds_name).replace("/", "_").replace("\\", "_").replace(" ", "_")


def _alignment_display(alignment):
    if alignment == "Trust":
        return "Group Pole 1 (P1)"
    if alignment == "Malicious":
        return "Group Pole 2 (P2)"
    return "Boundary / Ambiguous"


def _region_display(alignment):
    if alignment == "Trust":
        return "Pole 1 attractor region"
    if alignment == "Malicious":
        return "Pole 2 attractor region"
    return "Boundary region"


def get_node_profile(node_id, embed, anchors, ambiguity_eps=AMBIGUITY_EPS):
    sim_p1 = F.cosine_similarity(embed, anchors["P1"], dim=0).item()
    sim_p2 = F.cosine_similarity(embed, anchors["P2"], dim=0).item()

    if abs(sim_p1 - sim_p2) < ambiguity_eps:
        alignment = "Ambiguous"
    else:
        alignment = "Trust" if sim_p1 > sim_p2 else "Malicious"

    profile = (
        f"Node {node_id}: P1={sim_p1:+.3f}, "
        f"P2={sim_p2:+.3f}, aligned={_alignment_display(alignment)}"
    )
    return profile, alignment


def _binary_sbt_heuristic(u_align, v_align):
    if u_align == "Ambiguous" or v_align == "Ambiguous":
        return None
    return 1 if u_align == v_align else 0


def _format_edge_case(
    title,
    u_idx,
    v_idx,
    u_embed,
    v_embed,
    anchors,
    sign_prob,
    true_label,
    pred_label,
):
    u_profile, u_align = get_node_profile(u_idx, u_embed, anchors)
    v_profile, v_align = get_node_profile(v_idx, v_embed, anchors)

    true_str = "positive" if true_label == 1 else "negative"
    pred_str = "positive" if pred_label == 1 else "negative"
    sbt_expected = _binary_sbt_heuristic(u_align, v_align)
    if sbt_expected is None:
        sbt_expected_str = "underdetermined"
    else:
        sbt_expected_str = "positive" if sbt_expected == 1 else "negative"

    return [
        title,
        f"  edge: {u_idx} -> {v_idx}",
        f"  ground truth: {true_str}",
        f"  prediction: {pred_str}",
        f"  binary sbt heuristic: {sbt_expected_str}",
        f"  sign prob: {sign_prob:.4f}",
        f"  source profile: {u_profile}",
        f"  target profile: {v_profile}",
    ]


def create_bst_consistency_heatmap(
    true_labels,
    pred_labels,
    heuristic_consistency,
    total_edges,
    ds_name,
    output_dir="results",
):
    if not true_labels:
        return None

    true_labels = np.asarray(true_labels, dtype=np.int32)
    pred_labels = np.asarray(pred_labels, dtype=np.int32)
    heuristic_consistency = np.asarray(heuristic_consistency, dtype=np.int32)

    accuracy_matrix = np.full((2, 2), np.nan, dtype=np.float32)
    count_matrix = np.zeros((2, 2), dtype=np.int32)
    share_matrix = np.zeros((2, 2), dtype=np.float32)

    for row_idx, label_value in enumerate([1, 0]):
        for col_idx, is_consistent in enumerate([1, 0]):
            mask = (true_labels == label_value) & (heuristic_consistency == is_consistent)
            count = int(mask.sum())
            count_matrix[row_idx, col_idx] = count
            share_matrix[row_idx, col_idx] = count / max(1, total_edges) * 100.0
            if count > 0:
                accuracy_matrix[row_idx, col_idx] = (
                    (pred_labels[mask] == true_labels[mask]).mean() * 100.0
                )

    fig, ax = plt.subplots(figsize=(8.4, 6.6), constrained_layout=True)
    heatmap = ax.imshow(
        np.nan_to_num(accuracy_matrix, nan=0.0),
        cmap="YlGnBu",
        vmin=0.0,
        vmax=100.0,
    )
    cbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Model accuracy within subset (%)", fontsize=11)

    ax.set_xticks([0, 1], labels=["BST-consistent", "BST-inconsistent"])
    ax.set_yticks([0, 1], labels=["Ground truth: positive", "Ground truth: negative"])
    ax.set_title(
        f"Pole-Induced Triad Consistency on Held-Out Test Edges ({ds_name})",
        fontsize=15,
        pad=12,
    )

    for row_idx in range(2):
        for col_idx in range(2):
            count = count_matrix[row_idx, col_idx]
            share = share_matrix[row_idx, col_idx]
            accuracy = accuracy_matrix[row_idx, col_idx]
            if count == 0:
                cell_text = "No edges"
                text_color = "#1f1f1f"
                fontweight = "normal"
            else:
                cell_text = (
                    f"{share:.1f}% of test edges\n"
                    f"n = {count}\n"
                    f"model acc = {accuracy:.1f}%"
                )
                text_color = "white" if accuracy >= 62.0 else "#1f1f1f"
                fontweight = "semibold"
            ax.text(
                col_idx,
                row_idx,
                cell_text,
                ha="center",
                va="center",
                fontsize=11,
                color=text_color,
                fontweight=fontweight,
            )

    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    excluded_edges = total_edges - len(true_labels)
    excluded_share = excluded_edges / max(1, total_edges) * 100.0
    ax.text(
        0.5,
        -0.12,
        (
            "Cells use heuristic-covered edges only. "
            f"Ambiguous boundary cases excluded: {excluded_edges} edges ({excluded_share:.1f}%)."
        ),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        color="#444444",
    )

    os.makedirs(output_dir, exist_ok=True)
    safe_name = _safe_dataset_name(ds_name)
    plot_path = os.path.join(
        output_dir,
        f"dataset_{safe_name}_bst_consistency_heatmap.png",
    )
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def build_adjacency(dataloader):
    """Build a unique outgoing adjacency lookup from the evaluated split."""
    adjacency = defaultdict(set)
    for batch in dataloader:
        sources = batch["source"].cpu().tolist()
        targets = batch["target"].cpu().tolist()
        for source, target in zip(sources, targets):
            adjacency[int(source)].add(int(target))
    return {node: sorted(neighbors) for node, neighbors in adjacency.items()}


def analyze_echo_chamber(
    node_id,
    dataloader,
    aligner,
    anchors,
    device,
    mild_homo_thresh,
    extreme_homo_thresh,
    density_thresh,
    zero_positive=False,
):
    aligner.eval()

    pos_targets = set()
    neg_targets = set()

    for batch in dataloader:
        sources = batch["source"]
        targets = batch["target"]
        ratings = batch["rating"]

        mask = sources == node_id
        if not mask.any():
            continue

        masked_targets = targets[mask]
        masked_ratings = ratings[mask]
        for target_node, rating in zip(masked_targets.tolist(), masked_ratings.tolist()):
            is_pos = rating >= 0 if zero_positive else rating > 0
            if is_pos:
                pos_targets.add(int(target_node))
            else:
                neg_targets.add(int(target_node))

    if not pos_targets and not neg_targets:
        return None

    pos_targets_t = (
        torch.tensor(sorted(pos_targets), dtype=torch.long, device=device)
        if pos_targets
        else torch.empty(0, dtype=torch.long, device=device)
    )
    neg_targets_t = (
        torch.tensor(sorted(neg_targets), dtype=torch.long, device=device)
        if neg_targets
        else torch.empty(0, dtype=torch.long, device=device)
    )

    with torch.no_grad():
        u_embed = aligner(torch.tensor([node_id], dtype=torch.long, device=device))[0]
        _, node_align = get_node_profile(node_id, u_embed, anchors)

        bubble_alignment = 0
        bubble_density = 0.0
        if pos_targets_t.numel() > 0:
            pos_embeds = aligner(pos_targets_t)
            sim_pos_to_p1 = F.cosine_similarity(
                pos_embeds, anchors["P1"].unsqueeze(0), dim=-1
            )
            sim_pos_to_p2 = F.cosine_similarity(
                pos_embeds, anchors["P2"].unsqueeze(0), dim=-1
            )

            if node_align == "Trust":
                bubble_alignment = (sim_pos_to_p1 > sim_pos_to_p2).sum().item()
            elif node_align == "Malicious":
                bubble_alignment = (sim_pos_to_p2 > sim_pos_to_p1).sum().item()

            bubble_density = F.cosine_similarity(
                u_embed.unsqueeze(0), pos_embeds, dim=-1
            ).mean().item()

        enemy_alignment = 0
        if neg_targets_t.numel() > 0:
            neg_embeds = aligner(neg_targets_t)
            sim_neg_to_p1 = F.cosine_similarity(
                neg_embeds, anchors["P1"].unsqueeze(0), dim=-1
            )
            sim_neg_to_p2 = F.cosine_similarity(
                neg_embeds, anchors["P2"].unsqueeze(0), dim=-1
            )

            if node_align == "Trust":
                enemy_alignment = (sim_neg_to_p2 > sim_neg_to_p1).sum().item()
            elif node_align == "Malicious":
                enemy_alignment = (sim_neg_to_p1 > sim_neg_to_p2).sum().item()

    homophily_ratio = (
        bubble_alignment / len(pos_targets) * 100.0 if pos_targets else 0.0
    )
    heterophily_ratio = (
        enemy_alignment / len(neg_targets) * 100.0 if neg_targets else 0.0
    )

    total_edges = len(pos_targets) + len(neg_targets)
    if node_align == "Ambiguous":
        chamber_status = "Boundary / Mixed Alignment"
    elif total_edges < 5:
        chamber_status = "Sparse Local Neighborhood"
    elif homophily_ratio >= extreme_homo_thresh and bubble_density >= density_thresh:
        chamber_status = "Extreme Echo Chamber"
    elif homophily_ratio >= mild_homo_thresh:
        chamber_status = "Mild Echo Chamber"
    else:
        chamber_status = "Mixed / Heterophilous Neighborhood"

    return {
        "node_id": node_id,
        "alignment": node_align,
        "friends": len(pos_targets),
        "enemies": len(neg_targets),
        "in_group_homo": homophily_ratio,
        "out_group_hostility": heterophily_ratio,
        "spatial_density": bubble_density,
        "chamber_status": chamber_status,
    }


def hyperspherical_random_walk(
    node_id,
    adjacency,
    aligner,
    anchors,
    device,
    mild_homo_thresh,
    extreme_homo_thresh,
    sim_threshold=0.0,
    max_steps=5,
    num_walks=100,
):
    aligner.eval()

    with torch.no_grad():
        start_embed = aligner(torch.tensor([node_id], dtype=torch.long, device=device))[0]
        _, start_align = get_node_profile(node_id, start_embed, anchors)
        start_basin = _region_display(start_align)

        walk_lengths = []
        basin_purity_scores = []
        dead_ends = 0

        for _ in range(num_walks):
            current_node = node_id
            path_length = 0
            basin_matches = 0

            for _step in range(max_steps):
                neighbors = adjacency.get(current_node, [])
                if not neighbors:
                    dead_ends += 1
                    break

                current_embed = aligner(
                    torch.tensor([current_node], dtype=torch.long, device=device)
                )[0]
                neighbor_tensor = torch.tensor(
                    neighbors, dtype=torch.long, device=device
                )
                neighbor_embeds = aligner(neighbor_tensor)

                sims = F.cosine_similarity(
                    current_embed.unsqueeze(0), neighbor_embeds, dim=-1
                )
                valid_indices = (sims >= sim_threshold).nonzero(as_tuple=True)[0]
                if valid_indices.numel() == 0:
                    dead_ends += 1
                    break

                next_idx = random.choice(valid_indices.tolist())
                next_node = neighbors[next_idx]
                next_embed = neighbor_embeds[next_idx]
                _, next_align = get_node_profile(next_node, next_embed, anchors)

                if start_align != "Ambiguous" and next_align == start_align:
                    basin_matches += 1

                current_node = next_node
                path_length += 1

            walk_lengths.append(path_length)
            if path_length > 0:
                basin_purity_scores.append(basin_matches / path_length)

    avg_path_length = _safe_mean(walk_lengths)
    avg_basin_purity = _safe_mean(basin_purity_scores) * 100.0
    survival_rate = (
        sum(length == max_steps for length in walk_lengths) / max(1, len(walk_lengths)) * 100.0
    )
    out_degree = len(adjacency.get(node_id, []))

    if out_degree == 0:
        walk_status = "Topological Dead End"
    elif avg_path_length < max_steps * 0.4:
        walk_status = "Geometric Bottleneck"
    elif avg_basin_purity >= extreme_homo_thresh:
        walk_status = "Strong Pole Retention"
    elif avg_basin_purity >= mild_homo_thresh:
        walk_status = "Moderate Pole Retention"
    else:
        walk_status = "Cross-Pole Drift"

    return {
        "node_id": node_id,
        "start_basin": start_basin,
        "out_degree": out_degree,
        "dead_ends": dead_ends,
        "avg_path_length": avg_path_length,
        "avg_basin_purity": avg_basin_purity,
        "survival_rate": survival_rate,
        "transition_threshold": sim_threshold,
        "walk_status": walk_status,
    }


def run_interpretability_module(
    aligner,
    predictor,
    test_loader,
    device,
    anchors,
    threshold,
    zero_positive=False,
    ds_name="dataset",
):
    aligner.eval()
    predictor.eval()

    total_edges = 0
    heuristic_covered_edges = 0
    heuristic_gt_agree = 0
    heuristic_model_agree = 0
    heuristic_gt_correct_total = 0
    heuristic_model_agree_when_gt_correct = 0
    heuristic_inconsistent_correct_total = 0
    global_homophily_edges = 0
    global_heterophily_edges = 0
    global_sim_sum = 0.0
    global_sim_count = 0
    global_pos_sim_sum = 0.0
    global_pos_sim_count = 0

    node_stats = defaultdict(lambda: {"total": 0})
    exemplar_reports = []
    captured = {"tp": False, "tn": False, "fp": False, "fn": False}
    heuristic_true_labels = []
    heuristic_pred_labels = []
    heuristic_consistency_flags = []

    with torch.no_grad():
        for batch in test_loader:
            sources = batch["source"].to(device)
            targets = batch["target"].to(device)
            ratings = batch["rating"].to(device)

            u_embeds = aligner(sources)
            v_embeds = aligner(targets)
            _, sign_logits = predictor(u_embeds, v_embeds)
            sign_probs = torch.sigmoid(sign_logits)

            sims = F.cosine_similarity(u_embeds, v_embeds, dim=-1)
            global_sim_sum += sims.sum().item()
            global_sim_count += sims.numel()

            sign_labels = (ratings >= 0).int() if zero_positive else (ratings > 0).int()
            preds = (sign_logits >= threshold).int()

            pos_mask = sign_labels == 1
            if pos_mask.any():
                pos_sims = sims[pos_mask]
                global_pos_sim_sum += pos_sims.sum().item()
                global_pos_sim_count += pos_sims.numel()

            for idx in range(len(sources)):
                u = sources[idx].item()
                v = targets[idx].item()
                y_true = sign_labels[idx].item()
                y_pred = preds[idx].item()

                _, u_align = get_node_profile(u, u_embeds[idx], anchors)
                _, v_align = get_node_profile(v, v_embeds[idx], anchors)
                heuristic_label = _binary_sbt_heuristic(u_align, v_align)

                total_edges += 1
                node_stats[u]["total"] += 1

                if heuristic_label is not None:
                    heuristic_covered_edges += 1
                    heuristic_true_labels.append(y_true)
                    heuristic_pred_labels.append(y_pred)
                    heuristic_consistency_flags.append(int(heuristic_label == y_true))
                    if heuristic_label == y_true:
                        heuristic_gt_agree += 1
                        heuristic_gt_correct_total += 1
                        if y_pred == heuristic_label:
                            heuristic_model_agree_when_gt_correct += 1
                    elif y_pred == y_true:
                        heuristic_inconsistent_correct_total += 1
                    if heuristic_label == y_pred:
                        heuristic_model_agree += 1
                    if u_align == v_align:
                        global_homophily_edges += 1
                    else:
                        global_heterophily_edges += 1

                case_key = None
                case_title = None
                if y_true == 1 and y_pred == 1 and not captured["tp"]:
                    case_key = "tp"
                    case_title = "True Positive Example"
                elif y_true == 0 and y_pred == 0 and not captured["tn"]:
                    case_key = "tn"
                    case_title = "True Negative Example"
                elif y_true == 0 and y_pred == 1 and not captured["fp"]:
                    case_key = "fp"
                    case_title = "False Positive Example"
                elif y_true == 1 and y_pred == 0 and not captured["fn"]:
                    case_key = "fn"
                    case_title = "False Negative Example"

                if case_key is not None:
                    captured[case_key] = True
                    exemplar_reports.append(
                        _format_edge_case(
                            case_title,
                            u,
                            v,
                            u_embeds[idx],
                            v_embeds[idx],
                            anchors,
                            sign_probs[idx].item(),
                            y_true,
                            y_pred,
                        )
                    )

    global_homophily = (
        global_homophily_edges / max(1, heuristic_covered_edges) * 100.0
    )
    global_heterophily = (
        global_heterophily_edges / max(1, heuristic_covered_edges) * 100.0
    )
    global_transition_cost = global_sim_sum / max(1, global_sim_count)
    global_density = global_pos_sim_sum / max(1, global_pos_sim_count)

    mild_homo_thresh = global_homophily
    extreme_homo_thresh = min(95.0, global_homophily + 20.0)
    walk_threshold = max(-1.0, min(1.0, global_transition_cost - 0.10))

    adjacency = build_adjacency(test_loader)
    top_nodes = sorted(
        node_stats.keys(),
        key=lambda node_id: node_stats[node_id]["total"],
        reverse=True,
    )[:5]

    echo_results = []
    walk_results = []
    for node_id in top_nodes:
        echo_result = analyze_echo_chamber(
            node_id,
            test_loader,
            aligner,
            anchors,
            device,
            mild_homo_thresh,
            extreme_homo_thresh,
            global_density,
            zero_positive=zero_positive,
        )
        if echo_result is not None:
            echo_results.append(echo_result)

        walk_result = hyperspherical_random_walk(
            node_id,
            adjacency,
            aligner,
            anchors,
            device,
            mild_homo_thresh,
            extreme_homo_thresh,
            sim_threshold=walk_threshold,
        )
        if walk_result is not None:
            walk_results.append(walk_result)

    bst_plot_path = create_bst_consistency_heatmap(
        heuristic_true_labels,
        heuristic_pred_labels,
        heuristic_consistency_flags,
        total_edges,
        ds_name,
        output_dir="results",
    )

    lines = []
    lines.append(f"XAI Report: {ds_name}")
    lines.append("")

    lines.append("Anchor-Based SBT Heuristic Analysis")
    lines.append("  note: all values below are computed on the held-out test split only")
    lines.append("  note: the heuristic uses only the P1/P2 polarity frame")
    lines.append("  note: P1 and P2 are latent polarity poles, not observed node roles")
    lines.append(f"  total test edges evaluated: {total_edges}")
    lines.append(f"  heuristic-covered test edges: {heuristic_covered_edges}")
    lines.append(
        f"  binary heuristic agreement with ground truth: "
        f"{heuristic_gt_agree / max(1, heuristic_covered_edges) * 100:.2f}%"
    )
    lines.append(
        f"  model agreement with binary heuristic: "
        f"{heuristic_model_agree / max(1, heuristic_covered_edges) * 100:.2f}%"
    )
    if heuristic_gt_correct_total > 0:
        consistent_accuracy = (
            heuristic_model_agree_when_gt_correct / heuristic_gt_correct_total * 100.0
        )
        lines.append(
            f"  model accuracy on heuristic-consistent edges: {consistent_accuracy:.2f}%"
        )
    else:
        lines.append("  model accuracy on heuristic-consistent edges: N/A")
    heuristic_inconsistent_total = heuristic_covered_edges - heuristic_gt_correct_total
    if heuristic_inconsistent_total > 0:
        inconsistent_accuracy = (
            heuristic_inconsistent_correct_total / heuristic_inconsistent_total * 100.0
        )
        lines.append(
            f"  model accuracy on heuristic-inconsistent edges: {inconsistent_accuracy:.2f}%"
        )
    else:
        lines.append("  model accuracy on heuristic-inconsistent edges: N/A")
    lines.append(f"  test-split homophily rate: {global_homophily:.2f}%")
    lines.append(f"  test-split heterophily rate: {global_heterophily:.2f}%")
    lines.append(f"  mean test-edge cosine similarity: {global_transition_cost:+.4f}")
    lines.append(f"  mean positive test-edge cosine similarity: {global_density:+.4f}")
    if bst_plot_path is not None:
        lines.append(f"  bst consistency heatmap: {bst_plot_path}")
    lines.append("")

    if exemplar_reports:
        lines.append("  Example Edges")
        lines.append("  note: these are first-seen examples in dataloader order")
        for report in exemplar_reports:
            for line in report:
                lines.append(f"  {line}")
            lines.append("")

    lines.append("Local Echo Chamber Heuristic")
    lines.append("  note: node-level labels below are heuristic and induced from the test split")
    lines.append(f"  mild homophily threshold: {mild_homo_thresh:.2f}%")
    lines.append(f"  extreme homophily threshold: {extreme_homo_thresh:.2f}%")
    lines.append(f"  density threshold: {global_density:+.4f}")
    lines.append("")

    if echo_results:
        for result in echo_results:
            lines.append(f"  Node {result['node_id']}")
            lines.append(f"    alignment: {_alignment_display(result['alignment'])}")
            lines.append(f"    positive outgoing targets: {result['friends']}")
            lines.append(f"    negative outgoing targets: {result['enemies']}")
            lines.append(f"    in-group homophily: {result['in_group_homo']:.2f}%")
            lines.append(f"    out-group hostility: {result['out_group_hostility']:.2f}%")
            lines.append(f"    spatial density: {result['spatial_density']:+.4f}")
            lines.append(f"    heuristic status: {result['chamber_status']}")
            lines.append("")
    else:
        lines.append("  no eligible nodes found")
        lines.append("")

    lines.append("Geometric Random Walk Heuristic")
    lines.append("  note: walks run on the held-out test adjacency only")
    lines.append(f"  max steps per walk: 5")
    lines.append(f"  walks per node: 100")
    lines.append(f"  transition threshold: {walk_threshold:+.4f}")
    lines.append("")

    if walk_results:
        for result in walk_results:
            lines.append(f"  Node {result['node_id']}")
            lines.append(f"    initial anchor region: {result['start_basin']}")
            lines.append(f"    out-degree: {result['out_degree']}")
            lines.append(f"    average path length: {result['avg_path_length']:.2f}")
            lines.append(f"    dead ends: {result['dead_ends']}")
            lines.append(f"    within-pole trajectory purity: {result['avg_basin_purity']:.2f}%")
            lines.append(f"    survival rate: {result['survival_rate']:.2f}%")
            lines.append(f"    heuristic status: {result['walk_status']}")
            lines.append("")
    else:
        lines.append("  no eligible nodes found")
        lines.append("")

    os.makedirs("results", exist_ok=True)
    safe_name = _safe_dataset_name(ds_name)
    txt_path = os.path.join("results", f"xai_{safe_name}.txt")
    with open(txt_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")

    return txt_path
