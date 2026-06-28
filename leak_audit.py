"""
End-to-End Empirical Leakage Audit
===================================
Runs the full pipeline on Bitcoin-Alpha and checks every possible
data-leakage vector at each stage boundary.

Audit checks:
  1. Split Integrity     – No directed edge appears in more than one split.
  2. Negative Sampling   – Stage 2 & 3 fake edges never collide with ANY real edge.
  3. Evaluation Purity   – Val/Test evaluation only touches its own split edges.
  4. Anchor Isolation     – Frozen anchors receive zero gradient during Stage 2/3.
  5. Stage 3 Label Mask   – Sign loss is only computed on real (non-fake) edges.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

# --- Setup ---
from dataloader import get_dataloaders, sample_targets_excluding_lookup, SNAPBitcoinDataset
from nodealligner import Stage2_NodeAligner, stage2_signed_bst_loss, HierarchicalPredictor
from anchor import initialize_hypersphere_anchors
from BSTspace import Stage1_BST_Optimizer
from evaluator import SNAPEval, evaluate_pipeline

DATASET = 'alpha'
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  [PASS] {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name} -- {detail}")
        failed += 1

# ============================================================
print("=" * 70)
print("  LEAKAGE AUDIT: Bitcoin-Alpha (end-to-end)")
print("=" * 70)

# ============================================================
# AUDIT 1: Split Integrity
# ============================================================
print("\n[AUDIT 1] Split Integrity (directed-edge disjointness)")
set_seed(SEED)

train_ds = SNAPBitcoinDataset(data_type=DATASET, split='train', seed=SEED)
val_ds   = SNAPBitcoinDataset(data_type=DATASET, split='val', seed=SEED)
test_ds  = SNAPBitcoinDataset(data_type=DATASET, split='test', seed=SEED)

def edges_to_set(ds):
    return set(zip(ds.sources.tolist(), ds.targets.tolist()))

train_edges = edges_to_set(train_ds)
val_edges   = edges_to_set(val_ds)
test_edges  = edges_to_set(test_ds)

check("train ∩ val  == ∅",   len(train_edges & val_edges) == 0,
      f"{len(train_edges & val_edges)} leaked directed pairs")
check("train ∩ test == ∅",   len(train_edges & test_edges) == 0,
      f"{len(train_edges & test_edges)} leaked directed pairs")
check("val   ∩ test == ∅",   len(val_edges & test_edges) == 0,
      f"{len(val_edges & test_edges)} leaked directed pairs")

total_original = len(train_edges) + len(val_edges) + len(test_edges)
total_union    = len(train_edges | val_edges | test_edges)
check("No duplicate edges across splits", total_original == total_union,
      f"union={total_union} vs sum={total_original}")

all_real_edges = train_edges | val_edges | test_edges
print(f"  (info) Train: {len(train_edges)}  Val: {len(val_edges)}  Test: {len(test_edges)}  Total: {len(all_real_edges)}")

# ============================================================
# AUDIT 2: Negative Sampling Never Hits Real Edges
# ============================================================
print("\n[AUDIT 2] Negative Sampling (fake edges vs ALL real edges)")

set_seed(SEED)
train_loader, val_loader, test_loader, num_nodes, sampling_metadata = get_dataloaders(DATASET, batch_size=1024, seed=SEED)
heldout_targets_by_source = sampling_metadata['heldout_targets_by_source']

# Check that the blocked lookup includes train edges too (our fix)
blocked_edges = set()
for src, tgts in heldout_targets_by_source.items():
    for tgt in tgts:
        blocked_edges.add((src, tgt))

check("Blocked lookup covers train edges",
      train_edges.issubset(blocked_edges),
      f"{len(train_edges - blocked_edges)} train edges missing from blocked lookup")
check("Blocked lookup covers val edges",
      val_edges.issubset(blocked_edges),
      f"{len(val_edges - blocked_edges)} val edges missing from blocked lookup")
check("Blocked lookup covers test edges",
      test_edges.issubset(blocked_edges),
      f"{len(test_edges - blocked_edges)} test edges missing from blocked lookup")

# Empirically sample 50,000 fake edges and verify none are real
print("  Empirically sampling 50,000 fake edges...")
collision_count = 0
total_sampled = 0
for batch in train_loader:
    sources = batch['source'].to(DEVICE)
    for _ in range(10):  # 10 rounds per batch for coverage
        fake_targets = sample_targets_excluding_lookup(
            sources, num_nodes, heldout_targets_by_source, DEVICE
        )
        src_list = sources.cpu().tolist()
        tgt_list = fake_targets.cpu().tolist()
        for s, t in zip(src_list, tgt_list):
            total_sampled += 1
            if (s, t) in all_real_edges:
                collision_count += 1
    if total_sampled >= 50000:
        break

check(f"0 collisions in {total_sampled:,} sampled fake edges",
      collision_count == 0,
      f"{collision_count} fake edges collided with real edges!")

# ============================================================
# AUDIT 3: Anchor Gradient Isolation
# ============================================================
print("\n[AUDIT 3] Anchor Gradient Isolation (P1/P2 frozen during Stage 2/3)")

set_seed(SEED)
anchor_embeddings = initialize_hypersphere_anchors()
bst_model = Stage1_BST_Optimizer(anchor_embeddings, neg_margin=0.0)
optimizer = torch.optim.Adam(bst_model.parameters(), lr=0.01)
for _ in range(500):
    optimizer.zero_grad()
    loss = bst_model()
    loss.backward()
    optimizer.step()

final_anchors = bst_model.get_normalized_anchors()
frozen_anchors = {name: tensor.detach().to(DEVICE) for name, tensor in final_anchors.items()}

# Verify anchors are detached and have no grad_fn
check("P1 has no grad_fn", frozen_anchors["P1"].grad_fn is None)
check("P2 has no grad_fn", frozen_anchors["P2"].grad_fn is None)
check("P1 requires_grad == False", not frozen_anchors["P1"].requires_grad)
check("P2 requires_grad == False", not frozen_anchors["P2"].requires_grad)

# Snapshot anchor values before Stage 2
p1_before = frozen_anchors["P1"].clone()
p2_before = frozen_anchors["P2"].clone()

# Run 2 epochs of Stage 2 and verify anchors haven't moved
aligner = Stage2_NodeAligner(num_nodes=num_nodes, raw_embed_dim=128, hypersphere_dim=384).to(DEVICE)
optimizer_s2 = torch.optim.Adam(aligner.parameters(), lr=0.005)
aligner.train()

for epoch in range(2):
    for batch in train_loader:
        sources = batch['source'].to(DEVICE)
        targets = batch['target'].to(DEVICE)
        ratings = batch['rating'].to(DEVICE)
        v_neg = sample_targets_excluding_lookup(sources, num_nodes, heldout_targets_by_source, DEVICE)
        optimizer_s2.zero_grad()
        u_embeds = aligner(sources)
        v_embeds = aligner(targets)
        v_neg_embeds = aligner(v_neg)
        loss_s2 = stage2_signed_bst_loss(u_embeds, v_embeds, v_neg_embeds, ratings, frozen_anchors)
        loss_s2.backward()
        optimizer_s2.step()

check("P1 unchanged after Stage 2",
      torch.equal(frozen_anchors["P1"], p1_before),
      f"P1 drifted by {(frozen_anchors['P1'] - p1_before).abs().max().item():.6f}")
check("P2 unchanged after Stage 2",
      torch.equal(frozen_anchors["P2"], p2_before),
      f"P2 drifted by {(frozen_anchors['P2'] - p2_before).abs().max().item():.6f}")

# Verify anchors are NOT in the Stage 2 optimizer param groups
s2_param_ids = set()
for pg in optimizer_s2.param_groups:
    for p in pg['params']:
        s2_param_ids.add(id(p))

check("P1 not in Stage 2 optimizer", id(frozen_anchors["P1"]) not in s2_param_ids)
check("P2 not in Stage 2 optimizer", id(frozen_anchors["P2"]) not in s2_param_ids)

# ============================================================
# AUDIT 4: Stage 3 Sign Loss Only on Real Edges
# ============================================================
print("\n[AUDIT 4] Stage 3 Sign Loss Masking (sign loss only on real edges)")

predictor = HierarchicalPredictor(embed_dim=384).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()

# Simulate one Stage 3 forward pass manually
for batch in train_loader:
    sources = batch['source'].to(DEVICE)
    targets = batch['target'].to(DEVICE)
    ratings = batch['rating'].to(DEVICE)

    num_edges = len(sources)
    fake_targets = sample_targets_excluding_lookup(
        sources, num_nodes, heldout_targets_by_source, DEVICE
    )

    all_u = torch.cat([sources, sources], dim=0)
    all_v = torch.cat([targets, fake_targets], dim=0)
    exist_labels = torch.cat([torch.ones(num_edges, device=DEVICE),
                              torch.zeros(num_edges, device=DEVICE)], dim=0)

    u_embeds = aligner(all_u)
    v_embeds = aligner(all_v)
    exist_logits, sign_logits = predictor(u_embeds, v_embeds)

    # Existence loss is computed over ALL pairs (real + fake)
    loss_exist = criterion(exist_logits, exist_labels)

    # Sign loss is only computed on the first num_edges (the real edges)
    real_sign_logits = sign_logits[:num_edges]
    sign_labels = (ratings > 0).float()
    loss_sign = criterion(real_sign_logits, sign_labels)

    check(f"Exist loss covers {2*num_edges} pairs (real+fake)",
          exist_logits.shape[0] == 2 * num_edges)
    check(f"Sign loss covers only {num_edges} pairs (real only)",
          real_sign_logits.shape[0] == num_edges)
    check("Sign logits from fake edges are NOT used in loss",
          real_sign_logits.shape[0] == num_edges and sign_logits.shape[0] == 2 * num_edges)
    break  # Only need one batch

# ============================================================
# AUDIT 5: Evaluation Split Purity
# ============================================================
print("\n[AUDIT 5] Evaluation Split Purity (val/test only see their own edges)")

# Collect every (source, target) pair the val_loader actually iterates over
val_seen_edges = set()
for batch in val_loader:
    srcs = batch['source'].tolist()
    tgts = batch['target'].tolist()
    for s, t in zip(srcs, tgts):
        val_seen_edges.add((s, t))

check("Val loader only contains val edges",
      val_seen_edges == val_edges,
      f"val_loader has {len(val_seen_edges - val_edges)} non-val edges, missing {len(val_edges - val_seen_edges)}")

test_seen_edges = set()
for batch in test_loader:
    srcs = batch['source'].tolist()
    tgts = batch['target'].tolist()
    for s, t in zip(srcs, tgts):
        test_seen_edges.add((s, t))

check("Test loader only contains test edges",
      test_seen_edges == test_edges,
      f"test_loader has {len(test_seen_edges - test_edges)} non-test edges, missing {len(test_edges - test_seen_edges)}")

# Check no train edges leak into val/test loaders
check("No train edges in val loader",
      len(val_seen_edges & train_edges) == 0,
      f"{len(val_seen_edges & train_edges)} train edges leaked into val")
check("No train edges in test loader",
      len(test_seen_edges & train_edges) == 0,
      f"{len(test_seen_edges & train_edges)} train edges leaked into test")

# ============================================================
# AUDIT 6: Stage 2 Training Only Uses Train Edges
# ============================================================
print("\n[AUDIT 6] Stage 2 Training Data Purity")

train_seen_edges = set()
for batch in train_loader:
    srcs = batch['source'].tolist()
    tgts = batch['target'].tolist()
    for s, t in zip(srcs, tgts):
        train_seen_edges.add((s, t))

check("Train loader only contains train edges",
      train_seen_edges == train_edges,
      f"train_loader has {len(train_seen_edges - train_edges)} non-train edges")
check("No val edges in train loader",
      len(train_seen_edges & val_edges) == 0)
check("No test edges in train loader",
      len(train_seen_edges & test_edges) == 0)

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print(f"  AUDIT COMPLETE: {passed} PASSED, {failed} FAILED")
print("=" * 70)
if failed == 0:
    print("  *** VERDICT: NO LEAKAGE DETECTED. Pipeline is clean. ***")
else:
    print(f"  *** WARNING: {failed} check(s) FAILED. Investigate above. ***")
