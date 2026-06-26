import torch
import torch.nn.functional as F

def get_node_profile(node_id, embed, anchors):
    # Calculate cosine similarity with the semantic anchors
    sim_P1 = F.cosine_similarity(embed, anchors["P1"], dim=0).item()
    sim_P2 = F.cosine_similarity(embed, anchors["P2"], dim=0).item()
    
    # Determine the primary alignment
    if sim_P1 > sim_P2:
        alignment = "Trust"
    else:
        alignment = "Malicious"
        
    prof_str = f"P1 (Trust) Sim: {sim_P1:+.2f} | P2 (Malicious) Sim: {sim_P2:+.2f} --> Aligned: {alignment}"
    return prof_str, alignment

def print_explanation(u_idx, v_idx, u_embed, v_embed, anchors, exist_prob, sign_prob, true_label, pred_label):
    u_prof_str, u_align = get_node_profile(u_idx, u_embed, anchors)
    v_prof_str, v_align = get_node_profile(v_idx, v_embed, anchors)
    
    true_str = "POSITIVE" if true_label > 0 else "NEGATIVE"
    pred_str = "POSITIVE" if pred_label > 0 else "NEGATIVE"
    
    # Deduce expected sign from SBT
    sbt_expected = "POSITIVE" if u_align == v_align else "NEGATIVE"
    matched_sbt = "YES" if pred_str == sbt_expected else "NO"
    
    print("=" * 70)
    print(f"EDGE XAI REPORT: Node {u_idx} -> Node {v_idx}")
    print(f"Ground Truth: {true_str} | Model Predicted: {pred_str}")
    print("=" * 70)
    print("STEP 1: Anchor Geometry Extraction")
    print(f"   - Source Node {u_idx}: {u_prof_str}")
    print(f"   - Target Node {v_idx}: {v_prof_str}")
    print("")
    print("STEP 2: Structural Balance Theory (SBT) Deduction")
    if sbt_expected == "POSITIVE":
        print(f"   - Both nodes align with {u_align}. According to SBT (homophily), this edge should be POSITIVE.")
    else:
        print(f"   - Mismatch: {u_align} vs {v_align}. According to SBT (heterophily), this edge should be NEGATIVE.")
    print("")
    print("STEP 3: Model Output & Alignment")
    print(f"   - Hierarchical MLP Exist Probability: {exist_prob*100:.1f}%")
    print(f"   - Hierarchical MLP Sign Prediction: {pred_str} ({sign_prob*100:.1f}% prob)")
    print(f"   - Did the MLP follow classical SBT logic? {matched_sbt}")
    
    if matched_sbt == "YES":
        if pred_str == true_str:
            print("   -> CONCLUSION: The MLP correctly leveraged the hypersphere geometry and SBT rules.")
        else:
            print("   -> CONCLUSION: The MLP strictly followed SBT geometry, but the ground truth label violates classical balance.")
    else:
        print("   -> CONCLUSION: The MLP overrode strict SBT rules based on complex latent patterns.")
        
    print("=" * 70 + "\n")

def run_interpretability_module(aligner, predictor, test_loader, device, anchors, threshold, zero_positive=False):
    print("\n\n" + "="*80)
    print("                 ZERO-SHOT SEMANTIC INTERPRETABILITY MODULE")
    print("="*80)
    
    aligner.eval()
    predictor.eval()
    
    # We want to find exactly 1 of each edge case to explain
    found_tp, found_tn, found_fp, found_fn = False, False, False, False
    
    total_edges = 0
    sbt_correct = 0
    model_sbt_agree = 0
    
    sbt_gt_agree_total = 0
    model_sbt_agree_when_sbt_correct = 0
    
    with torch.no_grad():
        for batch in test_loader:
            sources = batch['source'].to(device)
            targets = batch['target'].to(device)
            ratings = batch['rating'].to(device)
            
            u_embeds = aligner(sources)
            v_embeds = aligner(targets)
            
            exist_logits, sign_logits = predictor(u_embeds, v_embeds)
            exist_probs = torch.sigmoid(exist_logits)
            sign_probs = torch.sigmoid(sign_logits)
            
            sign_labels = (ratings >= 0).int() if zero_positive else (ratings > 0).int()
            preds = (sign_logits >= threshold).int()
            
            for i in range(len(sources)):
                u = sources[i].item()
                v = targets[i].item()
                y_true = sign_labels[i].item()
                y_pred = preds[i].item()
                
                # Calculate SBT deduction for global metrics
                _, u_align = get_node_profile(u, u_embeds[i], anchors)
                _, v_align = get_node_profile(v, v_embeds[i], anchors)
                sbt_expected = 1 if u_align == v_align else 0
                
                total_edges += 1
                if sbt_expected == y_true:
                    sbt_correct += 1
                    sbt_gt_agree_total += 1
                    if y_pred == sbt_expected:
                        model_sbt_agree_when_sbt_correct += 1
                        
                if sbt_expected == y_pred:
                    model_sbt_agree += 1
                
                category = None
                if y_true == 1 and y_pred == 1 and not found_tp:
                    category = "TRUE POSITIVE (Correctly Predicted Trust)"
                    found_tp = True
                elif y_true == 0 and y_pred == 0 and not found_tn:
                    category = "TRUE NEGATIVE (Correctly Predicted Malicious)"
                    found_tn = True
                elif y_true == 0 and y_pred == 1 and not found_fp:
                    category = "FALSE POSITIVE (Type I Error)"
                    found_fp = True
                elif y_true == 1 and y_pred == 0 and not found_fn:
                    category = "FALSE NEGATIVE (Type II Error)"
                    found_fn = True
                    
                if category is not None:
                    print(f"\n>>> EXPLAINING {category} EDGE <<<")
                    print_explanation(
                        u, v, u_embeds[i], v_embeds[i], anchors, 
                        exist_probs[i].item(), sign_probs[i].item(), 
                        y_true, y_pred
                    )
                    
    print("\n" + "="*80)
    print("                 GLOBAL XAI METRICS (TEST SET OVERLAP)")
    print("="*80)
    print(f"Total Edges Evaluated: {total_edges}")
    print(f"Strict SBT Accuracy (SBT vs Ground Truth):  {sbt_correct / total_edges * 100:.2f}%")
    print(f"Model-SBT Agreement (Model vs SBT overall): {model_sbt_agree / total_edges * 100:.2f}%")
    if sbt_gt_agree_total > 0:
        clean_agreement = model_sbt_agree_when_sbt_correct / sbt_gt_agree_total * 100
        print(f"Model-SBT Agreement (On 'Clean' SBT Edges): {clean_agreement:.2f}%")
    print("="*80 + "\n")
