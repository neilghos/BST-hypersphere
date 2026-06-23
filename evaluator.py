import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

class SNAPEval:
    def __init__(self, zero_label_policy='negative'):
        """
        zero_label_policy:
            'negative': treat rating == 0 as the negative class
            'positive': treat rating == 0 as the positive class
        """
        if zero_label_policy not in {'negative', 'positive'}:
            raise ValueError(f"Unknown zero_label_policy: {zero_label_policy}")
        self.zero_label_policy = zero_label_policy

    def _binarize_sign_labels(self, y_true):
        if self.zero_label_policy == 'positive':
            return (y_true >= 0).astype(int)
        return (y_true > 0).astype(int)
        
    def find_best_threshold(self, y_true, y_pred, metric='acc'):
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
            
        binary_true = self._binarize_sign_labels(y_true)
        
        # Logit space thresholding
        thresholds = np.linspace(-5.0, 5.0, 101)
        best_t = 0.0
        best_score = -1
        
        for t in thresholds:
            pred_binary = (y_pred >= t).astype(int)
            if metric == 'acc':
                score = accuracy_score(binary_true, pred_binary)
            elif metric == 'f1_macro':
                score = f1_score(binary_true, pred_binary, average='macro', zero_division=0)
            elif metric == 'f1_pos':
                score = f1_score(binary_true, pred_binary, pos_label=1, zero_division=0)
            elif metric == 'f1_neg':
                score = f1_score(binary_true, pred_binary, pos_label=0, zero_division=0)
                
            if score > best_score:
                best_score = score
                best_t = t
                
        return best_t
        
    def eval(self, input_dict, threshold=None):
        y_true = input_dict['y_true']
        y_pred = input_dict['y_pred']
        
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
            
        binary_true = self._binarize_sign_labels(y_true)
        
        t = threshold if threshold is not None else 0.0
        pred_binary = (y_pred >= t).astype(int)
        probs = 1 / (1 + np.exp(-np.clip(y_pred, -10, 10))) 
        
        acc = accuracy_score(binary_true, pred_binary)
        f1_macro = f1_score(binary_true, pred_binary, average='macro', zero_division=0)
        f1_pos = f1_score(binary_true, pred_binary, pos_label=1, zero_division=0)
        f1_neg = f1_score(binary_true, pred_binary, pos_label=0, zero_division=0)
        
        try:
            auc = roc_auc_score(binary_true, probs)
        except ValueError:
            auc = float('nan')
            
        return {
            'acc': acc,
            'f1_macro': f1_macro,
            'f1_pos': f1_pos,
            'f1_neg': f1_neg,
            'auc': auc
        }

def evaluate_pipeline(
    aligner,
    dataloader,
    device,
    evaluator,
    predictor=None,
    split_name="Test",
    threshold=None,
    return_raw=False,
    verbose=True,
):
    aligner.eval()
    if predictor is not None:
        predictor.eval()
    
    all_preds = []
    all_labels = []
    
    if verbose:
        print(f"\n--- Running Evaluation on {split_name} Set ---")
    with torch.no_grad():
        for batch in dataloader:
            sources = batch['source'].to(device)
            targets = batch['target'].to(device)
            ratings = batch['rating'].to(device)
            
            u_embeds = aligner(sources)
            v_embeds = aligner(targets)
            
            if predictor is not None:
                _, sign_logits = predictor(u_embeds, v_embeds)
                logits = sign_logits
            else:
                cos_sim = F.cosine_similarity(u_embeds, v_embeds, dim=-1)
                logits = cos_sim - 0.5
            
            all_preds.append(logits)
            all_labels.append(ratings) # Pass raw ratings, SNAPEval handles the >0 mapping
            
    all_preds_tensor = torch.cat(all_preds)
    all_labels_tensor = torch.cat(all_labels)
    
    input_dict = {
        'y_true': all_labels_tensor,
        'y_pred': all_preds_tensor
    }
    
    metrics = evaluator.eval(input_dict, threshold=threshold)
    
    if verbose:
        print(f"Results for {split_name}" + (f" (Threshold: {threshold:.4f}):" if threshold is not None else ":"))
        for k, v in metrics.items():
            print(f"  {k.upper()}: {v:.4f}")
        
    if return_raw:
        return metrics, all_labels_tensor, all_preds_tensor
    return metrics
