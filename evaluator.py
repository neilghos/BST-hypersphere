import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, mean_squared_error, mean_absolute_error

class SNAPEval:
    def __init__(self, task_type='sign_prediction', score_space='logit', zero_label_policy='negative'):
        """
        task_type: 
            'sign_prediction': binary classification (positive trust vs negative distrust)
            'weight_prediction': regression task for exact rating (-10 to +10)
        score_space:
            'logit': y_pred contains raw logits or signed margins
            'probability': y_pred contains probabilities in [0, 1]
        zero_label_policy:
            'negative': treat rating == 0 as the negative class
            'positive': treat rating == 0 as the positive class
            'drop': exclude rating == 0 rows from evaluation
        """
        self.task_type = task_type
        if score_space not in {'logit', 'probability'}:
            raise ValueError(f"Unknown score_space: {score_space}")
        self.score_space = score_space
        if zero_label_policy not in {'negative', 'positive', 'drop'}:
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
        
        if self.score_space == 'probability':
            thresholds = np.linspace(0.01, 0.99, 99)
            best_t = 0.5
        else:
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
            
        if self.task_type == 'sign_prediction':
            binary_true = self._binarize_sign_labels(y_true)
            
            if self.score_space == 'probability':
                t = threshold if threshold is not None else 0.5
                pred_binary = (y_pred >= t).astype(int)
                probs = np.clip(y_pred, 0.0, 1.0)
            else:
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
            
        elif self.task_type == 'weight_prediction':
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            return {
                'rmse': rmse,
                'mae': mae
            }
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")

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
            
            if evaluator.task_type == 'sign_prediction' and evaluator.zero_label_policy == 'drop':
                valid_mask = ratings != 0
                sources = sources[valid_mask]
                targets = targets[valid_mask]
                ratings = ratings[valid_mask]
            
            if len(ratings) == 0:
                continue
            
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
