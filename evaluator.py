import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, mean_squared_error, mean_absolute_error

class SNAPEval:
    def __init__(self, task_type='sign_prediction'):
        """
        task_type: 
            'sign_prediction': binary classification (positive trust vs negative distrust)
            'weight_prediction': regression task for exact rating (-10 to +10)
        """
        self.task_type = task_type
        
    def eval(self, input_dict):
        y_true = input_dict['y_true']
        y_pred = input_dict['y_pred']
        
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
            
        if self.task_type == 'sign_prediction':
            # Map raw SNAP ratings (-10 to +10) to binary labels: >0 is 1 (trust), <=0 is 0 (distrust)
            binary_true = (y_true > 0).astype(int)
            
            if (y_pred >= 0).all() and (y_pred <= 1).all():
                pred_binary = (y_pred >= 0.5).astype(int)
                probs = y_pred
            else:
                pred_binary = (y_pred >= 0.0).astype(int)
                probs = 1 / (1 + np.exp(-np.clip(y_pred, -10, 10))) 
                
            acc = accuracy_score(binary_true, pred_binary)
            f1_macro = f1_score(binary_true, pred_binary, average='macro')
            f1_pos = f1_score(binary_true, pred_binary, pos_label=1)
            f1_neg = f1_score(binary_true, pred_binary, pos_label=0)
            
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

def evaluate_pipeline(aligner, dataloader, device, evaluator, predictor=None, split_name="Test"):
    aligner.eval()
    if predictor is not None:
        predictor.eval()
    
    all_preds = []
    all_labels = []
    
    print(f"\n--- Running Evaluation on {split_name} Set ---")
    with torch.no_grad():
        for batch in dataloader:
            sources = batch['source'].to(device)
            targets = batch['target'].to(device)
            ratings = batch['rating'].to(device)
            
            # Filter out any 0 ratings if they exist
            valid_mask = ratings != 0
            sources = sources[valid_mask]
            targets = targets[valid_mask]
            ratings = ratings[valid_mask]
            
            if len(ratings) == 0:
                continue
            
            # 1. Project nodes
            u_embeds = aligner(sources)
            v_embeds = aligner(targets)
            
            if predictor is not None:
                # Use the Learned Sign Head
                _, sign_logits = predictor(u_embeds, v_embeds)
                logits = sign_logits
            else:
                # 2. Measure raw geometric distance
                cos_sim = F.cosine_similarity(u_embeds, v_embeds, dim=-1)
                
                # 3. The Geometric Shift
                # Shift the 0.5 equator to 0.0 so it acts as a standard Logit for SNAPEval
                logits = cos_sim - 0.5
            
            all_preds.append(logits)
            all_labels.append(ratings) # Pass raw ratings, SNAPEval handles the >0 mapping
            
    # Compile
    all_preds_tensor = torch.cat(all_preds)
    all_labels_tensor = torch.cat(all_labels)
    
    # Run through your evaluator
    input_dict = {
        'y_true': all_labels_tensor,
        'y_pred': all_preds_tensor
    }
    
    metrics = evaluator.eval(input_dict)
    
    print(f"Results for {split_name}:")
    for k, v in metrics.items():
        print(f"  {k.upper()}: {v:.4f}")
        
    return metrics