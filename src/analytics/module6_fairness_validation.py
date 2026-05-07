# module6_fairness_validation.py
import pandas as pd
import numpy as np
from typing import Dict, List

class FairnessValidationFramework:
    """Validates and corrects detection bias"""
    
    def __init__(self, protected_attributes: List[str]):
        self.protected_attributes = protected_attributes
    
    def evaluate_fairness(self, predictions: pd.Series, 
                         ground_truth: pd.Series,
                         user_attributes: pd.DataFrame) -> Dict:
        """Evaluate fairness metrics"""
        
        results = {
            'overall_metrics': {},
            'group_metrics': {},
            'bias_detected': False
        }
        
        # Overall metrics
        results['overall_metrics'] = {
            'accuracy': (predictions == ground_truth).mean(),
            'precision': self._calculate_precision(predictions, ground_truth),
            'recall': self._calculate_recall(predictions, ground_truth)
        }
        
        # Per-group metrics
        for attr in self.protected_attributes:
            if attr in user_attributes.columns:
                for group_value in user_attributes[attr].unique():
                    mask = user_attributes[attr] == group_value
                    
                    group_pred = predictions[mask]
                    group_truth = ground_truth[mask]
                    
                    group_metrics = {
                        'precision': self._calculate_precision(group_pred, group_truth),
                        'recall': self._calculate_recall(group_pred, group_truth),
                        'flag_rate': group_pred.mean()
                    }
                    
                    results['group_metrics'][f"{attr}_{group_value}"] = group_metrics
        
        # Detect bias (simple threshold)
        flag_rates = [m['flag_rate'] for m in results['group_metrics'].values()]
        if flag_rates:
            max_disparity = max(flag_rates) - min(flag_rates)
            results['bias_detected'] = max_disparity > 0.2  # 20% threshold
            results['demographic_parity'] = max_disparity
        
        return results
    
    def _calculate_precision(self, pred, truth):
        tp = ((pred == 1) & (truth == 1)).sum()
        fp = ((pred == 1) & (truth == 0)).sum()
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    def _calculate_recall(self, pred, truth):
        tp = ((pred == 1) & (truth == 1)).sum()
        fn = ((pred == 0) & (truth == 1)).sum()
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

# Usage
fvf = FairnessValidationFramework(protected_attributes=['department', 'role'])

fairness_report = fvf.evaluate_fairness(
    predictions=model_predictions,
    ground_truth=true_labels,
    user_attributes=user_demographics
)

if fairness_report['bias_detected']:
    print("⚠️ Bias detected in detection system")
    print(f"Demographic parity difference: {fairness_report['demographic_parity']:.2%}")
