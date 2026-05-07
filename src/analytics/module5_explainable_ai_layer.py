# module5_explainable_ai_layer.py
import pandas as pd
import numpy as np
from typing import Dict, List

class ExplainableAILayer:
    """Generates human-readable explanations for detections"""
    
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        
        # Try to import SHAP
        try:
            import shap
            self.shap_available = True
            self.explainer = shap.Explainer(model)
        except ImportError:
            self.shap_available = False
            print("SHAP not available. Using simplified explanations.")
    
    def explain_prediction(self, session_data: Dict, prediction_label: str) -> Dict:
        """Generate explanation for a prediction"""
        
        # Extract features
        features = [session_data.get(f, 0) for f in self.feature_names]
        
        # Get feature importances (if available)
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        else:
            # Use simple feature values as proxy
            importances = np.abs(features)
        
        # Rank features
        feature_importance_pairs = list(zip(self.feature_names, importances, features))
        feature_importance_pairs.sort(key=lambda x: x, reverse=True)
        
        # Generate natural language summary
        top_features = feature_importance_pairs[:3]
        explanation_parts = []
        
        for feature, importance, value in top_features:
            explanation_parts.append(
                f"{feature.replace('_', ' ')} ({value:.2f}) contributed {importance:.1%}"
            )
        
        natural_language = (
            f"Prediction: {prediction_label}. "
            f"Key factors: {'; '.join(explanation_parts)}."
        )
        
        return {
            'prediction': prediction_label,
            'natural_language_summary': natural_language,
            'feature_importances': dict(zip(self.feature_names, importances)),
            'top_contributing_features': [f for f in top_features]
        }

# Usage
xai = ExplainableAILayer(
    model=your_trained_model,
    feature_names=['file_rate', 'logon_rate', 'device_rate']
)

explanation = xai.explain_prediction(
    session_data={'file_rate': 10.5, 'logon_rate': 3.2},
    prediction_label='insider_threat'
)

print(explanation['natural_language_summary'])
