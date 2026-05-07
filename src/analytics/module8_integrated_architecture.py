# module8_integrated_architecture.py
from typing import Dict, List
import pandas as pd

class InsiderThreatPipeline:
    """Complete integrated pipeline"""
    
    def __init__(self, **module_configs):
        # Initialize all modules
        from module1_contextual_reasoning_engine import ContextualReasoningEngine
        from module2_unified_threat_graph import UnifiedThreatCorrelationGraph
        from module3_continual_learning import ContinualLearningPipeline
        from module5_explainable_ai_layer import ExplainableAILayer
        from module7_analyst_feedback import AnalystFeedbackIntegration
        
        self.cre = ContextualReasoningEngine() if module_configs.get('cre_enabled') else None
        self.utcg = UnifiedThreatCorrelationGraph() if module_configs.get('utcg_enabled') else None
        self.clp = ContinualLearningPipeline() if module_configs.get('clp_enabled') else None
        self.xai = None  # Initialized after model training
        self.afi = AnalystFeedbackIntegration() if module_configs.get('afi_enabled') else None
    
    def process_batch(self, cert_data: pd.DataFrame) -> Dict:
        """Process a batch of data through the pipeline"""
        results = {
            'high_risk_alerts': [],
            'medium_risk_alerts': [],
            'low_risk_alerts': []
        }
        
        # Step 1: DBLOF anomaly detection
        anomaly_scores = self._run_dblof(cert_data)
        
        # Step 2: Contextual reasoning (if enabled)
        if self.cre:
            for session in anomaly_scores:
                intent = self.cre.analyze_intent(session, user_context)
                session['intent_analysis'] = intent
        
        # Step 3: Graph analysis (if enabled)
        if self.utcg:
            for session in anomaly_scores:
                patterns = self.utcg.detect_cross_domain_patterns(session['user_id'])
                session['graph_patterns'] = patterns
        
        # Categorize by risk
        for session in anomaly_scores:
            risk_score = self._calculate_composite_risk(session)
            
            if risk_score > 0.7:
                results['high_risk_alerts'].append(session)
            elif risk_score > 0.4:
                results['medium_risk_alerts'].append(session)
            else:
                results['low_risk_alerts'].append(session)
        
        return results

# Usage
pipeline = InsiderThreatPipeline(
    cre_enabled=True,
    utcg_enabled=True,
    afi_enabled=True
)

results = pipeline.process_batch(cert_data)
print(f"High-risk alerts: {len(results['high_risk_alerts'])}")
