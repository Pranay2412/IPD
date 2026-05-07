# module7_analyst_feedback.py
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict

class AnalystFeedbackIntegration:
    """Incorporates analyst feedback to improve detection"""
    
    def __init__(self, initial_threshold: float = 0.5, learning_rate: float = 0.01):
        self.threshold = initial_threshold
        self.learning_rate = learning_rate
        self.feedback_history = []
        self.pattern_adjustments = defaultdict(float)
    
    def process_feedback(self, session_id: str, prediction_score: float,
                        analyst_decision: str, investigation_notes: str = ""):
        """Process analyst feedback"""
        
        # Record feedback
        feedback = {
            'session_id': session_id,
            'prediction_score': prediction_score,
            'analyst_decision': analyst_decision,
            'notes': investigation_notes
        }
        self.feedback_history.append(feedback)
        
        # Update threshold using simple reinforcement
        if analyst_decision == 'false_positive':
            # Increase threshold (be more conservative)
            adjustment = self.learning_rate * (1 - prediction_score)
            self.threshold += adjustment
        elif analyst_decision == 'true_positive':
            # Decrease threshold (be more sensitive)
            adjustment = self.learning_rate * prediction_score
            self.threshold -= adjustment
        
        # Keep threshold in valid range
        self.threshold = np.clip(self.threshold, 0.1, 0.9)
        
        print(f"Feedback processed. New threshold: {self.threshold:.3f}")
    
    def get_adaptive_threshold(self, user_profile: str = 'default') -> float:
        """Get current adaptive threshold"""
        return self.threshold
    
    def get_feedback_summary(self) -> Dict:
        """Summarize analyst feedback"""
        if not self.feedback_history:
            return {'total_feedback': 0}
        
        decisions = [f['analyst_decision'] for f in self.feedback_history]
        
        return {
            'total_feedback': len(self.feedback_history),
            'true_positives': decisions.count('true_positive'),
            'false_positives': decisions.count('false_positive'),
            'current_threshold': self.threshold
        }

# Usage
afi = AnalystFeedbackIntegration(initial_threshold=0.5)

# Analyst reviews alert
afi.process_feedback(
    session_id='session_12345',
    prediction_score=0.75,
    analyst_decision='false_positive',
    investigation_notes='User was on legitimate project deadline'
)

# Get updated threshold
new_threshold = afi.get_adaptive_threshold()
print(f"Updated detection threshold: {new_threshold:.3f}")
