# Module 3: Continual Learning Pipeline (CLP)
# Maintains adaptive behavioral baselines using online learning

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try to import river for online learning
try:
    from river import cluster, drift
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False
    print("Warning: River library not available. Using fallback implementations.")


@dataclass
class UserProfile:
    """Represents evolving user behavior profile"""
    user_id: str
    feature_baseline: Dict[str, float]
    activity_count: int
    last_updated: datetime
    drift_detected: bool = False


class ContinualLearningPipeline:
    """
    Continual Learning Pipeline for adaptive anomaly detection
    
    How it works:
    1. Maintains sliding windows of recent user behavior
    2. Uses time-decayed weighting (recent data weighted more)
    3. Implements online clustering for evolving profiles
    4. Detects concept drift and triggers retraining
    5. Adapts detection thresholds based on behavioral changes
    
    Key Innovation:
    - Prevents model degradation over time
    - Reduces false positives from legitimate behavior changes
    - Maintains both short-term and long-term memory
    """
    
    def __init__(self, window_size: int = 1000, decay_factor: float = 0.95):
        """
        Initialize Continual Learning Pipeline
        
        Args:
            window_size: Number of recent samples to maintain
            decay_factor: Exponential decay for weighting (0-1, closer to 1 = slower decay)
        """
        print("Initializing Continual Learning Pipeline...")
        
        self.window_size = window_size
        self.decay_factor = decay_factor
        
        # User profile storage
        self.user_profiles: Dict[str, UserProfile] = {}
        
        # Sliding windows for each user
        self.user_windows: Dict[str, deque] = {}
        
        # Drift detection
        if RIVER_AVAILABLE:
            self.drift_detector = drift.ADWIN()
        else:
            self.drift_detector = None
        
        # Online clustering (if available)
        if RIVER_AVAILABLE:
            self.online_clusterer = cluster.DenStream(
                decaying_factor=decay_factor,
                beta=0.5,
                mu=3
            )
        else:
            self.online_clusterer = None
        
        # Feature statistics
        self.global_feature_stats = {}
        
        # Retraining history
        self.retrain_history = []
        
        # Performance tracking
        self.performance_window = deque(maxlen=100)
        
        print("CLP initialized successfully")
    
    def fit_initial(self, historical_data: pd.DataFrame):
        """
        Initial training on historical data
        
        Args:
            historical_data: DataFrame with columns: user_id, timestamp, features...
        """
        print("Performing initial training on historical data...")
        
        # Calculate global feature statistics
        time_columns = ['timestamp', 'session_start', 'session_end']
        feature_cols = [
            col for col in historical_data.columns
            if col not in ['user_id', 'session_id'] and col not in time_columns
        ]

        
        for col in feature_cols:
            self.global_feature_stats[col] = {
                'mean': historical_data[col].mean(),
                'std': historical_data[col].std(),
                'min': historical_data[col].min(),
                'max': historical_data[col].max()
            }
        
        # Build initial user profiles
        for user_id in historical_data['user_id'].unique():
            user_data = historical_data[historical_data['user_id'] == user_id]
            
            # Calculate user-specific baselines
            feature_baseline = {}
            for col in feature_cols:
                feature_baseline[col] = user_data[col].mean()
            
            # Create profile
            profile = UserProfile(
                user_id=user_id,
                feature_baseline=feature_baseline,
                activity_count=len(user_data),
                last_updated=datetime.now()
            )
            
            self.user_profiles[user_id] = profile
            
            # Initialize sliding window with recent data
            self.user_windows[user_id] = deque(
                user_data.tail(self.window_size).to_dict('records'),
                maxlen=self.window_size
            )
        
        print(f"Initial training complete: {len(self.user_profiles)} user profiles created")
    
    def update_online(self, new_session: Dict):
        """
        Update user profile with new session data (online learning)
        
        Args:
            new_session: Dict containing session features
        """
        user_id = new_session.get('user_id')
        
        if user_id not in self.user_profiles:
            # Create new profile for unknown user
            self._initialize_new_user(user_id, new_session)
            return
        
        # Add to sliding window
        if user_id not in self.user_windows:
            self.user_windows[user_id] = deque(maxlen=self.window_size)
        
        self.user_windows[user_id].append(new_session)
        
        # Update profile with time-decayed weighting
        self._update_profile_weighted(user_id)
        
        # Check for concept drift
        drift_detected = self._detect_drift(user_id, new_session)
        
        if drift_detected:
            self.user_profiles[user_id].drift_detected = True
            print(f"Concept drift detected for user {user_id}")
        
        # Update profile metadata
        self.user_profiles[user_id].activity_count += 1
        self.user_profiles[user_id].last_updated = datetime.now()
    
    def _initialize_new_user(self, user_id: str, first_session: Dict):
        """Initialize profile for new user"""
        feature_baseline = {}
        
        for key, value in first_session.items():
            if key not in ['user_id', 'timestamp', 'session_id'] and isinstance(value, (int, float)):
                # Start with global average
                feature_baseline[key] = self.global_feature_stats.get(key, {}).get('mean', value)
        
        profile = UserProfile(
            user_id=user_id,
            feature_baseline=feature_baseline,
            activity_count=1,
            last_updated=datetime.now()
        )
        
        self.user_profiles[user_id] = profile
        self.user_windows[user_id] = deque([first_session], maxlen=self.window_size)
    
    def _update_profile_weighted(self, user_id: str):
        """
        Update user profile using time-decayed exponential weighting
        
        Recent behavior gets higher weight, older behavior gradually decays
        """
        window = list(self.user_windows[user_id])
        n_samples = len(window)
        
        if n_samples == 0:
            return
        
        # Calculate time-decayed weights (recent = higher weight)
        weights = np.array([
            self.decay_factor ** (n_samples - i - 1)
            for i in range(n_samples)
        ])
        weights = weights / weights.sum()  # Normalize
        
        # Update each feature with weighted average
        feature_cols = self.user_profiles[user_id].feature_baseline.keys()
        
        for feature in feature_cols:
            feature_values = np.array([
                session.get(feature, 0.0) for session in window
            ])
            
            weighted_mean = np.average(feature_values, weights=weights)
            
            # Update baseline
            self.user_profiles[user_id].feature_baseline[feature] = weighted_mean
    
    def _detect_drift(self, user_id: str, new_session: Dict) -> bool:
        """
        Detect concept drift using ADWIN algorithm
        
        Returns:
            True if drift detected, False otherwise
        """
        if not RIVER_AVAILABLE or self.drift_detector is None:
            # Fallback: simple threshold-based drift detection
            return self._simple_drift_detection(user_id, new_session)
        
        # Calculate deviation from baseline
        profile = self.user_profiles[user_id]
        deviations = []
        
        for feature, baseline_value in profile.feature_baseline.items():
            if feature in new_session and isinstance(new_session[feature], (int, float)):
                current_value = new_session[feature]
                
                # Normalize by standard deviation
                std = self.global_feature_stats.get(feature, {}).get('std', 1.0)
                if std > 0:
                    deviation = abs(current_value - baseline_value) / std
                    deviations.append(deviation)
        
        if deviations:
            avg_deviation = np.mean(deviations)
            
            # Feed to drift detector
            self.drift_detector.update(avg_deviation)
            
            return self.drift_detector.drift_detected
        
        return False
    
    def _simple_drift_detection(self, user_id: str, new_session: Dict) -> bool:
        """Simple drift detection fallback (without river library)"""
        profile = self.user_profiles[user_id]
        
        # Count features that deviate significantly
        significant_deviations = 0
        total_features = 0
        
        for feature, baseline_value in profile.feature_baseline.items():
            if feature in new_session and isinstance(new_session[feature], (int, float)):
                current_value = new_session[feature]
                total_features += 1
                
                # Check if deviation > 3 standard deviations
                std = self.global_feature_stats.get(feature, {}).get('std', 1.0)
                if std > 0:
                    z_score = abs(current_value - baseline_value) / std
                    if z_score > 3.0:
                        significant_deviations += 1
        
        # Drift if > 30% of features show significant deviation
        if total_features > 0:
            drift_ratio = significant_deviations / total_features
            return drift_ratio > 0.3
        
        return False
    
    def get_current_baseline(self, user_id: str) -> Optional[Dict[str, float]]:
        """
        Get current behavioral baseline for a user
        
        Args:
            user_id: User to query
            
        Returns:
            Dict of feature baselines, or None if user not found
        """
        if user_id in self.user_profiles:
            return self.user_profiles[user_id].feature_baseline.copy()
        return None
    
    def calculate_anomaly_score(self, session: Dict) -> float:
        """
        Calculate anomaly score relative to current baseline
        
        Args:
            session: Session data to score
            
        Returns:
            Anomaly score (0-1, higher = more anomalous)
        """
        user_id = session.get('user_id')
        
        if user_id not in self.user_profiles:
            return 0.5  # Unknown user, neutral score
        
        profile = self.user_profiles[user_id]
        deviations = []
        
        for feature, baseline_value in profile.feature_baseline.items():
            if feature in session and isinstance(session[feature], (int, float)):
                current_value = session[feature]
                
                # Calculate normalized deviation
                std = self.global_feature_stats.get(feature, {}).get('std', 1.0)
                if std > 0:
                    z_score = abs(current_value - baseline_value) / std
                    # Convert to 0-1 score using sigmoid
                    deviation_score = 1 / (1 + np.exp(-z_score + 2))
                    deviations.append(deviation_score)
        
        if deviations:
            return np.mean(deviations)
        return 0.5
    
    def detect_concept_drift(self) -> bool:
        """
        Check if concept drift detected for any users
        
        Returns:
            True if drift detected and retraining recommended
        """
        drift_count = sum(
            1 for profile in self.user_profiles.values()
            if profile.drift_detected
        )
        
        drift_ratio = drift_count / max(len(self.user_profiles), 1)
        
        # Recommend retraining if >20% of users show drift
        return drift_ratio > 0.2
    
    def retrain(self):
        """
        Perform full retraining on recent data
        """
        print("Performing full model retraining...")
        
        # Collect recent data from all windows
        all_recent_data = []
        
        for user_id, window in self.user_windows.items():
            all_recent_data.extend(list(window))
        
        if all_recent_data:
            recent_df = pd.DataFrame(all_recent_data)
            
            # Re-initialize with recent data
            self.fit_initial(recent_df)
            
            # Reset drift flags
            for profile in self.user_profiles.values():
                profile.drift_detected = False
            
            # Log retraining
            self.retrain_history.append({
                'timestamp': datetime.now(),
                'samples_used': len(all_recent_data),
                'users_affected': len(self.user_profiles)
            })
            
            print(f"Retraining complete: {len(all_recent_data)} samples, {len(self.user_profiles)} users")
    
    def get_profile_age(self, user_id: str) -> Optional[timedelta]:
        """
        Get time since last profile update
        
        Args:
            user_id: User to query
            
        Returns:
            Timedelta since last update, or None if user not found
        """
        if user_id in self.user_profiles:
            return datetime.now() - self.user_profiles[user_id].last_updated
        return None
    
    def export_profiles(self, filepath: str):
        """Export current user profiles"""
        profiles_data = []
        
        for user_id, profile in self.user_profiles.items():
            profiles_data.append({
                'user_id': user_id,
                'activity_count': profile.activity_count,
                'last_updated': str(profile.last_updated),
                'drift_detected': profile.drift_detected,
                **profile.feature_baseline
            })
        
        df = pd.DataFrame(profiles_data)
        df.to_csv(filepath, index=False)
        print(f"Profiles exported to {filepath}")
    
    def get_statistics(self) -> Dict:
        """Get CLP statistics"""
        return {
            'total_users': len(self.user_profiles),
            'drift_detected_users': sum(1 for p in self.user_profiles.values() if p.drift_detected),
            'retraining_count': len(self.retrain_history),
            'avg_window_size': np.mean([len(w) for w in self.user_windows.values()]) if self.user_windows else 0,
            'oldest_profile_age': max(
                [(datetime.now() - p.last_updated).days for p in self.user_profiles.values()],
                default=0
            )
        }


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("MODULE 3: CONTINUAL LEARNING PIPELINE (CLP)")
    print("=" * 80)
    
    # Initialize CLP
    clp = ContinualLearningPipeline(window_size=500, decay_factor=0.95)
    
    # Create sample historical data
    np.random.seed(42)
    historical_data = pd.DataFrame({
        'user_id': np.repeat(['user001', 'user002', 'user003'], 100),
        'timestamp': pd.date_range('2025-01-01', periods=300, freq='H'),
        'file_rate': np.random.exponential(2, 300),
        'logon_rate': np.random.poisson(3, 300),
        'device_rate': np.random.gamma(2, 1, 300),
        'session_duration': np.random.normal(3600, 600, 300)
    })
    
    # Initial training
    clp.fit_initial(historical_data)
    
    # Simulate streaming updates
    print("\nSimulating streaming updates...")
    for i in range(10):
        new_session = {
            'user_id': 'user001',
            'timestamp': datetime.now(),
            'file_rate': np.random.exponential(3),  # Slightly different pattern
            'logon_rate': np.random.poisson(4),
            'device_rate': np.random.gamma(2.5, 1),
            'session_duration': np.random.normal(4000, 700)
        }
        
        clp.update_online(new_session)
        
        # Calculate anomaly score
        score = clp.calculate_anomaly_score(new_session)
        print(f"Session {i+1}: Anomaly score = {score:.3f}")
    
    # Check for drift
    if clp.detect_concept_drift():
        print("\n⚠️ Concept drift detected - retraining recommended")
        clp.retrain()
    
    # Get statistics
    stats = clp.get_statistics()
    print("\n" + "=" * 80)
    print("CLP STATISTICS:")
    print("=" * 80)
    for key, value in stats.items():
        print(f"{key}: {value}")
    print("=" * 80)
