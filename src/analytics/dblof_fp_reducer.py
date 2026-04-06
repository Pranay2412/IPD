# Enhanced DBLOF Post-Processing Module for False Positive Reduction (column names fixed)
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


class DBLOFFalsePositiveReducer:
    """
    Post-processing module to reduce false positives in DBLOF results
    while maintaining high recall for insider threat detection.
    """

    def __init__(self, dblof_scores_df, session_features_df, insider_user_ids):
        """
        Initialize the false positive reducer
        """
        self.scores_df = dblof_scores_df.copy()
        self.features_df = session_features_df.copy()
        self.insider_user_ids = insider_user_ids

        # Merge datasets
        self.merged_df = self.scores_df.merge(
            self.features_df,
            on=['user_id', 'session_id'],
            how='left'
        )

        # Add true labels
        self.merged_df['true_label'] = self.merged_df['user_id'].apply(
            lambda uid: 1 if uid in insider_user_ids else 0
        )

        # Add dummy total_emails column if missing
        if 'total_emails' not in self.merged_df.columns:
            self.merged_df['total_emails'] = 0

        print(f"Initialized with {len(self.merged_df)} sessions")
        print(f"Original outliers flagged: {self.merged_df['is_outlier'].sum()}")
        print(f"True positives in original: {((self.merged_df['is_outlier']) & (self.merged_df['true_label'] == 1)).sum()}")
        print(f"False positives in original: {((self.merged_df['is_outlier']) & (self.merged_df['true_label'] == 0)).sum()}")

    def apply_anomaly_score_threshold(self, percentile_threshold=80):
        """
        Apply anomaly score threshold to reduce low-confidence outliers
        """
        print(f"\n=== APPLYING ANOMALY SCORE THRESHOLD (>{percentile_threshold}th percentile) ===")

        # Calculate threshold from outliers
        outlier_scores = self.merged_df[self.merged_df['is_outlier']]['anomaly_score']
        threshold = np.percentile(outlier_scores, percentile_threshold)

        # Apply threshold
        self.merged_df['high_confidence_outlier'] = (
            self.merged_df['is_outlier'] &
            (self.merged_df['anomaly_score'] >= threshold)
        )

        original_outliers = self.merged_df['is_outlier'].sum()
        filtered_outliers = self.merged_df['high_confidence_outlier'].sum()

        print(f"Anomaly score threshold: {threshold:.2e}")
        print(f"Outliers after threshold: {filtered_outliers} (was {original_outliers})")
        print(f"Reduction: {original_outliers - filtered_outliers} sessions ({(1 - filtered_outliers/original_outliers)*100:.1f}%)")

        return threshold

    def apply_session_context_filters(self):
        """
        Apply rule-based filters on session characteristics
        """
        print(f"\n=== APPLYING SESSION CONTEXT FILTERS ===")

        # Calculate per-user baselines
        user_stats = self.merged_df.groupby('user_id').agg({
            'device_rate': 'mean',        # replaces total_device_ops
            'file_rate': 'mean',          # replaces total_file_ops
            'http_rate': 'mean',          # replaces total_http_requests
            'logon_rate': 'mean',         # replaces total_logons
            # If you add an email feature in future, map accordingly
        })

        # Rename columns for clarity
        user_stats = user_stats.rename(columns={
            'device_rate': 'device_rate_baseline',
            'file_rate': 'file_rate_baseline',
            'http_rate': 'http_rate_baseline',
            'logon_rate': 'logon_rate_baseline'
        })

        # Merge baselines
        df_with_baselines = self.merged_df.merge(
            user_stats, left_on='user_id', right_index=True, how='left'
        )

        # Suspicious: >3x file ops baseline, >2x device ops baseline, etc.
        suspicious_conditions = (
            (df_with_baselines['file_rate'] > df_with_baselines['file_rate_baseline'] * 3) |
            (df_with_baselines['device_rate'] > df_with_baselines['device_rate_baseline'] * 2) |
            # Email rate skipped (not available) or you may add:
            # (df_with_baselines['total_emails'] > 0), if ever present in your features
            (df_with_baselines.get('after_hours_activity', 0) > 0)
        )

        # Apply context filters to outliers
        self.merged_df['context_filtered_outlier'] = (
            self.merged_df.get('high_confidence_outlier', self.merged_df['is_outlier']) &
            suspicious_conditions
        )

        original = self.merged_df.get('high_confidence_outlier', self.merged_df['is_outlier']).sum()
        filtered = self.merged_df['context_filtered_outlier'].sum()

        print(f"Outliers after context filtering: {filtered} (was {original})")
        print(f"Reduction: {original - filtered} sessions ({(1 - filtered/original)*100:.1f}%)")

    def apply_ensemble_detection(self):
        """
        Use Isolation Forest as ensemble with DBLOF for cross-validation
        """
        print(f"\n=== APPLYING ENSEMBLE DETECTION (DBLOF + Isolation Forest) ===")

        feature_cols = [
            col for col in self.features_df.columns
            if col not in ['user_id', 'session_id'] and
               self.features_df[col].dtype in ['int64', 'float64']
        ]

        feature_data = self.merged_df[feature_cols].fillna(0)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_data)

        iso_forest = IsolationForest(
            contamination=0.10,
            random_state=42,
            n_jobs=-1
        )

        iso_predictions = iso_forest.fit_predict(scaled_features)
        iso_outliers = iso_predictions == -1

        current_outliers = self.merged_df.get('context_filtered_outlier',
                                              self.merged_df.get('high_confidence_outlier',
                                                                 self.merged_df['is_outlier']))

        self.merged_df['ensemble_outlier'] = current_outliers & iso_outliers

        original = current_outliers.sum()
        ensemble = self.merged_df['ensemble_outlier'].sum()

        print(f"Isolation Forest flagged: {iso_outliers.sum()} outliers")
        print(f"Ensemble outliers (both agree): {ensemble} (was {original})")
        print(f"Reduction: {original - ensemble} sessions ({(1 - ensemble/original)*100:.1f}%)")

    def apply_supervised_rescoring(self, test_size=0.3):
        """
        Use supervised learning to re-score outliers and reduce false positives
        """
        print(f"\n=== APPLYING SUPERVISED RE-SCORING ===")

        feature_cols = [
            col for col in self.features_df.columns
            if col not in ['user_id', 'session_id'] and
               self.features_df[col].dtype in ['int64', 'float64']
        ]

        feature_cols_with_score = feature_cols + ['anomaly_score']

        X = self.merged_df[feature_cols_with_score].fillna(0)
        y = self.merged_df['true_label']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )

        rf_model.fit(X_train, y_train)

        insider_probabilities = rf_model.predict_proba(X)[:, 1]
        prob_threshold = 0.3

        self.merged_df['supervised_insider_prob'] = insider_probabilities
        self.merged_df['supervised_outlier'] = (
            self.merged_df['supervised_insider_prob'] >= prob_threshold
        )

        y_test_pred = rf_model.predict_proba(X_test)[:, 1] >= prob_threshold

        test_precision = precision_score(y_test, y_test_pred, zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
        supervised_outliers = self.merged_df['supervised_outlier'].sum()

        print(f"Supervised model test performance:")
        print(f"  Precision: {test_precision:.3f}")
        print(f"  Recall: {test_recall:.3f}")
        print(f"  F1-Score: {test_f1:.3f}")
        print(f"Supervised outliers flagged: {supervised_outliers}")

        feature_importance = pd.DataFrame({
            'feature': feature_cols_with_score,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\nTop 5 most important features:")
        print(feature_importance.head())

        return rf_model

    def evaluate_final_performance(self):
        """
        Evaluate the final performance after all filtering steps
        """
        print(f"\n=== FINAL PERFORMANCE EVALUATION ===")

        outlier_cols = ['supervised_outlier', 'ensemble_outlier', 'context_filtered_outlier',
                        'high_confidence_outlier', 'is_outlier']

        final_outlier_col = None
        for col in outlier_cols:
            if col in self.merged_df.columns:
                final_outlier_col = col
                break

        if final_outlier_col is None:
            final_outlier_col = 'is_outlier'

        print(f"Using {final_outlier_col} for final evaluation")

        y_true = self.merged_df['true_label']
        y_pred = self.merged_df[final_outlier_col].astype(int)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        print(f"\n=== CONFUSION MATRIX (FINAL) ===")
        print(f"TP (Insiders flagged):     {tp}")
        print(f"FP (Benign flagged):       {fp}")
        print(f"FN (Insiders missed):      {fn}")
        print(f"TN (Benign not flagged):   {tn}")
        print(f"Total outliers flagged:    {tp + fp}")

        print(f"\n=== FINAL METRICS ===")
        print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")

        original_precision = precision_score(y_true, self.merged_df['is_outlier'].astype(int), zero_division=0)
        original_recall = recall_score(y_true, self.merged_df['is_outlier'].astype(int), zero_division=0)
        original_f1 = f1_score(y_true, self.merged_df['is_outlier'].astype(int), zero_division=0)

        print(f"\n=== IMPROVEMENT OVER ORIGINAL ===")
        print(f"Precision: {original_precision:.4f} → {precision:.4f} ({(precision-original_precision)*100:+.2f}pp)")
        print(f"Recall:    {original_recall:.4f} → {recall:.4f} ({(recall-original_recall)*100:+.2f}pp)")
        print(f"F1-Score:  {original_f1:.4f} → {f1:.4f} ({(f1-original_f1)*100:+.2f}pp)")

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        }

    def run_full_pipeline(self, score_percentile=80, use_ensemble=True, use_supervised=True):
        """
        Run the complete false positive reduction pipeline
        """
        print("=" * 80)
        print("DBLOF FALSE POSITIVE REDUCTION PIPELINE")
        print("=" * 80)

        # Step 1: Anomaly score thresholding
        self.apply_anomaly_score_threshold(percentile_threshold=score_percentile)

        # Step 2: Session context filters
        self.apply_session_context_filters()

        # Step 3: Ensemble detection (optional)
        if use_ensemble:
            self.apply_ensemble_detection()

        # Step 4: Supervised re-scoring (optional)
        if use_supervised:
            self.apply_supervised_rescoring()

        # Step 5: Final evaluation
        final_metrics = self.evaluate_final_performance()

        print("\n" + "=" * 80)
        print("FALSE POSITIVE REDUCTION COMPLETED")
        print("=" * 80)

        return final_metrics

    def save_results(self, output_dir):
        """
        Save the processed results
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Save processed scores
        results_path = os.path.join(output_dir, "dblof_scores_fp_reduced.csv")
        self.merged_df.to_csv(results_path, index=False)
        print(f"Processed results saved to: {results_path}")

        # Save final flagged sessions
        final_outlier_col = None
        outlier_cols = ['supervised_outlier', 'ensemble_outlier', 'context_filtered_outlier',
                        'high_confidence_outlier']
        for col in outlier_cols:
            if col in self.merged_df.columns:
                final_outlier_col = col
                break
        if final_outlier_col:
            final_outliers = self.merged_df[self.merged_df[final_outlier_col]]
            outliers_path = os.path.join(output_dir, "final_outliers_fp_reduced.csv")
            final_outliers.to_csv(outliers_path, index=False)
            print(f"Final outliers saved to: {outliers_path}")
