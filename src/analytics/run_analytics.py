#!/usr/bin/env python3
"""
run_analytics.py
Updated entry-point script for CERT Insider Threat Detection Pipeline
with enhanced feature_engineering.py and ueba_module.py compatibility.
"""

import os
import pandas as pd

from feature_Engineering import CERTFeatureEngineer
from UEBA_module import UEBAModule

# Dynamically resolve project root directory (2 levels above this script)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def run_cert_threat_pipeline(
    cert_data_path: str,
    user_dim_path: str = None,
    output_dir: str = "./cert_analysis_output"
):
    print("=" * 60)
    print("CERT Insider Threat Detection Pipeline")
    print("=" * 60)

    # Step 1: Load processed CERT data
    print("\n[1]  processed CERT data...")
    cert_df = pd.read_csv(cert_data_path)
    print(f" - Loaded {len(cert_df)} records from {cert_data_path}")

    # Step 2: Load user dimension data if available
    user_df = None
    if user_dim_path and os.path.exists(user_dim_path):
        user_df = pd.read_csv(user_dim_path)
        print(f" - Loaded user dimension data: {len(user_df)} users")
    else:
        print(" - No user dimension data provided or file not found; proceeding without context")

    # Step 3: Feature Engineering
    print("\n[2] Running Feature Engineering...")
    feature_engineer = CERTFeatureEngineer(cert_df, user_df)
    session_features = feature_engineer.run_feature_engineering()
    os.makedirs(output_dir, exist_ok=True)
    features_path = os.path.join(output_dir, "session_features.parquet")
    session_features.to_parquet(features_path, index=False)
    print(f" - Saved session features to {features_path}")

    # Step 4: Anomaly Detection (UEBA)
    print("\n[3] Running UEBA (Outlier Detection)...")
    ueba = UEBAModule(session_features)
    ueba.fit_lof(contamination=0.02)
    scores_path = os.path.join(output_dir, "ueba_scores.parquet")
    ueba.save_scores(scores_path)
    print(f" - Saved UEBA scores to {scores_path}")

    # Step 5: Save analysis reports
    print("\n[4] Saving Analysis Reports...")
    top_anomalies = ueba.top_anomalies(n=20)
    ta_path = os.path.join(output_dir, "top_anomalies.csv")
    top_anomalies.to_csv(ta_path, index=False)
    print(f" - Saved top anomalies to {ta_path}")

    user_risk_summary = ueba.user_risk_summary()
    urs_path = os.path.join(output_dir, "user_risk_summary.csv")
    user_risk_summary.to_csv(urs_path, index=True)
    print(f" - Saved user risk summary to {urs_path}")

    # Final summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Total processed sessions: {len(session_features)}")
    print(f"Top anomalies: {len(top_anomalies)}")
    print(f"Results saved in: {output_dir}/\n")

if __name__ == "__main__":
    run_cert_threat_pipeline(
        cert_data_path=os.path.join(
            ROOT_DIR, "data", "processed_data", "cert_batch_processed.csv"
        ),
        user_dim_path=os.path.join(
            ROOT_DIR, "data", "user_dimensions.csv"
        ),
        output_dir=os.path.join(
            ROOT_DIR, "cert_analysis_output"
        )
    )
