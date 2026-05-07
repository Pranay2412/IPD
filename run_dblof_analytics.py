#!/usr/bin/env python3
"""
Enhanced CERT Insider Threat Detection Pipeline using DBLOF
Implements the methodology from the research paper:
"Enhancing Insider Threat Detection in Imbalanced Cybersecurity Settings Using the Density-Based Local Outlier Factor Algorithm"
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.analytics.feature_Engineering import CERTFeatureEngineer  # Your existing module
from src.analytics.dblof_ueba_module import DBLOFUEBAModule  # New DBLOF module

# Known insider user IDs from your dataset
INSIDER_USER_IDS = {
    'AAF0535', 'AAM0658', 'ABC0174', 'ACA1126', 'ACM1770', 'ACM2278', 'ADC1257',
    'AJR0932', 'AKR0057', 'ALT1465', 'ALW0764', 'AMC1883', 'ASH0298', 'ASH0458',
    'AYG1697', 'BBS0039', 'BDP0096', 'BDV0168', 'BGC0686', 'BIH0745', 'BLS0678',
    'BSS0369', 'BTL0226', 'BYO1846', 'CAH0936', 'CBW1826', 'CCA0046', 'CCH0959',
    'CCL0068', 'CDE1846', 'CEJ0109', 'CGF1056', 'CHP1711', 'CIF1430', 'CKP0630',
    'CMP2946', 'CPD1525', 'CQW0652', 'CRD0272', 'CSC0217', 'CSF0929', 'CSF2712',
    'CWW1120', 'DAS1320', 'DCC1119', 'DCH0843', 'DIB0285', 'DIM0330', 'DKG0161',
    'DNJ0740', 'DQM0809', 'DRR0162', 'EDB0714', 'EGD0132', 'EHB0824', 'EHD0584',
    'ELM1123', 'ELT1370', 'EPG1196', 'ETW0002', 'FMG0527', 'FSC0601', 'FTM0406',
    'FZG0389', 'GCD0194', 'GFM1815', 'GHL0460', 'GKW0043', 'GMB0400', 'GTD0219',
    'GWG0497', 'HBO0413', 'HBP1076', 'HFC0492', 'HIS1394', 'HJB0462', 'HJB0742',
    'HMS1658', 'HSN0675', 'HXL0968', 'HXP0976', 'ICB1354', 'IHC0561', 'IJM0776',
    'IKR0401', 'IRH1224', 'ISW0738', 'ITA0159', 'IUB0565', 'JAK0783', 'JAL0811',
    'JCC0016', 'JCE0258', 'JGT0221', 'JIG1593', 'JJM0203', 'JKB0287', 'JLM0364',
    'JMB0308', 'JMM0613', 'JNR1592', 'JPH1910', 'JRG0207', 'JTC1885', 'JTM0223',
    'JUP1472', 'KBC1390', 'KCM0466', 'KDC0996', 'KEW0198', 'KLH0596', 'KPC0073',
    'KRL0501', 'KSS1005', 'KTW0365', 'LAH0463', 'LBE0376', 'LCC0819', 'LJO1360',
    'LJR0523', 'LQC0479', 'LVF1626', 'MAR0955', 'MAS0025', 'MBG3183', 'MCF0600',
    'MCP0611', 'MDH0580', 'MDS0680', 'MGB1235', 'MIB0203', 'MJR0711', 'MOS0047',
    'MPF0690', 'MPM0220', 'MSO0222', 'MTP1582', 'MYD0978', 'NAH1366', 'NIV1608',
    'NKM1738', 'NKN1405', 'NWT0098', 'NZL1395', 'OFS0030', 'OHS0036', 'OKM1092',
    'ONS0995', 'OSS1463', 'PBC0077', 'PLJ1771', 'PNL0301', 'PPF0435', 'PSF0133',
    'PTH0005', 'RAB0589', 'RAR0725', 'RCG0584', 'RCW0822', 'REF1924', 'RGG0064',
    'RHL0992', 'RKD0604', 'RMW0542', 'RRS0056', 'SAF1942', 'SIS0042', 'SLL0193',
    'SNK1280', 'TAP0551', 'TMC0934', 'TMT0851', 'TNB1616', 'TNM0961', 'TRC1838',
    'VAH1292', 'VCF1602', 'VRP0267', 'VSS0154', 'WCW1013', 'WDD0366', 'WDT1634',
    'WHB1247', 'WOS1834', 'WSK1857', 'WWW0701', 'XHW0498', 'ZEH0685', 'ZIE0741',
    'ZKP0542'
}

def main():
    """Main pipeline execution"""
    print("=" * 80)
    print("ENHANCED CERT INSIDER THREAT DETECTION PIPELINE (DBLOF)")
    print("'Enhancing Insider Threat Detection in Imbalanced Cybersecurity Settings'")
    print("=" * 80)
    
    # Configuration
    cert_data_path = "data/processed_data/cert_batch_processed.csv"
    user_data_path = "data/user_dimensions.csv"
    output_dir = "cert_dblof_analysis_output"
    
    try:
        # Step 1: Load Data
        print("\n[1] Loading processed CERT data...")
        cert_df = pd.read_csv(cert_data_path)
        user_df = pd.read_csv(user_data_path)
        
        print(f" - Loaded {len(cert_df)} records from {cert_data_path}")
        print(f" - Loaded user dimension data: {len(user_df)} users")
        
        # Step 2: Feature Engineering (using your existing pipeline)
        print("\n[2] Running Feature Engineering...")
        feature_engineer = CERTFeatureEngineer(cert_df, user_df)
        session_features = feature_engineer.run_feature_engineering()
        
        os.makedirs(output_dir, exist_ok=True)
        features_path = os.path.join(output_dir, "session_features.csv")
        session_features.to_csv(features_path, index=False)
        print(f" - Saved session features to {features_path}")
        
        # Step 3: Enhanced DBLOF Analysis
        print("\n[3] Running Enhanced DBLOF Analysis...")
        print(" - Implementing contamination parameter optimization...")
        
        # Initialize DBLOF UEBA module
        dblof_ueba = DBLOFUEBAModule(session_features)
        
        # Fit DBLOF model with insider knowledge for optimization
        dblof_ueba.fit_dblof(insider_user_ids=INSIDER_USER_IDS)
        
        # Get contamination optimization report
        print("\n[4] Contamination Parameter Optimization Results:")
        optimization_report = dblof_ueba.get_contamination_optimization_report()
        
        # Step 5: Save Enhanced Results
        print("\n[5] Saving Enhanced Analysis Results...")
        dblof_ueba.save_results(output_dir)
        # Step 6: Performance Summary
        print("\n[6] Analysis Summary:")
        print("=" * 50)
        
        if dblof_ueba.scores_df is not None:
            total_sessions = len(dblof_ueba.scores_df)
            total_outliers = dblof_ueba.scores_df['is_outlier'].sum()
            outlier_rate = total_outliers / total_sessions
            
            # Calculate insider detection metrics
            insider_sessions = dblof_ueba.scores_df[
                dblof_ueba.scores_df['user_id'].isin(INSIDER_USER_IDS)
            ]
            total_insider_sessions = len(insider_sessions)
            detected_insider_sessions = insider_sessions['is_outlier'].sum()
            
            print(f"Total processed sessions: {total_sessions}")
            print(f"Sessions flagged as outliers: {total_outliers}")
            print(f"Overall outlier rate: {outlier_rate:.4f}")
            print(f"Best contamination parameter: {dblof_ueba.best_contamination}")
            
            if total_insider_sessions > 0:
                insider_detection_rate = detected_insider_sessions / total_insider_sessions
                print(f"\nInsider Threat Detection:")
                print(f"Total insider sessions: {total_insider_sessions}")
                print(f"Detected insider sessions: {detected_insider_sessions}")
                print(f"Insider detection rate: {insider_detection_rate:.4f}")
            
            # Show top risky users
            print(f"\nTop 10 Users by Risk Level:")
            user_summary = dblof_ueba.get_user_risk_summary()
            if user_summary is not None:
                top_users = user_summary.head(10)
                print(top_users[['total_sessions', 'outlier_sessions', 'outlier_rate', 'max_anomaly_score']])
        
        print("\n" + "=" * 80)
        print("ENHANCED DBLOF PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Results saved in: {output_dir}/")
        print("\nNext steps:")
        print("1. Review contamination_optimization.csv for parameter tuning results")
        print("2. Check dblof_scores.csv for detailed session-level predictions")
        print("3. Analyze top_dblof_anomalies.csv for highest risk sessions")
        print("4. Examine dblof_user_risk_summary.csv for user-level risk assessment")
        
    except FileNotFoundError as e:
        print(f"Error: Required file not found - {e}")
        print("Please ensure all data files are in the correct locations.")
        sys.exit(1)
    except Exception as e:
        print(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()