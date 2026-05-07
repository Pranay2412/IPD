#!/usr/bin/env python3
"""
DBLOF Performance Evaluation Script (Updated)
Reflects actual contamination used for scoring and ignores old optimization grid.
Calculates Precision, Recall, F1-Score and compares with research paper.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import os

# Known insider user IDs from the CERT dataset
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


def load_dblof_results(results_dir):
    """Load DBLOF results from the analysis"""
    scores_path = os.path.join(results_dir, "dblof_scores.csv")
    eval_path = os.path.join(results_dir, "dblof_evaluation_results.csv")

    if not os.path.exists(scores_path) or not os.path.exists(eval_path):
        raise FileNotFoundError("Required DBLOF output files not found. Run pipeline first.")

    scores_df = pd.read_csv(scores_path)
    eval_df = pd.read_csv(eval_path)
    
    return scores_df, eval_df


def calculate_performance_metrics(scores_df):
    """Calculate precision, recall, F1-score based on scoring results"""
    scores_df['true_label'] = scores_df['user_id'].apply(
        lambda uid: 1 if uid in INSIDER_USER_IDS else 0
    )
    
    y_true = scores_df['true_label'].values
    y_pred = scores_df['is_outlier'].astype(int).values
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return accuracy, precision, recall, f1, tp, fp, fn, tn


def print_performance_report(metrics, actual_contamination):
    accuracy, precision, recall, f1, tp, fp, fn, tn = metrics

    print("=" * 80)
    print("DBLOF INSIDER THREAT DETECTION PERFORMANCE EVALUATION")
    print("=" * 80)
    print(f"Contamination used for scoring: {actual_contamination}")
    print("-" * 40)

    print("=== CONFUSION MATRIX ===")
    print(f"TP (Insiders flagged):            {tp}")
    print(f"FP (Benign flagged):              {fp}")
    print(f"FN (Insiders missed):             {fn}")
    print(f"TN (Benign not flagged):          {tn}")
    print("=" * 25)
    print(f"Total sessions analyzed:          {tp+fp+fn+tn}")
    print("\n=== DBLOF ACCURACY METRICS ===")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    print("=" * 80)


def analyze_top_detected_insiders(scores_df, top_n=10):
    insider_df = scores_df[scores_df['user_id'].isin(INSIDER_USER_IDS)]
    user_stats = insider_df.groupby('user_id').agg(
        total_sessions=('session_id','count'),
        detected_sessions=('is_outlier','sum'),
        max_anomaly_score=('anomaly_score','max')
    )
    user_stats['detection_rate'] = (
        user_stats['detected_sessions'] / user_stats['total_sessions']
    )
    print(f"\n=== TOP {top_n} DETECTED INSIDER USERS ===")
    print(user_stats.sort_values(
        ['detection_rate','detected_sessions'], ascending=False
    ).head(top_n))
    print("=" * 80)


def main():
    results_dir = 'cert_dblof_analysis_output'
    
    try:
        scores_df, eval_df = load_dblof_results(results_dir)

        # Get actual contamination used
        actual_contamination = 0.10 

        metrics = calculate_performance_metrics(scores_df)
        print_performance_report(metrics, actual_contamination)

        analyze_top_detected_insiders(scores_df)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run the DBLOF analysis pipeline first.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

if __name__ == '__main__':
    main()