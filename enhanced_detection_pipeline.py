# enhanced_detection_pipeline.py
import pandas as pd
import os
import sys
from src.analytics.dblof_ueba_module import DBLOFUEBAModule
from src.analytics.module1_contextual_reasoning_engine import ContextualReasoningEngine, UserContext
from src.analytics.module2_unified_threat_graph import UnifiedThreatCorrelationGraph
from src.analytics.module3_continual_learning import ContinualLearningPipeline
from src.utils.date_utils import extract_and_convert_dates_to_numeric
sys.path.append(os.path.join(os.path.dirname(__file__), 'tests'))
from tests.generate_synthetic_api_logs import generate_synthetic_api_logs
from datetime import timedelta

api_logs_df = generate_synthetic_api_logs(num_entries=50)

def build_user_context(user_id, user_dim_df):
    row = user_dim_df[user_dim_df['user_id'] == user_id]
    if row.empty:
        print(f"Warning: No user context found for user_id {user_id}")
        return UserContext(
            user_id=user_id,
            role='unknown',
            department='unknown',
            access_level='employee',
            project='unknown',
            temporal_privileges={},
            recent_activities=[]
        )
    row = row.iloc[0]
    print(f"User {user_id} role: {row['role']}, project: {row.get('project', 'unknown')}")
    return UserContext(
        user_id=row['user_id'],
        role=row['role'],
        department=row['department'],
        access_level='manager' if row['is_manager'] else 'employee',
        project=row.get('project', 'unknown'),  # update as per real project info if available
        temporal_privileges={},  # extend as needed
        recent_activities=[]
    )

def main():
    suspicious_results = []
    # Load user dimensions CSV for contextual data
    user_dim_df = pd.read_csv('tests/generated_user_dimensions.csv')
    print(user_dim_df.columns)
    user_dim_df.rename(columns={
    # 'userid': 'user_id',
    'manager_level': 'is_manager',
    'hire date': 'hire_date',
    'tenure days': 'tenure_days'
    }, inplace=True)

    # Load session features and CERT logs
    session_features = pd.read_csv('cert_dblof_analysis_output/session_features.csv')
    cert_data = pd.read_csv('data/processed_data/cert_batch_processed.csv')
    cert_data['timestamp_numeric'] = cert_data['timestamp'].apply(
    lambda x: extract_and_convert_dates_to_numeric(str(x))[0] if not pd.isna(x) and len(extract_and_convert_dates_to_numeric(str(x))) > 0 else None
)
    
    def infer_action_from_row(row):
        print(f"Inferring action for session_id {row.get('session_id', '')}")
        print(f" activity_subtype: {row.get('activity_subtype', '')}")
        print(f" email_content: {str(row.get('email_content', ''))[:50]}")
        print(f" file_name: {row.get('file_name', '')}")
        print(f" logon_activity: {row.get('logon_activity', '')}")
        print(f" http_content: {row.get('http_content', '')}")
        print(f" device_activity: {row.get('device_activity', '')}")
        
        if 'email' in str(row.get('activity_subtype', '')).lower() or row.get('email_content', ''):
            print(" Inferred action: email_send")
            return 'email_send'
        if row.get('file_name', '') or row.get('file_tree', ''):
            print(" Inferred action: file_access")
            return 'file_access'
        if row.get('logon_activity', ''):
            la = str(row['logon_activity']).lower()
            if 'login' in la:
                print(" Inferred action: logon")
                return 'logon'
            if 'logout' in la:
                print(" Inferred action: logout")
                return 'logout'
        if row.get('http_content', '') or row.get('url', ''):
            if 'query' in str(row.get('http_content', '')).lower() or 'query' in str(row.get('url', '')).lower():
                print(" Inferred action: database_query")
                return 'database_query'
        if row.get('device_activity', ''):
            da = str(row['device_activity']).lower()
            if 'removable' in da:
                print(" Inferred action: device_use")
                return 'device_use'
            if 'insert' in da:
                print(" Inferred action: device_insert")
                return 'device_insert'
        print(" Inferred action: unknown")
        return 'unknown'




    role_permissions = {
    'Engineer': ['file_access', 'build_deploy', 'code_commit', 'database_query'],
    'Manager': ['user_management', 'approve_budget', 'email_send', 'team_management'],
    'Admin': ['create_user', 'delete_user', 'system_config'],
    'Analyst': ['run_report', 'data_export', 'database_query']
    }

    temporal_rules = {
    'Engineer': {'business_hours': (9, 18)},
    'Manager': {'business_hours': (8, 19)},
    'Admin': {'business_hours': (9, 17)},
    'Analyst': {'business_hours': (9, 17)},
    }

    project_lifecycles = {
        'Alpha': {
            'stage': 'active',
            'expected_actions': ['database_export', 'code_commit', 'run_report']
        },
        'Beta': {
            'stage': 'active',
            'expected_actions': ['file_access', 'team_management']
        },
        'Gamma': {
            'stage': 'completed',
            'expected_actions': []
        },
        'Delta': {
            'stage': 'active',
            'expected_actions': ['database_query', 'data_export']
        },
        'Epsilon': {
            'stage': 'active',
            'expected_actions': ['email_send', 'file_access']
        }
    }

    # 1. Run existing DBLOF anomaly detection
    print('Running DBLOF anomaly detection...')
    dblof = DBLOFUEBAModule(session_features)
    dblof.fit_dblof(contamination=0.10)
    scores_df = dblof.scores_df

    # 2. Initialize Contextual Reasoning Engine and prepare knowledge base (assumed prebuilt)
    cre = ContextualReasoningEngine()
    # cre.build_knowledge_base(historical_data_df)  # Provide historical labeled activities
    cre.register_role_permissions(role_permissions)
    cre.register_temporal_rules(temporal_rules)
    cre.register_project_lifecycle(project_lifecycles)

    # 3. Initialize Unified Threat Correlation Graph
    utcg = UnifiedThreatCorrelationGraph()
    utcg.build_from_cert_data(cert_data,user_dim_df)  # optionally add HR, stress data
    print(f"Nodes in UTCG: {list(utcg.graph.nodes)[:50]}")
    # user_id = 'U003'
    # print(utcg.graph.neighbors(user_id))
    # print(utcg.detect_cross_domain_patterns(user_id))

    # 4. Initialize Continual Learning Pipeline and fit initial baseline
    clp = ContinualLearningPipeline()
    clp.fit_initial(session_features)

    # 5. Analyze each flagged anomaly enriched with context and graph info
    flagged = scores_df[scores_df['is_outlier']]
    print(f'\nAnalyzing {len(flagged)} flagged sessions...')
    
    # print(flagged[['session_id', 'activity_subtype', 'file_name', 'logon_activity', 'email_content']].head(10))
    flagged = flagged.merge(session_features[['user_id', 'session_id', 'session_start']],
                        on=['user_id', 'session_id'], how='left')

    for idx, row in flagged.iterrows():
        user_context = build_user_context(row['user_id'], user_dim_df)
        session_start = row.get('session_start')

        if session_start is None or pd.isna(session_start):
            print(f"Skipping session with missing session_start: {row['session_id']}")
            action = 'unknown'
        else:
            session_start = pd.to_datetime(session_start)

            matching_cert_rows = cert_data[
                (cert_data['user_id'] == row['user_id']) &
                (pd.to_datetime(cert_data['timestamp']) >= session_start - pd.Timedelta(minutes=10)) &
                (pd.to_datetime(cert_data['timestamp']) <= session_start + pd.Timedelta(minutes=10))
            ]

            if not matching_cert_rows.empty:
                cert_row = matching_cert_rows.iloc[0]
                action = infer_action_from_row(cert_row)
            else:
                print(f"No cert log found near timestamp for session_id {row['session_id']}")
                action = 'unknown'

        activity = {
            'activity_id': row['session_id'],
            'action': action,
            'timestamp': session_start,
            'project': user_context.project
        }

        intent = cre.analyze_intent(activity, user_context)
   
        print(f"User role: {user_context.role}")
        print(f"Project: {user_context.project}")
        print(f"Action used for intent: {action}")
        print(f"User role: {user_context.role}, Project: {user_context.project}, Action: {action}")
        print(f"Intent Score: {intent.intent_score:.3f}, Label: {intent.intent_label}")
        print(f"cross_domain_patterns: {utcg.detect_cross_domain_patterns(row['user_id'])}")

        suspicious_results.append({
            'user_id': row['user_id'],
            'session_id': row['session_id'],
            'intent_label': intent.intent_label,
            'assessment': intent.reasoning,
        })

    print("Sample flagged session_ids:", flagged.head(10))
    print("Sample cert_data session_ids:", cert_data['session_id'].head(10).tolist())

    print(f"Cert log columns: {list(cert_data.columns)}")    
    df_results = pd.DataFrame(suspicious_results)
    df_results.to_csv('suspicious_activity_report.csv', index=False)
    print("Suspicious activity report saved to suspicious_activity_report.csv")        

if __name__ == '__main__':
    main()

