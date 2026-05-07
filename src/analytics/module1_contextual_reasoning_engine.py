# Module 1: Contextual Reasoning Engine (CRE)
# Leverages LLMs with RAG to provide intent-level interpretations of user activities

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import faiss
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

@dataclass
class UserContext:
    """Represents contextual information about a user"""
    user_id: str
    role: str
    department: str
    project: str
    access_level: str
    temporal_privileges: Dict[str, List[str]]
    recent_activities: List[Dict]

@dataclass
class ActivityIntent:
    """Represents interpreted intent of user activity"""
    activity_id: str
    user_id: str
    action: str
    timestamp: datetime
    intent_score: float
    intent_label: str
    reasoning: str
    risk_level: str


class ContextualReasoningEngine:
    """
    Contextual Reasoning Engine (CRE) using RAG for intent-level analysis
    
    How it works:
    1. Builds semantic knowledge graph of user roles, operations, justifications
    2. Uses sentence embeddings to match activities to historical patterns
    3. Employs RAG pipeline to retrieve relevant context and generate reasoning
    4. Distinguishes benign deviations from malicious intent
    """
    
    def __init__(self, knowledge_base_path: Optional[str] = None):
        """
        Initialize CRE with embedding model and knowledge base
        
        Args:
            knowledge_base_path: Path to pre-built knowledge base (optional)
        """
        # Initialize sentence transformer for semantic embeddings
        print("Initializing Contextual Reasoning Engine...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = 384
        
        # Initialize FAISS index for fast semantic search
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Knowledge base: historical activities with justifications
        self.knowledge_base = []
        self.activity_embeddings = []
        
        # Context mappings
        self.role_permissions = {}
        self.project_lifecycles = {}
        self.temporal_access_rules = {}
        
        # Intent classification thresholds
        self.intent_thresholds = {
            'benign': 0.7,
            'suspicious': 0.4,
            'malicious': 0.2
        }
        
        if knowledge_base_path:
            self.load_knowledge_base(knowledge_base_path)
        
        print("CRE initialized successfully")
    
    def build_knowledge_base(self, historical_data: pd.DataFrame):
        """
        Build knowledge base from historical activity data
        
        Args:
            historical_data: DataFrame with columns:
                - user_id, role, action, timestamp, justification, outcome
        """
        print("Building knowledge base from historical data...")
        
        self.knowledge_base = []
        embeddings_list = []
        
        for idx, row in historical_data.iterrows():
            # Create semantic representation of activity
            activity_text = (
                f"User role: {row['role']}. "
                f"Action: {row['action']}. "
                f"Time: {row.get('time_of_day', 'business_hours')}. "
                f"Justification: {row.get('justification', 'routine_work')}. "
                f"Project: {row.get('project', 'general')}."
            )
            
            # Generate embedding
            embedding = self.embedding_model.encode(activity_text)
            embeddings_list.append(embedding)
            
            # Store in knowledge base
            self.knowledge_base.append({
                'activity_id': f"kb_{idx}",
                'user_id': row['user_id'],
                'role': row['role'],
                'action': row['action'],
                'justification': row.get('justification', ''),
                'outcome': row.get('outcome', 'benign'),
                'embedding': embedding
            })
        
        # Build FAISS index
        embeddings_array = np.array(embeddings_list).astype('float32')
        self.index.add(embeddings_array)
        self.activity_embeddings = embeddings_array
        
        print(f"Knowledge base built with {len(self.knowledge_base)} activities")
    
    def register_role_permissions(self, role_permissions: Dict[str, List[str]]):
        """
        Register expected permissions for each role

        Args:
            role_permissions: Dict mapping roles to allowed actions
        """
        role_permissions = {
        'Engineer': ['file_access', 'build_deploy', 'code_commit', 'database_query'],
        'Manager': ['user_management', 'approve_budget', 'email_send', 'team_management'],
        'Admin': ['create_user', 'delete_user', 'system_config'],
        'Analyst': ['run_report', 'data_export', 'database_query']
        }
        # cre.register_role_permissions(role_permissions)

        self.role_permissions = role_permissions
        print(f"Registered permissions for {len(role_permissions)} roles")
    
    def register_project_lifecycle(self, project_lifecycles: Dict[str, Dict]):
        """
        Register project lifecycle stages and expected activities
        
        Args:
            project_lifecycles: Dict mapping projects to lifecycle metadata
        """
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
        # cre.register_project_lifecycle(project_lifecycles)

        self.project_lifecycles = project_lifecycles
        print(f"Registered {len(project_lifecycles)} project lifecycles")
    
    def register_temporal_rules(self, temporal_rules: Dict[str, Dict]):
        """
        Register time-based access rules
        
        Args:
            temporal_rules: Dict mapping roles to time-based permissions
        """
        temporal_rules = {
            'Engineer': {'business_hours': (9, 18)},
            'Manager': {'business_hours': (8, 19)},
            'Admin': {'business_hours': (9, 17)},
            'Analyst': {'business_hours': (9, 17)},
        }
        # cre.register_temporal_rules(temporal_rules)

        self.temporal_access_rules = temporal_rules
        print(f"Registered temporal access rules for {len(temporal_rules)} roles")
    
    def retrieve_similar_activities(self, activity: Dict, k: int = 5) -> List[Dict]:
        """
        Retrieve k most similar historical activities using RAG
        
        Args:
            activity: Current activity to analyze
            k: Number of similar activities to retrieve
            
        Returns:
            List of similar activities with relevance scores
        """
        # Create semantic representation
        activity_text = (
            f"User role: {activity.get('role', 'unknown')}. "
            f"Action: {activity.get('action', 'unknown')}. "
            f"Time: {activity.get('time_of_day', 'unknown')}. "
            f"Project: {activity.get('project', 'unknown')}."
        )
        
        # Generate embedding
        query_embedding = self.embedding_model.encode(activity_text).reshape(1, -1).astype('float32')
        
        # Search in FAISS index
        if self.index.ntotal == 0:
            return []
        
        distances, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        
        # Retrieve similar activities
        similar_activities = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.knowledge_base):
                similar_act = self.knowledge_base[idx].copy()
                similar_act['similarity_score'] = float(1 / (1 + dist))  # Convert distance to similarity
                similar_activities.append(similar_act)
        
        return similar_activities
    
    def analyze_intent(self, activity: Dict, user_context: UserContext) -> ActivityIntent:
        """
        Analyze activity intent using contextual reasoning
        
        Args:
            activity: Activity to analyze
            user_context: User's contextual information
            
        Returns:
            ActivityIntent object with reasoning
        """
        # Retrieve similar historical activities
        similar_activities = self.retrieve_similar_activities(activity, k=5)
        
        # Check role-based permissions
        role_match = self._check_role_permission(activity, user_context)
        
        # Check temporal appropriateness
        temporal_match = self._check_temporal_rules(activity, user_context)
        
        # Check project lifecycle alignment
        project_match = self._check_project_lifecycle(activity, user_context)
        
        # Calculate intent score (0-1, lower = more suspicious)
        intent_score = self._calculate_intent_score(
            role_match, temporal_match, project_match, similar_activities
        )
        print(f"[DEBUG] role_match: {role_match}, temporal_match: {temporal_match}, project_match: {project_match}, similar_activities: {len(similar_activities)}")
        # print(f"[DEBUG] intent_score: {intent_score:.3f}, label: {intent_label}")

        # Determine intent label and risk
        if intent_score >= self.intent_thresholds['benign']:
            intent_label = "benign"
            risk_level = "low"
        elif intent_score >= self.intent_thresholds['suspicious']:
            intent_label = "suspicious"
            risk_level = "medium"
        else:
            intent_label = "malicious"
            risk_level = "high"
        
        # Generate human-readable reasoning
        reasoning = self._generate_reasoning(
            activity, user_context, role_match, temporal_match, 
            project_match, similar_activities, intent_score
        )
        
        return ActivityIntent(
            activity_id=activity.get('activity_id', 'unknown'),
            user_id=user_context.user_id,
            action=activity.get('action', 'unknown'),
            timestamp=activity.get('timestamp', datetime.now()),
            intent_score=intent_score,
            intent_label=intent_label,
            reasoning=reasoning,
            risk_level=risk_level
        )
    
    def _check_role_permission(self, activity: Dict, user_context: UserContext) -> float:
        """Check if action is permitted for user's role (0-1 score)"""
        role = user_context.role
        action = activity.get('action', '')
        
        if role not in self.role_permissions:
            return 0.5  # Unknown role, neutral score
        
        allowed_actions = self.role_permissions[role]
        if action in allowed_actions:
            return 1.0
        elif any(allowed in action for allowed in allowed_actions):
            return 0.7  # Partial match
        else:
            return 0.2  # Not in allowed actions
    
    from datetime import datetime

    def _check_temporal_rules(self, activity: Dict, user_context: UserContext) -> float:
        """Check if action is temporally appropriate (0-1 score)"""
    
        timestamp = activity.get('timestamp', None)
    
        # Safely parse string timestamp or use current time fallback
        if timestamp is None:
            timestamp = datetime.now()
        elif isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp)
            except ValueError:
                timestamp = datetime.now()
    
        hour = timestamp.hour
        weekday = timestamp.weekday()
    
        role = user_context.role
        if role not in self.temporal_access_rules:
            return 0.8  # No specific rules, assume reasonable
    
        rules = self.temporal_access_rules[role]
    
    # Check business hours
        if 'business_hours' in rules:
            start, end = rules['business_hours']
            if start <= hour <= end and weekday < 5:
                return 1.0
            elif weekday < 5:
                return 0.5  # Work day but off hours
            else:
                return 0.3  # Weekend
    
        return 0.8

    
    def _check_project_lifecycle(self, activity: Dict, user_context: UserContext) -> float:
        """Check if action aligns with project lifecycle stage (0-1 score)"""
        project = user_context.project
        action = activity.get('action', '')
        
        if project not in self.project_lifecycles:
            return 0.7  # Unknown project, neutral-positive
        
        lifecycle_stage = self.project_lifecycles[project].get('stage', 'active')
        expected_actions = self.project_lifecycles[project].get('expected_actions', [])
        
        if action in expected_actions:
            return 1.0
        elif lifecycle_stage == 'completed' and 'access' in action:
            return 0.4  # Accessing completed project
        else:
            return 0.6
    
    def _calculate_intent_score(self, role_match: float, temporal_match: float,
                                project_match: float, similar_activities: List[Dict]) -> float:
        """
        Calculate overall intent score combining multiple factors
        """
        # Base score from contextual checks
        context_score = (role_match * 0.4 + temporal_match * 0.3 + project_match * 0.3)
        
        # Adjust based on similar historical activities
        if similar_activities:
            historical_outcomes = [
                1.0 if act.get('outcome') == 'benign' else 0.0
                for act in similar_activities
            ]
            historical_score = np.mean(historical_outcomes) if historical_outcomes else 0.5
            
            # Weight by similarity scores
            similarities = [act.get('similarity_score', 0.5) for act in similar_activities]
            weighted_historical = np.average(historical_outcomes, weights=similarities)
            
            # Combine context and historical (70-30 split)
            final_score = context_score * 0.7 + weighted_historical * 0.3
        else:
            final_score = context_score
        
        return np.clip(final_score, 0.0, 1.0)
    
    def _generate_reasoning(self, activity: Dict, user_context: UserContext,
                          role_match: float, temporal_match: float, project_match: float,
                          similar_activities: List[Dict], intent_score: float) -> str:
        """
        Generate human-readable reasoning for the intent classification
        """
        reasoning_parts = []
        
        # Role-based reasoning
        if role_match < 0.5:
            reasoning_parts.append(
                f"Action '{activity.get('action')}' is NOT typically permitted for role '{user_context.role}'"
            )
        elif role_match >= 0.8:
            reasoning_parts.append(
                f"Action aligns with expected permissions for role '{user_context.role}'"
            )
        
        # Temporal reasoning
        timestamp = activity.get('timestamp', datetime.now())
        if temporal_match < 0.5:
            reasoning_parts.append(
                f"Activity occurred at {timestamp.strftime('%H:%M on %A')}, outside typical business hours"
            )
        
        # Project lifecycle reasoning
        if project_match < 0.5:
            reasoning_parts.append(
                f"Action unusual for project '{user_context.project}' current lifecycle stage"
            )
        
        # Historical pattern reasoning
        if similar_activities:
            benign_count = sum(1 for act in similar_activities if act.get('outcome') == 'benign')
            reasoning_parts.append(
                f"Similar activities in history: {benign_count}/{len(similar_activities)} were benign"
            )
        
        # Overall assessment
        if intent_score >= 0.7:
            assessment = "appears BENIGN and contextually justified"
        elif intent_score >= 0.4:
            assessment = "appears SUSPICIOUS and warrants review"
        else:
            assessment = "appears MALICIOUS with high risk indicators"
        
        reasoning_parts.append(f"Overall assessment: Activity {assessment}")
        
        return ". ".join(reasoning_parts) + "."
    
    def save_knowledge_base(self, filepath: str):
        """Save knowledge base to disk"""
        data = {
            'knowledge_base': self.knowledge_base,
            'role_permissions': self.role_permissions,
            'project_lifecycles': self.project_lifecycles,
            'temporal_access_rules': self.temporal_access_rules
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, default=str, indent=2)
        
        # Save FAISS index
        faiss.write_index(self.index, filepath + '.faiss')
        print(f"Knowledge base saved to {filepath}")
    
    def load_knowledge_base(self, filepath: str):
        """Load knowledge base from disk"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.knowledge_base = data['knowledge_base']
        self.role_permissions = data['role_permissions']
        self.project_lifecycles = data['project_lifecycles']
        self.temporal_access_rules = data['temporal_access_rules']
        
        # Load FAISS index
        self.index = faiss.read_index(filepath + '.faiss')
        print(f"Knowledge base loaded from {filepath}")


# Example usage demonstration
if __name__ == "__main__":
    print("=" * 80)
    print("MODULE 1: CONTEXTUAL REASONING ENGINE (CRE)")
    print("=" * 80)
    
    # Initialize CRE
    cre = ContextualReasoningEngine()
    
    # Build sample knowledge base
    historical_data = pd.DataFrame({
        'user_id': ['user001', 'user002', 'user003', 'user004', 'user005'],
        'role': ['Engineer', 'Manager', 'Analyst', 'Admin', 'Engineer'],
        'action': ['file_access', 'email_send', 'database_query', 'system_config', 'code_commit'],
        'justification': ['project_work', 'team_communication', 'data_analysis', 'maintenance', 'feature_development'],
        'outcome': ['benign', 'benign', 'benign', 'malicious', 'benign'],
        'project': ['Alpha', 'Beta', 'Delta', 'Gamma', 'Alpha']
    })
    cre.build_knowledge_base(historical_data)
    
    # Register role permissions
    role_permissions = {
        'engineer': ['file_access', 'code_commit', 'build_deploy'],
        'manager': ['email_send', 'file_access', 'team_management'],
        'analyst': ['database_query', 'report_generate', 'data_export']
    }
    cre.register_role_permissions(role_permissions)
    
    # Register temporal rules
    temporal_rules = {
        'engineer': {'business_hours': (9, 18)},
        'manager': {'business_hours': (8, 19)},
        'analyst': {'business_hours': (9, 17)}
    }
    cre.register_temporal_rules(temporal_rules)
    
    # Analyze a suspicious activity
    suspicious_activity = {
        'activity_id': 'act_12345',
        'action': 'database_export',
        'timestamp': datetime(2025, 10, 21, 23, 30),  # Late night
        'project': 'project_alpha'
    }
    
    user_context = UserContext(
        user_id='user001',
        role='engineer',
        department='engineering',
        project='project_alpha',
        access_level='standard',
        temporal_privileges={},
        recent_activities=[]
    )
    
    # Analyze intent
    intent_result = cre.analyze_intent(suspicious_activity, user_context)
    
    print("\n" + "=" * 80)
    print("INTENT ANALYSIS RESULT:")
    print("=" * 80)
    print(f"Activity ID: {intent_result.activity_id}")
    print(f"User: {intent_result.user_id}")
    print(f"Action: {intent_result.action}")
    print(f"Intent Score: {intent_result.intent_score:.3f}")
    print(f"Intent Label: {intent_result.intent_label}")
    print(f"Risk Level: {intent_result.risk_level}")
    print(f"\nReasoning:\n{intent_result.reasoning}")
    print("=" * 80)
