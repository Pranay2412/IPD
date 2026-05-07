import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, SAGEConv
    from torch_geometric.data import Data
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch Geometric not available. GNN features will be limited.")


@dataclass
class ThreatEntity:
    entity_id: str
    entity_type: str  # user, system, asset, communication, hr_record
    attributes: Dict
    risk_score: float = 0.0


@dataclass
class ThreatRelation:
    source_id: str
    target_id: str
    relation_type: str
    weight: float
    anomaly_score: float
    timestamp: str
    metadata: Dict


class UnifiedThreatCorrelationGraph:
    def __init__(self, use_gnn: bool = True):
        print("Initializing Unified Threat Correlation Graph...")
        self.graph = nx.DiGraph()
        self.entities: Dict[str, ThreatEntity] = {}
        self.entity_types = {'user': set(), 'system': set(), 'asset': set(), 'communication': set(), 'hr_record': set()}
        self.relations: List[ThreatRelation] = []
        self.use_gnn = use_gnn and TORCH_AVAILABLE
        self.gnn_model = None
        self.detected_patterns = []
        self.feature_extractors = {
            'user': self._extract_user_features,
            'system': self._extract_system_features,
            'asset': self._extract_asset_features,
            'communication': self._extract_communication_features,
            'hr_record': self._extract_hr_features
        }
        print("UTCG initialized successfully")

    def _extract_user_features(self, entity_id: str) -> np.ndarray:
        entity = self.entities.get(entity_id)
        if entity is None:
            return np.zeros(4)
        return np.array([
            entity.attributes.get('activity_count', 0) / 1000,
            entity.attributes.get('stress_level', 0.0),
            entity.risk_score,
            len(list(self.graph.neighbors(entity_id))) / 100
        ])

    def _extract_system_features(self, entity_id: str) -> np.ndarray:
        in_degree = self.graph.in_degree(entity_id)
        return np.array([
            in_degree / 100,
            0.0,
            0.0,
            0.0
        ])

    def _extract_asset_features(self, entity_id: str) -> np.ndarray:
        in_degree = self.graph.in_degree(entity_id)
        return np.array([
            in_degree / 50,
            0.5,  # Placeholder sensitivity
            0.0,
            0.0
        ])

    def _extract_communication_features(self, entity_id: str) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0, 0.0])

    def _extract_hr_features(self, entity_id: str) -> np.ndarray:
        entity = self.entities.get(entity_id)
        if entity is None:
            return np.zeros(4)
        return np.array([
            entity.attributes.get('conflicts', 0) / 10,
            entity.attributes.get('termination_risk', 0.0),
            entity.risk_score,
            0.0
        ])

    def add_entity(self, entity: ThreatEntity):
        self.entities[entity.entity_id] = entity
        self.entity_types[entity.entity_type].add(entity.entity_id)
        self.graph.add_node(
            entity.entity_id,
            entity_type=entity.entity_type,
            risk_score=entity.risk_score,
            **entity.attributes
        )

    def add_relation(self, relation: ThreatRelation):
        self.relations.append(relation)
        self.graph.add_edge(
            relation.source_id,
            relation.target_id,
            relation_type=relation.relation_type,
            weight=relation.weight,
            anomaly_score=relation.anomaly_score,
            timestamp=relation.timestamp,
            **relation.metadata
        )

    def build_from_cert_data(self, cert_df: pd.DataFrame,
                             hr_df: Optional[pd.DataFrame] = None,
                             stress_scores: Optional[Dict[str, float]] = None):
        print("Building UTCG from data sources...")

        unique_users = cert_df['user_id'].unique()
        for user_id in unique_users:
            user_activities = cert_df[cert_df['user_id'] == user_id]
            entity = ThreatEntity(
                entity_id=user_id,
                entity_type='user',
                attributes={
                    'activity_count': len(user_activities),
                    'stress_level': stress_scores.get(user_id, 0.0) if stress_scores else 0.0,
                    'departments': list(user_activities['department'].unique()) if 'department' in user_activities else []
                },
                risk_score=0.0
            )
            self.add_entity(entity)

        unique_pcs = cert_df['pc_id'].unique()
        for pc_id in unique_pcs:
            pc_entity = ThreatEntity(
                entity_id=pc_id,
                entity_type='asset',
                attributes={
                    'asset_type': 'PC',
                },
                risk_score=0.0
            )
            self.add_entity(pc_entity)

        for idx, row in cert_df.iterrows():
            relation = ThreatRelation(
                source_id=row['user_id'],
                target_id=row['pc_id'],
                relation_type='uses_pc',
                weight=1.0,
                anomaly_score=row.get('anomaly_score', 0.0),
                timestamp=str(row.get('session_start', pd.Timestamp.now())),
                metadata={'endpoint': row.get('endpoint', '')}
            )
            self.add_relation(relation)

        if hr_df is not None and not hr_df.empty:
            hr_df_renamed = hr_df.rename(columns={
                'userid': 'user_id',
                'manager_level': 'is_manager',
                'hire date': 'hire_date',
                'tenure days': 'tenure_days'
            })
            for idx, row in hr_df_renamed.iterrows():
                entity = ThreatEntity(
                    entity_id=f"hr_{row['user_id']}",
                    entity_type='hr_record',
                    attributes={
                        'role': row.get('role', ''),
                        'department': row.get('department', ''),
                        'is_manager': row.get('is_manager', False),
                        'hire_date': row.get('hire_date', ''),
                        'tenure_days': row.get('tenure_days', 0),
                    },
                    risk_score=0.0
                )
                self.add_entity(entity)
                relation = ThreatRelation(
                    source_id=row['user_id'],
                    target_id=f"hr_{row['user_id']}",
                    relation_type='has_hr_record',
                    weight=1.0,
                    anomaly_score=0.0,
                    timestamp=str(pd.Timestamp.now()),
                    metadata={}
                )
                self.add_relation(relation)

        print(f"UTCG built: {len(self.entities)} entities, {len(self.relations)} relations")

    def detect_cross_domain_patterns(self, user_id: str) -> Dict:
        if user_id not in self.graph:
            return {
                "pattern_count": 0,
                "entity_types": [],
                "connected_entities": [],
                "overall_risk_score": 0.0
            }

        neighbors = list(self.graph.neighbors(user_id))
        connected_entities = []
        entity_types = set()
        risk_scores = []

        for neighbor in neighbors:
            node_data = self.graph.nodes[neighbor]
            entity_type = node_data.get("entity_type", "unknown")
            entity_types.add(entity_type)
            risk_scores.append(node_data.get("risk_score", 0.0))
            connected_entities.append({
                "entity_id": neighbor,
                "entity_type": entity_type,
                "risk_score": node_data.get("risk_score", 0.0)
            })

        pattern_count = len(entity_types)
        if not risk_scores:
            overall_risk = 0.0
        else:
            overall_risk = np.mean(risk_scores) + 0.05 * len(neighbors)

        if pattern_count >= 3:
            overall_risk += 0.3
        elif pattern_count == 2:
            overall_risk += 0.1

        overall_risk_score = min(1.0, round(overall_risk, 3))

        result = {
            "pattern_count": pattern_count,
            "entity_types": list(entity_types),
            "connected_entities": connected_entities[:10],
            "overall_risk_score": overall_risk_score
        }
        return result

