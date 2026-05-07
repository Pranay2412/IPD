"""
ueba_module.py
Enhanced UEBA scoring using LOF with variance filtering,
skew normalisation and adaptive contamination.
----------------------------------------------------------------------
Dependencies:
    pandas
    numpy
    scikit-learn
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class UEBAModule:
    def __init__(self, session_features: pd.DataFrame):
        self.sf = session_features.copy()
        self.feature_cols: list[str] = []
        self.scores_df: pd.DataFrame | None = None

    # ------------------------------------------------------------------ #
    # 1. feature matrix construction
    # ------------------------------------------------------------------ #
    def _build_feature_matrix(self) -> pd.DataFrame:
        numeric = self.sf.select_dtypes(include=[np.number]).copy()

        # drop metadata columns
        drop_cols = {"session_start", "session_end"}
        numeric.drop(columns=[c for c in drop_cols if c in numeric.columns], inplace=True)

        # remove near-constant cols
        selector = VarianceThreshold(0.01)
        matrix = selector.fit_transform(numeric)
        kept = numeric.columns[selector.get_support()]
        feature_matrix = pd.DataFrame(matrix, columns=kept)

        # log-normalise skewed features
        skewed = feature_matrix.columns[feature_matrix.skew().abs() > 1]
        feature_matrix[skewed] = np.log1p(feature_matrix[skewed])

        # ensure no NaN values
        imputer = SimpleImputer(strategy="median")
        feature_matrix[feature_matrix.columns] = imputer.fit_transform(feature_matrix)

        self.feature_cols = feature_matrix.columns.tolist()
        return feature_matrix

    # ------------------------------------------------------------------ #
    # 2. LOF training & scoring
    # ------------------------------------------------------------------ #
    def fit_lof(self, contamination: str | float = "auto", n_neighbors: int | None = None):
        X = self._build_feature_matrix()

        if n_neighbors is None:
            n_neighbors = int(np.clip(np.sqrt(len(X)), 5, 50))

        if contamination == "auto":
            contamination = float(np.clip(50 / len(X), 0.005, 0.05))

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=False,
        )
        preds = lof.fit_predict(X_scaled)
        neg_factor = lof.negative_outlier_factor_

        # score dataframe
        scores = self.sf[["user_id", "session_id"]].copy()
        scores["lof_factor"] = neg_factor
        scores["is_outlier"] = preds == -1

        # scale anomaly score 0-100 (higher = worse)
        max_s, min_s = neg_factor.max(), neg_factor.min()
        scores["anomaly_score"] = 100.0 * (max_s - neg_factor) / (max_s - min_s)
        self.scores_df = scores

    # ------------------------------------------------------------------ #
    # 3. helpers
    # ------------------------------------------------------------------ #
    def top_anomalies(self, n: int = 20) -> pd.DataFrame:
        if self.scores_df is None:
            raise RuntimeError("Call fit_lof() first.")
        return self.scores_df.nlargest(n, "anomaly_score")

    def user_risk_summary(self) -> pd.DataFrame:
        if self.scores_df is None:
            raise RuntimeError("Call fit_lof() first.")
        grp = self.scores_df.groupby("user_id")
        summary = grp.agg(
            sessions=("session_id", "count"),
            outlier_sessions=("is_outlier", "sum"),
            avg_score=("anomaly_score", "mean"),
            max_score=("anomaly_score", "max"),
        )
        summary["outlier_rate"] = summary["outlier_sessions"] / summary["sessions"]
        return summary.sort_values("avg_score", ascending=False)

    def save_scores(self, path: str):
        if self.scores_df is None:
            raise RuntimeError("Call fit_lof() first.")
        self.scores_df.to_parquet(path, index=False)
