"""
feature_engineering.py
Enhanced feature-engineering pipeline for CERT insider-threat analytics
----------------------------------------------------------------------
Dependencies:
    pandas
    numpy
    scikit-learn  (KMeans)
"""

import hashlib
from itertools import islice
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


class CERTFeatureEngineer:
    """
    1.  Sessionise events ➜ session_id
    2.  Derive temporal, frequency, contextual, burst & rare-hour flags
    3.  Build dynamic behaviour-based peer groups
    4.  Add peer-normalised z-scores
    5.  Extract sequential n-gram & unusual-transition features
    6.  Return session-level feature matrix
    """

    # ------------------------------------------------------------------ #
    # initialisation & high-level runner
    # ------------------------------------------------------------------ #
    def __init__(self, events_df: pd.DataFrame, user_dim: pd.DataFrame | None = None):
        self.df = events_df.copy()
        self.user_dim = (
            user_dim.set_index("user_id") if user_dim is not None else pd.DataFrame()
        )
        self.session_features: pd.DataFrame | None = None
        self._global_transition_prob: dict[tuple[str, str], float] = {}

    def run_all(self) -> pd.DataFrame:
        "Main entry-point"
        self._sessionise()
        self._add_temporal_features()
        self._add_frequency_metrics()
        self._add_context()
        self._add_rare_hour_flag()
        self._add_burst_flags()
        self._create_global_transition_lookup()
        self._add_sequential_features()
        self._create_dynamic_peer_groups()
        self._add_peer_group_zscores()
        self._aggregate_session_features()
        return self.session_features

    # back-compat alias
    def run_feature_engineering(self):
        return self.run_all()

    # ------------------------------------------------------------------ #
    # 1. sessionisation
    # ------------------------------------------------------------------ #
    def _sessionise(self, gap_minutes: int = 10) -> None:
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])
        self.df.sort_values(["user_id", "timestamp"], inplace=True)
        self.df["prev_ts"] = self.df.groupby("user_id")["timestamp"].shift()
        gap = (self.df["timestamp"] - self.df["prev_ts"]).dt.total_seconds().div(60)
        date_change = self.df["timestamp"].dt.date.ne(
            self.df["prev_ts"].dt.date.fillna(self.df["timestamp"].dt.date)
        )
        split = gap.gt(gap_minutes) | gap.isna() | date_change
        self.df["session_counter"] = split.groupby(self.df["user_id"]).cumsum()
        self.df["session_id"] = (
            self.df["user_id"]
            .astype(str)
            .str.cat(self.df["session_counter"].astype(str), sep="_")
            .apply(lambda x: hashlib.md5(x.encode()).hexdigest()[:16])
        )
        self.df.drop(columns=["prev_ts", "session_counter"], inplace=True)

    # ------------------------------------------------------------------ #
    # 2. temporal features
    # ------------------------------------------------------------------ #
    def _add_temporal_features(self) -> None:
        g = self.df.groupby("session_id")["timestamp"]
        self.df["session_duration_s"] = g.transform(lambda s: (s.max() - s.min()).total_seconds())
        self.df["hour"] = self.df["timestamp"].dt.hour
        self.df["day_of_week"] = self.df["timestamp"].dt.dayofweek
        self.df["time_bin"] = pd.cut(
            self.df["hour"], [0, 6, 12, 18, 24], labels=["night", "morning", "afternoon", "evening"], right=False
        )
        # inter-arrival
        self.df["prev_ts"] = g.shift()
        self.df["interarrival_s"] = (
            self.df["timestamp"] - self.df["prev_ts"]
        ).dt.total_seconds().fillna(0)
        stats = self.df.groupby("session_id")["interarrival_s"].agg(
            interarrival_mean_s="mean", interarrival_std_s="std", interarrival_max_s="max"
        )
        self.df = self.df.merge(stats, left_on="session_id", right_index=True)
        self.df.drop(columns="prev_ts", inplace=True)

    # ------------------------------------------------------------------ #
    # 3. frequency metrics
    # ------------------------------------------------------------------ #
    def _add_frequency_metrics(self) -> None:
        acts = ["logon", "file", "device", "http", "email"]
        session_rows = []
        for sid, sess in self.df.groupby("session_id"):
            dur = max(sess["session_duration_s"].iloc[0], 1.0)
            row = {"session_id": sid, "event_cnt": len(sess)}
            for a in acts:
                cnt = (sess["activity_type"] == a).sum()
                row[f"{a}_cnt"] = cnt
                row[f"{a}_rate"] = cnt / dur
            session_rows.append(row)
        metrics = pd.DataFrame(session_rows)
        self.df = self.df.merge(metrics, on="session_id")

    # ------------------------------------------------------------------ #
    # 4. user-context enrichment
    # ------------------------------------------------------------------ #
    def _add_context(self) -> None:
        if self.user_dim.empty:
            return
        if 'hire_date' in self.user_dim.columns:
    # Ensure both sides are pandas Timestamps, so diff is timedelta64
            last_ts = self.df['timestamp'].max()
            hire_dt = pd.to_datetime(self.user_dim['hire_date'])
            self.user_dim['tenure_days'] = (last_ts - hire_dt).dt.days

        self.df = self.df.merge(self.user_dim, left_on="user_id", right_index=True, how="left")

    # ------------------------------------------------------------------ #
    # 5. rare hour & burst
    # ------------------------------------------------------------------ #
    def _add_rare_hour_flag(self) -> None:
        logons = self.df[self.df["activity_type"] == "logon"]
        hour_freq = logons["hour"].value_counts(normalize=True)
        self.rare_hours = hour_freq[hour_freq < 0.01].index
        self.df["rare_logon_hour"] = (
            (self.df["activity_type"] == "logon") & self.df["hour"].isin(self.rare_hours)
        ).astype(int)

    def _add_burst_flags(self, window: int = 3, burst_threshold: int = 3) -> None:
        f = self.df["activity_type"] == "file"
        rolling = (
            f.groupby([self.df["user_id"], self.df["session_id"]])
            .apply(lambda x: x.rolling(window, min_periods=1).sum())
            .reset_index(level=[0, 1], drop=True)
        )
        self.df["file_burst"] = (rolling >= burst_threshold).astype(int)

    # ------------------------------------------------------------------ #
    # 6. sequential features
    # ------------------------------------------------------------------ #
    def _create_global_transition_lookup(self) -> None:
        all_transitions = (
            self.df.sort_values("timestamp")
            .groupby("user_id")["activity_type"]
            .apply(lambda x: list(zip(x.iloc[:-1], x.iloc[1:])))
            .explode()
            .dropna()
        )
        counts = all_transitions.value_counts()
        probs = counts.div(counts.sum())
        self._global_transition_prob = probs.to_dict()

    def _add_sequential_features(self, max_n: int = 3) -> None:
        seq_rows = []
        for (uid, date), grp in self.df.assign(date=self.df["timestamp"].dt.date).groupby(["user_id", "date"]):
            seq = grp.sort_values("timestamp")["activity_type"].tolist()
            row = {
                "user_id": uid,
                "date": date,
                "seq_len": len(seq),
                "unusual_transitions": self._count_unusual_trans(seq),
            }
            # n-gram diversity
            for n in range(2, max_n + 1):
                ngrams = set(tuple(seq[i : i + n]) for i in range(len(seq) - n + 1))
                row[f"{n}gram_diversity"] = len(ngrams)
            seq_rows.append(row)
        seq_df = pd.DataFrame(seq_rows)
        # merge diversity back using left join on user_id & date
        self.df["date"] = self.df["timestamp"].dt.date
        self.df = self.df.merge(seq_df, on=["user_id", "date"], how="left")
        self.df.drop(columns="date", inplace=True)

    def _count_unusual_trans(self, seq: list[str]) -> int:
        transitions = [tuple(seq[i : i + 2]) for i in range(len(seq) - 1)]
        return sum(self._global_transition_prob.get(t, 0) < 0.01 for t in transitions)

    # ------------------------------------------------------------------ #
    # 7. dynamic peer groups & z-scores
    # ------------------------------------------------------------------ #
    def _create_dynamic_peer_groups(self, n_clusters: int | None = None) -> None:
        behav = (
            self.df.groupby("user_id")[
                ["file_rate", "logon_rate", "device_rate", "http_rate"]
            ].mean()
        ).fillna(0)
        if n_clusters is None:
            n_clusters = max(4, min(15, len(behav) // 25))
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        behav["peer_group"] = km.fit_predict(behav)
        self.peer_group_lookup = behav["peer_group"].to_dict()
        self.df["peer_group"] = self.df["user_id"].map(self.peer_group_lookup)

    def _add_peer_group_zscores(self) -> None:
        numeric = [
            "session_duration_s",
            "file_rate",
            "logon_rate",
            "device_rate",
            "http_rate",
        ]
        for col in numeric:
            grp_mean = self.df.groupby("peer_group")[col].transform("mean")
            grp_std = self.df.groupby("peer_group")[col].transform("std").replace(0, 1)
            self.df[f"{col}_z_peer"] = (self.df[col] - grp_mean) / grp_std

    # ------------------------------------------------------------------ #
    # 8. session-level aggregation
    # ------------------------------------------------------------------ #
    def _aggregate_session_features(self) -> None:
        # Build a dict mapping output column names to (input column, aggfunc)
        agg_funcs = {
            "session_start": ("timestamp", "min"),
            "session_end":   ("timestamp", "max"),
            "session_duration_s": ("session_duration_s", "mean"),
            "event_cnt":    ("event_cnt", "max"),
            "file_rate":    ("file_rate", "mean"),
            "logon_rate":   ("logon_rate", "mean"),
            "device_rate":  ("device_rate", "mean"),
            "http_rate":    ("http_rate", "mean"),
            "interarrival_mean_s": ("interarrival_mean_s", "mean"),
            "interarrival_std_s":  ("interarrival_std_s", "mean"),
            "rare_logon_hour": ("rare_logon_hour", "max"),
            "file_burst":   ("file_burst", "sum"),
            "2gram_diversity": ("2gram_diversity", "mean"),
            "3gram_diversity": ("3gram_diversity", "mean"),
            "unusual_transitions": ("unusual_transitions", "mean"),
        }
        # Include any peer-zscore columns
        for zcol in [c for c in self.df.columns if c.endswith("_z_peer")]:
            agg_funcs[zcol] = (zcol, "mean")

        # Use direct dict-based aggregation
        session_df = (
            self.df
                .groupby(["user_id", "session_id"])
                .agg(**agg_funcs)        # pandas ≥0.25 supports named aggregation
                .reset_index()
                .fillna(0)
        )
        self.session_features = session_df

