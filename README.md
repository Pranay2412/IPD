# Insider Threat Detection System

This repository contains a prototype insider threat detection system designed to analyze enterprise user behavior and certificate telemetry for anomalous activity. The platform demonstrates end-to-end analytics for data processing, feature engineering, unsupervised anomaly scoring, false-positive reduction, and risk reporting.

## Project overview

The system is built around a UEBA-style workflow focused on certificate-based authentication and user activity. It supports:

- ingesting and preprocessing multiple telemetry sources, including certificate events and user metadata
- normalizing session and identity features for analysis
- engineering behavioral features that capture unusual certificate usage, logon patterns, and cross-channel activity
- applying density-based local outlier factor (DBLOF) models to detect anomalous users and certificate sessions
- evaluating detection quality, precision/recall, and false positive reduction strategies
- generating user risk summaries and anomaly reports for analyst review

## Architecture and modules

- `src/config.py`: configuration settings and pipeline parameters
- `src/feature_engineering.py`: raw feature construction and session aggregation
- `src/normalization.py`: scaling and normalization routines applied to engineerd features
- `src/utils.py`: shared utilities for data handling and workflow orchestration
- `src/validation.py`: validation helper logic and sanity checks
- `src/analytics/`: analytics modules for scoring, false-positive reduction, and threat reasoning
  - `dblof_fp_reducer.py`: reduces DBLOF false positives using score and context heuristics
  - `dblof_ueba_module.py`: DBLOF-based UEBA scoring engine
  - `run_analytics.py`: orchestrates analytics workflows and output generation
  - other modules support reasoning, threat graph synthesis, explainability, fairness validation, and analyst feedback
- `src/models/`: model implementations for LOF detection and supervised baselines
- `src/streaming/`: Kafka connector and streaming processor for ingesting live telemetry

## Data organization

- `data/`: input datasets and raw telemetry sources
- `data/cert_data/`: certificate-specific event files such as device, email, file, http, logon, and psychometric datasets
- `processed_data/`: preprocessed and intermediate feature datasets used by analytics pipelines
- `cert_analysis_output/`, `cert_dblof_analysis_output/`, `cert_dblof_fp_reduced_output/`: generated analysis results, scores, anomaly lists, and evaluation summaries

## Usage

Run the included scripts to execute the analytic pipelines and evaluate model performance:

- `run_dblof_analytics.py`: execute certificate-focused DBLOF analytics and generate anomaly output
- `run_fp_reduction.py`: apply false positive reduction techniques and summarize reduced anomalies
- `evaluate_dblof_performance.py`: compute performance metrics and compare detection results

## Testing and validation

- `tests/`: contains synthetic log generation and dataset validation utilities
- `tests/generate_synthetic_api_logs.py`: generate mock telemetry for testing workflows

## Goal

The repository is intended to support experimentation with insider threat detection techniques, particularly around certificate-based anomalies and user risk profiling. It is useful for prototyping UEBA workflows, evaluating unsupervised anomaly scoring, and exploring false positive reduction strategies.
