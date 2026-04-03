# Demonstration of the complete ETL pipeline
import os
from pathlib import Path
import tempfile
from sklearn import pipeline
from src.normalization import CERTDataIngestionPipeline
from datetime import datetime
import pandas as pd
from src.validation import UNIFIED_SCHEMA
from src.streaming.streaming_processor import StreamingCERTProcessor
import time
import json

# Minimal sample data for demonstration purposes
sample_data = {
    "logon": pd.DataFrame([
        {
            "id": 1,
            "date": "01/02/2010 08:28:06",
            "user": "U123",
            "pc": "PC01",
            "activity": "Logon"
        }
    ]),
    "file": pd.DataFrame([
        {
            "id": 1,
            "date": "01/02/2010 10:30:00",
            "user": "U123",
            "pc": "PC01",
            "filename": "secret.pdf",
            "content": "C:\\Users\\U123\\Documents\\"
        }
    ]),
    "email": pd.DataFrame([
        {
            "id": 1,
            "date": "01/02/2010 13:22:10",
            "user": "U123",
            "pc": "PC01",
            "to": "admin@company.com",
            "cc": "",
            "bcc": "",
            "from": "user@company.com",
            "size": 512,
            "attachments": 0,
            "content": "Here’s the data you asked for."
        }
    ]),
    "device": pd.DataFrame([
        {
            "id": 1,
            "date": "01/02/2010 12:45:00",
            "user": "U123",
            "pc": "PC01",
            "activity": "connect"
        }
    ]),
    "http": pd.DataFrame([
        {
            "id": 1,
            "date": "01/02/2010 14:55:00",
            "user": "U123",
            "pc": "PC01",
            "url": "http://suspicious-site.com",
            "content": "download"
        }
    ])
}

# Create a comprehensive demonstration
def demonstrate_pipeline():
    """Demonstrate both batch and streaming processing capabilities"""
    
    print("="*60)
    print("CERT INSIDER THREAT DETECTION - ETL PIPELINE DEMONSTRATION") 
    print("="*60)
    
    # Create temporary directories for demonstration
    data_dir = Path("C:/IPD_Project/data/cert_data")
    output_dir = Path("C:/IPD_Project/data/processed_data")
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nWorking directories:")
    print(f"  Data directory: {data_dir}")
    print(f"  Output directory: {output_dir}")
    
    # Initialize the pipeline
    pipeline = CERTDataIngestionPipeline(data_dir, output_dir, mode="batch")
    
    print("\n" + "="*60)
    print("BATCH PROCESSING DEMONSTRATION")
    print("="*60)
    
    # Demonstrate batch processing with sample data
    print("\n1. Processing actual CERT files using pipeline logic...")
    
    combined_batch, processing_summary = pipeline.batch_process_all_files()

    batch_results = {}  # Initialize the dictionary to store normalized results

    print(f"   - Total combined records: {len(combined_batch)}")
    for file_type, summary in processing_summary.items():
        print(f"   - {file_type}: {summary.get('records_processed', 0)} records, "
              f"valid: {summary.get('valid_records', 0)}, "
              f"quality: {summary.get('quality_score', 0):.2f}")

        
        
        # Validate quality
        pipeline = CERTDataIngestionPipeline(data_dir, output_dir)
        normalized = pipeline.process_single_file(file_type)

        if normalized.empty:
            print(f"     - No data processed for {file_type}")
            continue
        print(f"     - Records processed: {len(normalized)}")
        print(f"     - Valid records: {normalized['is_valid'].sum()}")
        print(f"     - Avg quality score: {normalized['data_quality_score'].mean():.1%}")
        
        # Add quality metrics and align to schema
        def add_quality_metrics_simple(df, quality_report):
            df = df.copy()
            df["data_quality_score"] = quality_report["overall_quality_score"] / 100.0
            df["is_valid"] = df[["event_id", "user_id", "pc_id"]].notna().all(axis=1)
            df["validation_errors"] = ""
            return df
        
        def align_to_unified_schema_simple(df):
            aligned_df = df.copy()
            for column in UNIFIED_SCHEMA.keys():
                if column not in aligned_df.columns:
                    if UNIFIED_SCHEMA[column] == "string":
                        aligned_df[column] = ""
                    elif UNIFIED_SCHEMA[column] == "int64":
                        aligned_df[column] = 0
                    elif UNIFIED_SCHEMA[column] == "float64":
                        aligned_df[column] = 0.0
                    elif UNIFIED_SCHEMA[column] == "bool":
                        aligned_df[column] = True
                    elif "datetime" in UNIFIED_SCHEMA[column]:
                        aligned_df[column] = pd.NaT
                    else:
                        aligned_df[column] = None
            
            return aligned_df[list(UNIFIED_SCHEMA.keys())]
        
        # Retrieve or define quality_report before using it
        quality_report = summary.get('quality_report', {"overall_quality_score": 100})

        normalized = add_quality_metrics_simple(normalized, quality_report)
        normalized = align_to_unified_schema_simple(normalized)
        
        print(f"     - Normalized records: {len(normalized)}")
        print(f"     - Valid records: {normalized['is_valid'].sum()}")
        
        batch_results[file_type] = normalized
    
    # Combine all batch data
    print("\n2. Combining all data types...")
    combined_batch = pd.concat(batch_results.values(), ignore_index=True)
    combined_batch = combined_batch.sort_values(by="timestamp", na_position='last')
    
    print(f"   - Total combined records: {len(combined_batch)}")
    print(f"   - Activity type distribution:")
    for activity, count in combined_batch["activity_type"].value_counts().items():
        print(f"     • {activity}: {count} records")
    
    # Save batch results
    output_file = "cert_batch_processed"
    csv_path = os.path.join(output_dir, f"{output_file}.csv")
    combined_batch.to_csv(csv_path, index=False)
    print(f"\n   - Saved batch results to: {csv_path}")
    
    print("\n" + "="*60)
    print("STREAMING PROCESSING DEMONSTRATION") 
    print("="*60)
    
    # Demonstrate streaming processing
    print("\n1. Initializing streaming processor...")
    streaming_pipeline = CERTDataIngestionPipeline(data_dir, output_dir, mode="stream")
    stream_processor = StreamingCERTProcessor(streaming_pipeline, buffer_size=3, flush_interval=2)
    
    print("   - Buffer size: 3 records per type")
    print("   - Flush interval: 2 seconds")
    
    # Simulate streaming data processing
    print("\n2. Processing streaming events (5-second simulation)...")
    
    stream_processor.is_running = True
    
    # Simulate some streaming events
    streaming_events = []

# Only take 1 record per type for demo, safely
    for file_type in ["logon", "file", "email"]:
        df = sample_data[file_type]
        if not df.empty:
            for idx, row in df.head(1).iterrows():  # Safe for short DataFrames
                streaming_events.append({
                "file_type": file_type,
                "data": row.to_dict(),
                "timestamp": datetime.now().isoformat()
            })
    
    # Process streaming events
    for i, event in enumerate(streaming_events):
        stream_processor.add_event_to_buffer(event["file_type"], event)
        print(f"   - Added {event['file_type']} event to buffer")
        time.sleep(0.5)  # Simulate real-time processing
        
        # Check if any buffers were flushed
        metrics = stream_processor.get_metrics()
        if metrics["events_processed"] > 0:
            print(f"     Buffer flushed: {metrics['events_processed']} events processed")
    
    # Final flush
    print("\n3. Final buffer flush...")
    total_flushed = 0
    for file_type in ["logon", "file", "email"]:
        flushed = stream_processor.flush_buffer(file_type)
        if flushed > 0:
            print(f"   - Flushed {flushed} {file_type} events")
            total_flushed += flushed
    
    # Process any remaining items in the processed queue
    processed_items = []
    while not stream_processor.processed_queue.empty():
        try:
            item = stream_processor.processed_queue.get_nowait()
            processed_items.append(item)
        except:
            break
    
    if processed_items:
        print(f"\n4. Streaming processing results:")
        for item in processed_items:
            print(f"   - {item['file_type']}: {item['record_count']} records processed")
            print(f"     Quality score: {item['quality_report']['overall_quality_score']:.1f}%")
    
    # Final metrics
    final_metrics = stream_processor.get_metrics()
    print(f"\n5. Streaming metrics:")
    print(f"   - Total events processed: {final_metrics['events_processed']}")
    print(f"   - Failed events: {final_metrics['events_failed']}")
    print(f"   - Buffer levels: {final_metrics['buffer_levels']}")
    
    stream_processor.stop_streaming()
    
    print("\n" + "="*60)
    print("DATA QUALITY ANALYSIS")
    print("="*60)
    
    # Analyze data quality across the pipeline
    print("\n1. Data Quality Summary:")
    quality_summary = {
        "total_records_processed": len(combined_batch),
        "valid_records": combined_batch["is_valid"].sum(),
        "average_quality_score": combined_batch["data_quality_score"].mean(),
        "timestamp_coverage": (combined_batch["timestamp"].notna().sum() / len(combined_batch)) * 100,
        "user_coverage": len(combined_batch["user_id"].unique()),
        "pc_coverage": len(combined_batch["pc_id"].unique()),
        "date_range": {
            "start": combined_batch["timestamp"].min(),
            "end": combined_batch["timestamp"].max()
        }
    }
    
    for key, value in quality_summary.items():
        if key != "date_range":
            print(f"   - {key.replace('_', ' ').title()}: {value}")
        else:
            print(f"   - Date Range: {value['start']} to {value['end']}")
    
    print("\n2. Activity Pattern Analysis:")
    # Session analysis
    session_stats = combined_batch.groupby("session_id").agg({
        "event_id": "count",
        "activity_type": lambda x: list(x.unique()),
        "timestamp": ["min", "max"]
    }).reset_index()
    
    session_stats.columns = ["session_id", "event_count", "activity_types", "session_start", "session_end"]
    print(f"   - Total unique sessions: {len(session_stats)}")
    print(f"   - Average events per session: {session_stats['event_count'].mean():.1f}")
    
    # Activity patterns
    print(f"\n3. Activity Type Patterns:")
    activity_patterns = combined_batch.groupby(["user_id", "activity_type"]).size().unstack(fill_value=0)
    print(f"   - Users with logon activity: {(activity_patterns['logon'] > 0).sum()}")
    print(f"   - Users with file activity: {(activity_patterns.get('file', pd.Series([0])) > 0).sum()}")
    print(f"   - Users with email activity: {(activity_patterns.get('email', pd.Series([0])) > 0).sum()}")
    
    print("\n" + "="*60)
    print("DEPLOYMENT CONFIGURATION")
    print("="*60)
    
    print("\n1. Production Configuration Template:")
    config_template = {
        "data_sources": {
            "kafka": {
                "bootstrap_servers": "localhost:9092",
                "topics": ["cert-logon", "cert-file", "cert-email", "cert-device", "cert-http"],
                "consumer_group": "cert-insider-threat-detection"
            },
            "file_watch": {
                "directories": ["/var/log/cert/", "/data/cert_feeds/"],
                "file_patterns": ["*.csv", "*.json", "*.log"]
            }
        },
        "processing": {
            "batch_size": 1000,
            "flush_interval_seconds": 30,
            "max_buffer_size": 10000,
            "worker_threads": 4,
            "quality_thresholds": {
                "missing_data": 0.1,
                "duplicates": 0.05,
                "timestamp_validity": 0.95
            }
        },
        "output": {
            "formats": ["parquet", "csv", "json"],
            "destinations": {
                "local": "/data/processed/cert/",
                "s3": "s3://insider-threat-data/cert/",
                "database": {
                    "connection_string": "postgresql://user:pass@localhost/insider_threat",
                    "table": "cert_events"
                }
            }
        },
        "monitoring": {
            "metrics_interval": 60,
            "log_level": "INFO",
            "alerts": {
                "quality_score_threshold": 0.8,
                "processing_lag_threshold": 300
            }
        }
    }
    
    config_file = os.path.join(output_dir, "pipeline_config.json")
    with open(config_file, 'w') as f:
        json.dump(config_template, f, indent=2, default=str)
    
    print(f"   - Configuration saved to: {config_file}")
    print("   - Kafka integration ready")
    print("   - File watching capabilities")
    print("   - Multi-destination output")
    print("   - Monitoring and alerting")
    
    print(f"\n2. Directory structure created:")
    print(f"   {data_dir}/")
    print(f"   ├── cert_data/          # Input data directory")
    print(f"   └── processed_data/     # Output directory")
    print(f"       ├── cert_batch_processed.csv")
    print(f"       └── pipeline_config.json")
    
    print("\n" + "="*60)
    print("PIPELINE DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return {
        "batch_results": combined_batch,
        "streaming_processor": stream_processor,
        "config_file": config_file,
        "output_directory": output_dir
    }

# Run the demonstration
demo_results = demonstrate_pipeline()
print(f"\nProcessed {len(demo_results['batch_results'])} records successfully!")
print(f"Pipeline configuration and results saved in: {demo_results['output_directory']}")   