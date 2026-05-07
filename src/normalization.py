# Continue with the pipeline implementation - add the remaining methods and demonstrate processing
import hashlib
from pathlib import Path
import pandas as pd
import json
# Use local normalization functions defined below instead of importing
# from src.normalization_modules.logon import normalize_logon_data
# from src.normalization_modules.file import normalize_file_data
# from src.normalization_modules.email import normalize_email_data
# from src.normalization_modules.device import normalize_device_data
# from src.normalization_modules.http import normalize_http_data
from typing import Dict, Any, Optional
from datetime import datetime
from src.validation import CERT_FILES, UNIFIED_SCHEMA
from src.utils import get_logger
logger = get_logger(__name__)

class CERTDataIngestionPipeline:
    """
    Modular ETL pipeline for CERT insider threat dataset
    Supports both batch and streaming ingestion modes
    """
    
    def __init__(self, data_dir: str, output_dir: str, mode: str = "batch"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.mode = mode
        self.unified_schema = UNIFIED_SCHEMA
        self.batch_size = 10000  # For streaming mode
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data quality thresholds
        self.quality_thresholds = {
            "missing_data_threshold": 0.1,  # 10% missing data threshold
            "duplicate_threshold": 0.05,    # 5% duplicate threshold
            "timestamp_validity_threshold": 0.95  # 95% valid timestamps
        }
        
        logger.info(f"Initialized CERT Data Ingestion Pipeline in {mode} mode")
    
    # ... (previous methods remain the same)
    
    def load_cert_file(self, file_type: str) -> pd.DataFrame:
        """Load and perform initial validation on CERT data files"""
        file_config = CERT_FILES[file_type]
        file_path = self.data_dir / file_config["filename"]
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return pd.DataFrame()
        
        try:
            # Load with error handling for malformed files
            df = pd.read_csv(file_path, 
                           names=file_config["columns"], 
                           skiprows=1,  # Skip header
                           encoding='utf-8',
                           on_bad_lines='skip')
            
            logger.info(f"Loaded {len(df)} records from {file_config['filename']}")
            return df
            
        except Exception as e:
            logger.error(f"Error  {file_path}: {str(e)}")
            return pd.DataFrame()

    def add_quality_metrics(self, df: pd.DataFrame, quality_report: Dict[str, Any]) -> pd.DataFrame:
        """Add data quality metrics to the normalized dataframe"""
        df = df.copy()
        
        # Add quality score based on completeness and validity
        df["data_quality_score"] = quality_report["overall_quality_score"] / 100.0
        
        # Mark records as valid based on key field presence
        key_fields = ["event_id", "timestamp", "user_id", "pc_id", "activity_type"]
        df["is_valid"] = df[key_fields].notna().all(axis=1)
        
        # Collect validation errors
        validation_errors = []
        for idx, row in df.iterrows():
            errors = []
            if pd.isna(row["timestamp"]):
                errors.append("invalid_timestamp")
            if pd.isna(row["user_id"]) or row["user_id"] == "":
                errors.append("missing_user_id")
            if pd.isna(row["pc_id"]) or row["pc_id"] == "":
                errors.append("missing_pc_id")
            
            validation_errors.append("|".join(errors) if errors else "")
        
        df["validation_errors"] = validation_errors
        
        return df
    
    def align_to_unified_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure dataframe aligns with unified schema by adding missing columns"""
        aligned_df = df.copy()
        
        # Add missing columns with appropriate default values
        for column, dtype in self.unified_schema.items():
            if column not in aligned_df.columns:
                if dtype == "string":
                    aligned_df[column] = ""
                elif dtype == "int64":
                    aligned_df[column] = 0
                elif dtype == "float64":
                    aligned_df[column] = 0.0
                elif dtype == "bool":
                    aligned_df[column] = True
                elif "datetime" in dtype:
                    aligned_df[column] = pd.NaT
                else:
                    aligned_df[column] = None
        
        # Reorder columns to match schema
        aligned_df = aligned_df[list(self.unified_schema.keys())]
        
        return aligned_df
    
    def validate_data_quality(self, df: pd.DataFrame, file_type: str) -> Dict[str, Any]:
        """
        Comprehensive data quality validation
        Returns quality metrics and validation results
        """
        total_records = len(df)
        quality_report = {
            "file_type": file_type,
            "total_records": total_records,
            "quality_checks": {}
        }
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        missing_percentage = (missing_counts / total_records) * 100
        quality_report["quality_checks"]["missing_data"] = {
            "missing_counts": missing_counts.to_dict(),
            "missing_percentages": missing_percentage.to_dict(),
            "passes_threshold": all(missing_percentage < (self.quality_thresholds["missing_data_threshold"] * 100))
        }
        
        # Check for duplicates
        duplicate_count = df.duplicated().sum()
        duplicate_percentage = (duplicate_count / total_records) * 100
        quality_report["quality_checks"]["duplicates"] = {
            "duplicate_count": duplicate_count,
            "duplicate_percentage": duplicate_percentage,
            "passes_threshold": duplicate_percentage < (self.quality_thresholds["duplicate_threshold"] * 100)
        }
        
        # Validate timestamps if present
        if "timestamp" in df.columns:
            valid_timestamps = df["timestamp"].notna().sum()
            timestamp_validity = (valid_timestamps / total_records) * 100
            quality_report["quality_checks"]["timestamp_validity"] = {
                "valid_timestamps": valid_timestamps,
                "validity_percentage": timestamp_validity,
                "passes_threshold": timestamp_validity >= (self.quality_thresholds["timestamp_validity_threshold"] * 100)
            }
        
        # Calculate overall quality score
        checks_passed = sum(1 for check in quality_report["quality_checks"].values() 
                          if check.get("passes_threshold", False))
        total_checks = len(quality_report["quality_checks"])
        quality_score = (checks_passed / total_checks) * 100 if total_checks > 0 else 0
        quality_report["overall_quality_score"] = quality_score
        
        return quality_report

    def process_single_file(self, file_type: str) -> pd.DataFrame:
        """Process a single CERT file type through the complete ETL pipeline"""
        logger.info(f"Processing {file_type} data...")
        
        # Extract: Load raw data
        raw_data = self.load_cert_file(file_type)
        
        if raw_data.empty:
            logger.warning(f"No data found for {file_type}")
            return pd.DataFrame()
        
        # Validate raw data quality
        quality_report = self.validate_data_quality(raw_data, file_type)
        logger.info(f"Data quality score for {file_type}: {quality_report['overall_quality_score']:.2f}%")
        
        # Transform: Normalize data based on file type
        if file_type == "logon":
            normalized_data = normalize_logon_data(raw_data)
        elif file_type == "file":
            normalized_data = normalize_file_data(raw_data)
        elif file_type == "email":
            normalized_data = normalize_email_data(raw_data)
        elif file_type == "device":
            normalized_data = normalize_device_data(raw_data)
        elif file_type == "http":
            normalized_data = normalize_http_data(raw_data)
        else:
            logger.error(f"Unsupported file type: {file_type}")
            return pd.DataFrame()
        
        # Add quality metrics
        normalized_data = self.add_quality_metrics(normalized_data, quality_report)
        
        # Align to unified schema
        normalized_data = self.align_to_unified_schema(normalized_data)
        
        return normalized_data
    
    def handle_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle duplicate records with configurable strategy"""
        initial_count = len(df)
        
        # Remove exact duplicates
        df_deduplicated = df.drop_duplicates()
        
        # Handle near-duplicates based on key fields
        key_fields = ["user_id", "pc_id", "timestamp", "activity_type"]
        df_final = df_deduplicated.drop_duplicates(subset=key_fields, keep='first')
        
        removed_count = initial_count - len(df_final)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} duplicate records")
        
        return df_final
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with appropriate imputation strategies"""
        df_cleaned = df.copy()
        
        # Strategy 1: Forward fill for session-based fields
        session_fields = ["user_id", "pc_id"]
        for field in session_fields:
            if field in df_cleaned.columns:
                df_cleaned[field] = df_cleaned[field].fillna(method='ffill')
        
        # Strategy 2: Fill categorical fields with "Unknown"
        categorical_fields = ["activity_type", "activity_subtype", "source_file"]
        for field in categorical_fields:
            if field in df_cleaned.columns:
                df_cleaned[field] = df_cleaned[field].fillna("Unknown")
        
        # Strategy 3: Fill numeric fields with 0 
        numeric_fields = ["email_size", "email_attachments"]
        for field in numeric_fields:
            if field in df_cleaned.columns:
                df_cleaned[field] = df_cleaned[field].fillna(0)
        
        # Strategy 4: Fill text fields with empty string
        text_fields = ["email_content", "file_name", "url", "validation_errors"]
        for field in text_fields:
            if field in df_cleaned.columns:
                df_cleaned[field] = df_cleaned[field].fillna("")
        
        return df_cleaned
    
    def batch_process_all_files(self) -> pd.DataFrame:
        """Process all CERT files in batch mode"""
        logger.info("Starting batch processing of all CERT files...")
        
        all_data = []
        processing_summary = {}
        
        for file_type in CERT_FILES.keys():
            try:
                processed_data = self.process_single_file(file_type)
                if not processed_data.empty:
                    # Apply data cleaning
                    processed_data = self.handle_duplicates(processed_data)
                    processed_data = self.handle_missing_values(processed_data)
                    
                    all_data.append(processed_data)
                    processing_summary[file_type] = {
                        "records_processed": len(processed_data),
                        "valid_records": processed_data["is_valid"].sum(),
                        "quality_score": processed_data["data_quality_score"].mean()
                    }
                    
                    logger.info(f"Successfully processed {len(processed_data)} records from {file_type}")
                
            except Exception as e:
                logger.error(f"Error processing {file_type}: {str(e)}")
                processing_summary[file_type] = {"error": str(e)}
        
        # Combine all data
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            combined_data = combined_data.sort_values(by="timestamp", na_position='last')
            
            logger.info(f"Combined dataset created with {len(combined_data)} total records")
            return combined_data, processing_summary
        else:
            logger.warning("No data was successfully processed")
            return pd.DataFrame(), processing_summary
    
    def save_processed_data(self, df: pd.DataFrame, filename: str):
        """Save processed data to various formats"""
        if df.empty:
            logger.warning("No data to save")
            return
        
        base_path = self.output_dir / filename
        
        # Save as CSV
        csv_path = base_path.with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV to {csv_path}")
        
        # Save as Parquet (more efficient for large datasets)
        parquet_path = base_path.with_suffix('.parquet')
        df.to_parquet(parquet_path, index=False)
        logger.info(f"Saved Parquet to {parquet_path}")
        
        # Save schema information
        schema_info = {
            "total_records": len(df),
            "columns": list(df.columns),
            "data_types": df.dtypes.astype(str).to_dict(),
            "date_range": {
                "start": df["timestamp"].min().isoformat() if df["timestamp"].notna().any() else None,
                "end": df["timestamp"].max().isoformat() if df["timestamp"].notna().any() else None
            },
            "quality_metrics": {
                "avg_quality_score": df["data_quality_score"].mean(),
                "valid_records_percentage": (df["is_valid"].sum() / len(df)) * 100,
                "records_by_activity_type": df["activity_type"].value_counts().to_dict()
            }
        }
        
        schema_path = base_path.with_suffix('.json')
        with open(schema_path, 'w') as f:
            json.dump(schema_info, f, indent=2, default=str)
        logger.info(f"Saved schema info to {schema_path}")

# Re-create the methods that were defined earlier (copying from first code block)
def generate_session_id(user: str, pc: str, date_str: str) -> str:
    """Generate session ID based on user, PC, and date"""
    session_key = f"{user}_{pc}_{date_str[:10]}"  # Use date part only
    return hashlib.md5(session_key.encode()).hexdigest()[:12]

def standardize_timestamp(timestamp_str: str) -> Optional[datetime]:
    """
    Standardize timestamp formats to ISO 8601 UTC format
    Handles various timestamp formats commonly found in logs
    """
    if pd.isna(timestamp_str) or timestamp_str == "":
        return None
        
    # Common timestamp formats in CERT dataset
    formats = [
        "%m/%d/%Y %H:%M:%S",     # 01/02/2010 08:28:06
        "%Y-%m-%d %H:%M:%S",     # 2010-01-02 08:28:06  
        "%m/%d/%Y %H:%M",        # 01/02/2010 08:28
        "%d-%m-%Y %H:%M",
        "%Y-%m-%dT%H:%M:%S",     # 2010-01-02T08:28:06
        "%Y-%m-%dT%H:%M:%SZ",    # 2010-01-02T08:28:06Z
        "%Y-%m-%d",              # 2010-01-02
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(str(timestamp_str), fmt)
            return dt
        except ValueError:
            continue
    
    logger.warning(f"Could not parse timestamp: {timestamp_str}")
    return None

def validate_data_quality(df: pd.DataFrame, file_type: str) -> Dict[str, Any]:
    """
    Comprehensive data quality validation
    Returns quality metrics and validation results
    """
    total_records = len(df)
    quality_report = {
        "file_type": file_type,
        "total_records": total_records,
        "quality_checks": {}
    }
    
    # Check for missing values
    missing_counts = df.isnull().sum()
    missing_percentage = (missing_counts / total_records) * 100
    quality_report["quality_checks"]["missing_data"] = {
        "missing_counts": missing_counts.to_dict(),
        "missing_percentages": missing_percentage.to_dict(),
        "passes_threshold": all(missing_percentage < 10.0)  # 10% threshold
    }
    
    # Check for duplicates
    duplicate_count = df.duplicated().sum()
    duplicate_percentage = (duplicate_count / total_records) * 100
    quality_report["quality_checks"]["duplicates"] = {
        "duplicate_count": duplicate_count,
        "duplicate_percentage": duplicate_percentage,
        "passes_threshold": duplicate_percentage < 5.0  # 5% threshold
    }
    
    # Validate timestamps if date column exists
    if "date" in df.columns:
        valid_timestamps = sum(1 for ts in df["date"] if standardize_timestamp(ts) is not None)
        timestamp_validity = (valid_timestamps / total_records) * 100
        quality_report["quality_checks"]["timestamp_validity"] = {
            "valid_timestamps": valid_timestamps,
            "validity_percentage": timestamp_validity,
            "passes_threshold": timestamp_validity >= 95.0  # 95% threshold
        }
    
    # Calculate overall quality score
    checks_passed = sum(1 for check in quality_report["quality_checks"].values() 
                      if check.get("passes_threshold", False))
    total_checks = len(quality_report["quality_checks"])
    quality_score = (checks_passed / total_checks) * 100 if total_checks > 0 else 0
    quality_report["overall_quality_score"] = quality_score
    
    return quality_report

def normalize_logon_data(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize logon data to unified schema"""
        if df.empty:
            return pd.DataFrame()
            
        normalized = pd.DataFrame()
        print("Columns in logon df:", df.columns.tolist())
        
        # Map to unified schema
        normalized["event_id"] = df["id"].astype(str)
        normalized["timestamp"] = df["date"].apply(standardize_timestamp)
        normalized["user_id"] = df["user"].astype(str)
        normalized["pc_id"] = df["pc"].astype(str)
        normalized["activity_type"] = "logon"
        normalized["activity_subtype"] = df["activity"]
        normalized["logon_activity"] = df["activity"]
        normalized["source_file"] = "logon.csv"
        
        # Generate session IDs
        normalized["session_id"] = df.apply(
            lambda row: generate_session_id(str(row["user"]), str(row["pc"]), str(row["date"])), 
            axis=1
        )
        
        return normalized

def normalize_file_data( df: pd.DataFrame) -> pd.DataFrame:
        """Normalize file access data to unified schema"""
        if df.empty:
            return pd.DataFrame()
            
        normalized = pd.DataFrame()
        
        # Map to unified schema
        normalized["event_id"] = df["id"].astype(str) 
        normalized["timestamp"] = df["date"].apply(standardize_timestamp)
        normalized["user_id"] = df["user"].astype(str)
        normalized["pc_id"] = df["pc"].astype(str)
        normalized["activity_type"] = "file"
        normalized["activity_subtype"] = "access"
        normalized["file_name"] = df["filename"]
        normalized["file_tree"] = df["content"] 
        normalized["source_file"] = "file.csv"
        
        # Generate session IDs
        normalized["session_id"] = df.apply(
            lambda row: generate_session_id(str(row["user"]), str(row["pc"]), str(row["date"])), 
            axis=1
        )
        
        return normalized

def normalize_email_data(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize email data to unified schema"""
        if df.empty:
            return pd.DataFrame()
            
        normalized = pd.DataFrame()
        
        # Map to unified schema
        normalized["event_id"] = df["id"].astype(str)
        normalized["timestamp"] = df["date"].apply(standardize_timestamp)
        normalized["user_id"] = df["user"].astype(str)
        normalized["pc_id"] = df["pc"].astype(str)
        normalized["activity_type"] = "email"
        normalized["activity_subtype"] = "send"
        normalized["email_to"] = df["to"]
        normalized["email_cc"] = df["cc"]  
        normalized["email_bcc"] = df["bcc"]
        normalized["email_from"] = df["from"]
        normalized["email_size"] = pd.to_numeric(df["size"], errors="coerce")
        normalized["email_attachments"] = pd.to_numeric(df["attachments"], errors="coerce")
        normalized["email_content"] = df["content"]
        normalized["source_file"] = "email.csv"
        
        # Generate session IDs
        normalized["session_id"] = df.apply(
            lambda row: generate_session_id(str(row["user"]), str(row["pc"]), str(row["date"])), 
            axis=1
        )
        
        return normalized

def normalize_device_data(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize device data to unified schema"""
        if df.empty:
            return pd.DataFrame()
            
        normalized = pd.DataFrame()
        
        # Map to unified schema  
        normalized["event_id"] = df["id"].astype(str)
        normalized["timestamp"] = df["date"].apply(standardize_timestamp)
        normalized["user_id"] = df["user"].astype(str)
        normalized["pc_id"] = df["pc"].astype(str)
        normalized["activity_type"] = "device"
        normalized["activity_subtype"] = df["activity"]
        normalized["device_activity"] = df["activity"]
        normalized["source_file"] = "device.csv"
        
        # Generate session IDs
        normalized["session_id"] = df.apply(
            lambda row: generate_session_id(str(row["user"]), str(row["pc"]), str(row["date"])), 
            axis=1
        )
        
        return normalized
    
def normalize_http_data(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize HTTP data to unified schema"""
        if df.empty:
            return pd.DataFrame()
            
        normalized = pd.DataFrame()
        
        # Map to unified schema
        normalized["event_id"] = df["id"].astype(str)
        normalized["timestamp"] = df["date"].apply(standardize_timestamp)
        normalized["user_id"] = df["user"].astype(str)  
        normalized["pc_id"] = df["pc"].astype(str)
        normalized["activity_type"] = "http"
        normalized["activity_subtype"] = "request"
        normalized["url"] = df["url"]
        normalized["http_content"] = df["content"]
        normalized["source_file"] = "http.csv"
        
        # Generate session IDs
        normalized["session_id"] = df.apply(
            lambda row: generate_session_id(str(row["user"]), str(row["pc"]), str(row["date"])), 
            axis=1
        )
        
        return normalized    

print("Extended pipeline methods created successfully!")
print("\nNew capabilities added:")
print("- Data quality validation and scoring")
print("- Missing value handling strategies") 
print("- Duplicate detection and removal")
print("- Schema alignment and validation")
print("- Multi-format output (CSV, Parquet, JSON metadata)")
print("- Comprehensive error handling and logging")