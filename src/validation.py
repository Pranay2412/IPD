# Create the complete ETL pipeline structure for the CERT dataset
import os
import pandas as pd
import numpy as np
from datetime import datetime
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
import hashlib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the unified schema for the CERT dataset
UNIFIED_SCHEMA = {
    "event_id": "string",
    "timestamp": "datetime64[ns]",  
    "user_id": "string",
    "pc_id": "string",
    "activity_type": "string",  # logon, file, email, device, http
    "activity_subtype": "string",  # connect/disconnect for device, logon/logoff for logon
    "source_file": "string",
    "session_id": "string",
    # Logon specific
    "logon_activity": "string",
    # File specific  
    "file_tree": "string",
    "file_name": "string",
    # Email specific
    "email_to": "string",
    "email_cc": "string", 
    "email_bcc": "string",
    "email_from": "string",
    "email_size": "int64",
    "email_attachments": "int64",
    "email_content": "string",
    # Device specific
    "device_activity": "string",
    # HTTP specific
    "url": "string",
    "http_content": "string",
    # Quality metrics
    "data_quality_score": "float64",
    "is_valid": "bool",
    "validation_errors": "string"
}

# CERT dataset file configurations
CERT_FILES = {
    "logon": {
        "filename": "logon.csv",
        "columns": ["id", "date", "user", "pc", "activity"],
        "activity_type": "logon"
    },
    "file": {
        "filename": "file.csv", 
        "columns": ["id", "date", "user", "pc", "filename", "content"],
        "activity_type": "file"
    },
    "email": {
        "filename": "email.csv",
        "columns": ["id", "date", "user", "pc", "to", "cc", "bcc", "from", "size", "attachments", "content"],
        "activity_type": "email"
    },
    "device": {
        "filename": "device.csv",
        "columns": ["id", "date", "user", "pc", "activity"],
        "activity_type": "device"
    },
    "http": {
        "filename": "http.csv",
        "columns": ["id", "date", "user", "pc", "url", "content"],
        "activity_type": "http"
    }
}

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
    
    def generate_session_id(self, user: str, pc: str, date_str: str) -> str:
        """Generate session ID based on user, PC, and date"""
        session_key = f"{user}_{pc}_{date_str[:10]}"  # Use date part only
        return hashlib.md5(session_key.encode()).hexdigest()[:12]
    
    def standardize_timestamp(self, timestamp_str: str) -> Optional[datetime]:
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
    
    
    def normalize_logon_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize logon data to unified schema"""
        if df.empty:
            return pd.DataFrame()
            
        normalized = pd.DataFrame()
        
        # Map to unified schema
        normalized["event_id"] = df["id"].astype(str)
        normalized["timestamp"] = df["date"].apply(self.standardize_timestamp)
        normalized["user_id"] = df["user"].astype(str)
        normalized["pc_id"] = df["pc"].astype(str)
        normalized["activity_type"] = "logon"
        normalized["activity_subtype"] = df["activity"]
        normalized["logon_activity"] = df["activity"]
        normalized["source_file"] = "logon.csv"
        
        # Generate session IDs
        normalized["session_id"] = df.apply(
            lambda row: self.generate_session_id(str(row["user"]), str(row["pc"]), str(row["date"])), 
            axis=1
        )
        
        return normalized
    
    def normalize_file_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize file access data to unified schema"""
        if df.empty:
            return pd.DataFrame()
            
        normalized = pd.DataFrame()
        
        # Map to unified schema
        normalized["event_id"] = df["id"].astype(str) 
        normalized["timestamp"] = df["date"].apply(self.standardize_timestamp)
        normalized["user_id"] = df["user"].astype(str)
        normalized["pc_id"] = df["pc"].astype(str)
        normalized["activity_type"] = "file"
        normalized["activity_subtype"] = "access"
        normalized["file_name"] = df["filename"]
        normalized["file_tree"] = df["content"] 
        normalized["source_file"] = "file.csv"
        
        # Generate session IDs
        normalized["session_id"] = df.apply(
            lambda row: self.generate_session_id(str(row["user"]), str(row["pc"]), str(row["date"])), 
            axis=1
        )
        
        return normalized
    
    def normalize_email_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize email data to unified schema"""
        if df.empty:
            return pd.DataFrame()
            
        normalized = pd.DataFrame()
        
        # Map to unified schema
        normalized["event_id"] = df["id"].astype(str)
        normalized["timestamp"] = df["date"].apply(self.standardize_timestamp)
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
            lambda row: self.generate_session_id(str(row["user"]), str(row["pc"]), str(row["date"])), 
            axis=1
        )
        
        return normalized
    
    def normalize_device_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize device data to unified schema"""
        if df.empty:
            return pd.DataFrame()
            
        normalized = pd.DataFrame()
        
        # Map to unified schema  
        normalized["event_id"] = df["id"].astype(str)
        normalized["timestamp"] = df["date"].apply(self.standardize_timestamp)
        normalized["user_id"] = df["user"].astype(str)
        normalized["pc_id"] = df["pc"].astype(str)
        normalized["activity_type"] = "device"
        normalized["activity_subtype"] = df["activity"]
        normalized["device_activity"] = df["activity"]
        normalized["source_file"] = "device.csv"
        
        # Generate session IDs
        normalized["session_id"] = df.apply(
            lambda row: self.generate_session_id(str(row["user"]), str(row["pc"]), str(row["date"])), 
            axis=1
        )
        
        return normalized
    
    def normalize_http_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize HTTP data to unified schema"""
        if df.empty:
            return pd.DataFrame()
            
        normalized = pd.DataFrame()
        
        # Map to unified schema
        normalized["event_id"] = df["id"].astype(str)
        normalized["timestamp"] = df["date"].apply(self.standardize_timestamp)
        normalized["user_id"] = df["user"].astype(str)  
        normalized["pc_id"] = df["pc"].astype(str)
        normalized["activity_type"] = "http"
        normalized["activity_subtype"] = "request"
        normalized["url"] = df["url"]
        normalized["http_content"] = df["content"]
        normalized["source_file"] = "http.csv"
        
        # Generate session IDs
        normalized["session_id"] = df.apply(
            lambda row: self.generate_session_id(str(row["user"]), str(row["pc"]), str(row["date"])), 
            axis=1
        )
        
        return normalized

# Create some sample data to demonstrate the pipeline
sample_data = {
    "logon": pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "date": ["01/02/2010 08:28:06", "01/02/2010 09:15:22", "01/02/2010 17:45:30", 
                "01/03/2010 08:30:15", "01/03/2010 18:00:45"],
        "user": ["ACM2278", "ACM2278", "ACM2278", "CMP2946", "CMP2946"],
        "pc": ["PC-1648", "PC-1648", "PC-1648", "PC-2190", "PC-2190"],
        "activity": ["Logon", "Logoff", "Logon", "Logon", "Logoff"]
    }),
    
    "file": pd.DataFrame({
        "id": [1, 2, 3],
        "date": ["01/02/2010 10:30:00", "01/02/2010 14:15:30", "01/03/2010 11:45:15"],
        "user": ["ACM2278", "ACM2278", "CMP2946"],
        "pc": ["PC-1648", "PC-1648", "PC-2190"],
        "filename": ["document1.txt", "report.docx", "data.xlsx"],
        "content": ["C:\\Users\\ACM2278\\Documents\\", "C:\\Shared\\Reports\\", "C:\\Data\\"]
    }),
    
    "email": pd.DataFrame({
        "id": [1, 2],
        "date": ["01/02/2010 13:22:10", "01/03/2010 09:30:25"],
        "user": ["ACM2278", "CMP2946"],
        "pc": ["PC-1648", "PC-2190"], 
        "to": ["colleague@company.com", "manager@company.com"],
        "cc": ["", "team@company.com"],
        "bcc": ["", ""],
        "from": ["ACM2278@company.com", "CMP2946@company.com"],
        "size": [1024, 2048],
        "attachments": [0, 1],
        "content": ["Meeting notes", "Weekly report"]
    })
}

print("CERT Data Ingestion Pipeline - Core Components Created Successfully")
print("\nUnified Schema Fields:")
for field, dtype in UNIFIED_SCHEMA.items():
    print(f"  {field}: {dtype}")
    
print(f"\nSupported CERT Files: {list(CERT_FILES.keys())}")
print("\nSample data created for demonstration")