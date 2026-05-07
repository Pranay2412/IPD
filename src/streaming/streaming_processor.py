# Create streaming processing components for real-time data ingestion
import asyncio
from datetime import datetime
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Generator, Callable, List
import json
import pandas as pd  # For DataFrame operations
import numpy as np
from src.utils import get_logger
logger = get_logger(__name__)
from src.normalization import CERTDataIngestionPipeline, normalize_logon_data, normalize_file_data, normalize_email_data, validate_data_quality
from pathlib import Path
from src.normalization import CERTDataIngestionPipeline
from src.validation import CERT_FILES

# Load actual CERT data (use full file, not just 1 row)
data_dir = Path("cert_data")  # Change to your actual folder
output_dir = Path("processed_data")

pipeline = CERTDataIngestionPipeline(data_dir=data_dir, output_dir=output_dir)

sample_data = {
    file_type: pipeline.load_cert_file(file_type)
    for file_type in CERT_FILES
}
class StreamingCERTProcessor:
    """
    Real-time streaming processor for CERT data
    Supports event-driven processing with configurable batch sizes and processing intervals
    """
    
    def __init__(self, pipeline: 'CERTDataIngestionPipeline', buffer_size: int = 1000, flush_interval: int = 5):
        self.pipeline = pipeline
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        
        # Streaming buffers for each data type
        self.buffers = {file_type: [] for file_type in CERT_FILES.keys()}
        self.buffer_locks = {file_type: threading.Lock() for file_type in CERT_FILES.keys()}
        
        # Event processing queues
        self.event_queue = queue.Queue(maxsize=10000)
        self.processed_queue = queue.Queue(maxsize=10000)
        
        # Threading components
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.is_running = False
        
        # Metrics tracking
        self.metrics = {
            "events_processed": 0,
            "events_failed": 0,
            "processing_rate": 0.0,
            "buffer_levels": {},
            "last_flush_time": time.time()
        }
        
        logger.info("Initialized Streaming CERT Processor")
    
    def simulate_real_time_data_stream(self) -> Generator[Dict[str, Any], None, None]:
        """
        Simulate real-time data stream from CERT dataset
        In production, this would connect to Kafka, file watchers, or API endpoints
        """
        # Use sample data and simulate streaming
        data_sources = sample_data
        
        while self.is_running:
            for file_type, df in data_sources.items():
                for _, row in df.iterrows():
                    # Add timestamp and metadata for streaming
                    event = {
                        "file_type": file_type,
                        "timestamp": datetime.now().isoformat(),
                        "data": row.to_dict(),
                        "source": "cert_simulator"
                    }
                    yield event
                    
                    # Simulate realistic data arrival rate
                    time.sleep(np.random.exponential(0.1))  # Average 10 events per second
    
    def kafka_consumer_simulation(self, topics: List[str]) -> Generator[Dict[str, Any], None, None]:
        """
        Simulate Kafka consumer for real-time data ingestion
        In production, this would use kafka-python or confluent-kafka
        """
        logger.info(f"Starting Kafka consumer simulation for topics: {topics}")
        
        # Simulate Kafka message consumption
        while self.is_running:
            for topic in topics:
                if topic in CERT_FILES:
                    # Simulate Kafka message
                    if topic in sample_data and not sample_data[topic].empty:
                        sample_row = sample_data[topic].sample(1).iloc[0]
                        kafka_message = {
                            "topic": topic,
                            "partition": 0,
                            "offset": int(time.time() * 1000),
                            "timestamp": datetime.now().isoformat(),
                            "key": f"user_{sample_row.get('user', 'unknown')}",
                            "value": sample_row.to_dict()
                        }
                        yield kafka_message
            
            time.sleep(0.05)  # 20 messages per second per topic
    
    def add_event_to_buffer(self, file_type: str, event_data: Dict[str, Any]):
        """Thread-safe method to add events to type-specific buffers"""
        with self.buffer_locks[file_type]:
            self.buffers[file_type].append(event_data)
            
            # Auto-flush if buffer is full
            if len(self.buffers[file_type]) >= self.buffer_size:
                self.flush_buffer(file_type)
    
    def flush_buffer(self, file_type: str) -> int:
        """Flush buffer and process accumulated events"""
        with self.buffer_locks[file_type]:
            if not self.buffers[file_type]:
                return 0
            
            buffer_data = self.buffers[file_type].copy()
            self.buffers[file_type].clear()
        
        # Convert buffer to DataFrame for processing
        df_data = [event["data"] if isinstance(event, dict) and "data" in event else event 
                  for event in buffer_data]
        
        if df_data:
            df = pd.DataFrame(df_data)
            
            # Process through normalization pipeline
            try:
                if file_type == "logon":
                    normalized_df = normalize_logon_data(df)
                elif file_type == "file":
                    normalized_df = normalize_file_data(df)
                elif file_type == "email":
                    normalized_df = normalize_email_data(df)
                else:
                    logger.warning(f"Unsupported file type for streaming: {file_type}")
                    return 0
                
                # Apply quality checks and cleaning
                quality_report = validate_data_quality(df, file_type)
                
                # Add to processed queue for further handling
                self.processed_queue.put({
                    "file_type": file_type,
                    "data": normalized_df,
                    "quality_report": quality_report,
                    "timestamp": datetime.now().isoformat(),
                    "record_count": len(normalized_df)
                })
                
                # Update metrics
                self.metrics["events_processed"] += len(buffer_data)
                self.metrics["buffer_levels"][file_type] = len(self.buffers[file_type])
                
                logger.info(f"Flushed {len(buffer_data)} {file_type} events")
                return len(buffer_data)
                
            except Exception as e:
                self.metrics["events_failed"] += len(buffer_data)
                logger.error(f"Error processing {file_type} buffer: {str(e)}")
                return 0
        
        return 0
    
    def periodic_flush(self):
        """Periodically flush all buffers based on time interval"""
        while self.is_running:
            time.sleep(self.flush_interval)
            
            total_flushed = 0
            for file_type in CERT_FILES.keys():
                flushed_count = self.flush_buffer(file_type)
                total_flushed += flushed_count
            
            if total_flushed > 0:
                logger.info(f"Periodic flush: processed {total_flushed} events")
            
            self.metrics["last_flush_time"] = time.time()
    
    def process_streaming_events(self, data_generator: Generator[Dict[str, Any], None, None]):
        """Main streaming event processor"""
        logger.info("Starting streaming event processing")
        
        for event in data_generator:
            if not self.is_running:
                break
            
            try:
                # Determine event type
                if "file_type" in event:
                    file_type = event["file_type"]
                elif "topic" in event:  # Kafka message
                    file_type = event["topic"]
                else:
                    logger.warning("Unknown event format, skipping")
                    continue
                
                if file_type in CERT_FILES:
                    self.add_event_to_buffer(file_type, event)
                else:
                    logger.warning(f"Unknown file type: {file_type}")
                    
            except Exception as e:
                self.metrics["events_failed"] += 1
                logger.error(f"Error processing streaming event: {str(e)}")
    
    def start_streaming(self, use_kafka_simulation: bool = False):
        """Start the streaming processor"""
        self.is_running = True
        
        # Start periodic flush thread
        flush_thread = threading.Thread(target=self.periodic_flush, daemon=True)
        flush_thread.start()
        
        # Choose data source
        if use_kafka_simulation:
            topics = list(CERT_FILES.keys())
            data_generator = self.kafka_consumer_simulation(topics)
            logger.info("Using Kafka consumer simulation")
        else:
            data_generator = self.simulate_real_time_data_stream()
            logger.info("Using real-time data stream simulation")
        
        # Process events
        try:
            self.process_streaming_events(data_generator)
        except KeyboardInterrupt:
            logger.info("Streaming processor interrupted by user")
        except Exception as e:
            logger.error(f"Streaming processor error: {str(e)}")
        finally:
            self.stop_streaming()
    
    def stop_streaming(self):
        """Stop the streaming processor and flush remaining data"""
        logger.info("Stopping streaming processor...")
        self.is_running = False
        
        # Final flush of all buffers
        total_final_flush = 0
        for file_type in CERT_FILES.keys():
            flushed_count = self.flush_buffer(file_type)
            total_final_flush += flushed_count
        
        if total_final_flush > 0:
            logger.info(f"Final flush: processed {total_final_flush} events")
        
        self.executor.shutdown(wait=True)
        logger.info("Streaming processor stopped")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current processing metrics"""
        current_time = time.time()
        time_elapsed = current_time - self.metrics["last_flush_time"]
        
        self.metrics["processing_rate"] = (self.metrics["events_processed"] / time_elapsed 
                                         if time_elapsed > 0 else 0)
        
        # Update buffer levels
        for file_type in CERT_FILES.keys():
            with self.buffer_locks[file_type]:
                self.metrics["buffer_levels"][file_type] = len(self.buffers[file_type])
        
        return self.metrics.copy()


class KafkaConnector:
    """
    Kafka integration for production streaming data ingestion
    This would integrate with confluent-kafka or kafka-python in production
    """
    
    def __init__(self, bootstrap_servers: str = "localhost:9092", group_id: str = "cert-consumer"):
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.consumer_config = {
            'bootstrap.servers': bootstrap_servers,
            'group.id': group_id,
            'auto.offset.reset': 'latest',
            'enable.auto.commit': True,
            'auto.commit.interval.ms': 1000,
            'max.poll.records': 100
        }
        
        self.producer_config = {
            'bootstrap.servers': bootstrap_servers,
            'acks': 'all',
            'retries': 3,
            'batch.size': 16384,
            'linger.ms': 10,
            'buffer.memory': 33554432
        }
        
        logger.info(f"Kafka connector initialized for {bootstrap_servers}")
    
    def create_consumer(self, topics: List[str]):
        """Create Kafka consumer (simulation)"""
        logger.info(f"Would create Kafka consumer for topics: {topics}")
        # In production: from confluent_kafka import Consumer
        # return Consumer(self.consumer_config)
        return None
    
    def create_producer(self):
        """Create Kafka producer (simulation)"""
        logger.info("Would create Kafka producer")
        # In production: from confluent_kafka import Producer  
        # return Producer(self.producer_config)
        return None
    
    def publish_processed_data(self, topic: str, data: Dict[str, Any]):
        """Publish processed data to output topic"""
        logger.info(f"Would publish to topic {topic}: {len(data)} records")
        # In production: producer.produce(topic, json.dumps(data))
        pass


# Demonstration of streaming capabilities
print("Streaming CERT Processor Created Successfully!")
print("\nStreaming Features:")
print("- Real-time event processing with configurable buffers")
print("- Kafka integration framework (simulation mode)")  
print("- Multi-threaded processing with thread-safe buffers")
print("- Automatic buffer flushing based on size and time")
print("- Comprehensive metrics and monitoring")
print("- Error handling and resilience")

print("\nConfiguration Options:")
print(f"- Buffer size: Configurable per data type")
print(f"- Flush interval: Time-based processing")
print(f"- Multi-threading: Concurrent processing")
print(f"- Quality validation: Real-time data quality checks")