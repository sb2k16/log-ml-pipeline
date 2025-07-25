"""
Kafka Consumer for Log Ingestion

Handles real-time log ingestion from Kafka topics with error handling,
monitoring, and batch processing capabilities.
"""

import json
import logging
import time
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
from kafka import KafkaConsumer, KafkaError
from kafka.consumer.fetcher import ConsumerRecord
import threading
from queue import Queue
import signal
import sys

logger = logging.getLogger(__name__)


class KafkaConsumer:
    """Kafka consumer for log ingestion with real-time processing."""
    
    def __init__(self, config: Dict[str, Any], message_handler: Optional[Callable] = None):
        self.config = config
        self.message_handler = message_handler
        self.consumer = None
        self.running = False
        self.message_queue = Queue(maxsize=10000)
        self.stats = {
            "messages_received": 0,
            "messages_processed": 0,
            "errors": 0,
            "last_message_time": None,
            "start_time": None
        }
        
        # Signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop()
        sys.exit(0)
    
    def connect(self) -> bool:
        """Connect to Kafka broker."""
        try:
            kafka_config = self.config.get("kafka", {})
            
            self.consumer = KafkaConsumer(
                kafka_config.get("topic", "log-stream"),
                bootstrap_servers=kafka_config.get("bootstrap_servers", "localhost:9092"),
                group_id=kafka_config.get("group_id", "anomaly-detection"),
                auto_offset_reset=kafka_config.get("auto_offset_reset", "latest"),
                enable_auto_commit=kafka_config.get("enable_auto_commit", True),
                auto_commit_interval_ms=kafka_config.get("auto_commit_interval_ms", 1000),
                value_deserializer=lambda x: x.decode('utf-8'),
                key_deserializer=lambda x: x.decode('utf-8') if x else None,
                consumer_timeout_ms=kafka_config.get("consumer_timeout_ms", 1000),
                max_poll_records=kafka_config.get("max_poll_records", 500),
                session_timeout_ms=kafka_config.get("session_timeout_ms", 30000),
                heartbeat_interval_ms=kafka_config.get("heartbeat_interval_ms", 3000)
            )
            
            logger.info(f"Connected to Kafka topic: {kafka_config.get('topic')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            return False
    
    def start(self):
        """Start consuming messages."""
        if not self.connect():
            logger.error("Failed to connect to Kafka, cannot start consumer")
            return
        
        self.running = True
        self.stats["start_time"] = datetime.now()
        
        # Start consumer thread
        consumer_thread = threading.Thread(target=self._consume_messages, daemon=True)
        consumer_thread.start()
        
        # Start processor thread
        processor_thread = threading.Thread(target=self._process_messages, daemon=True)
        processor_thread.start()
        
        logger.info("Kafka consumer started")
        
        try:
            # Keep main thread alive
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
            self.stop()
    
    def stop(self):
        """Stop the consumer gracefully."""
        logger.info("Stopping Kafka consumer...")
        self.running = False
        
        if self.consumer:
            self.consumer.close()
            logger.info("Kafka consumer stopped")
    
    def _consume_messages(self):
        """Consume messages from Kafka topic."""
        while self.running:
            try:
                # Poll for messages
                message_batch = self.consumer.poll(
                    timeout_ms=1000,
                    max_records=100
                )
                
                for topic_partition, messages in message_batch.items():
                    for message in messages:
                        if not self.running:
                            break
                        
                        try:
                            self._handle_message(message)
                            self.stats["messages_received"] += 1
                            self.stats["last_message_time"] = datetime.now()
                            
                        except Exception as e:
                            logger.error(f"Error handling message: {e}")
                            self.stats["errors"] += 1
                
            except KafkaError as e:
                logger.error(f"Kafka error: {e}")
                self.stats["errors"] += 1
                time.sleep(5)  # Wait before retrying
                
            except Exception as e:
                logger.error(f"Unexpected error in consumer: {e}")
                self.stats["errors"] += 1
                time.sleep(1)
    
    def _handle_message(self, message: ConsumerRecord):
        """Handle individual Kafka message."""
        try:
            # Parse message value
            if isinstance(message.value, str):
                data = json.loads(message.value)
            else:
                data = message.value
            
            # Add metadata
            message_data = {
                "value": data,
                "key": message.key,
                "topic": message.topic,
                "partition": message.partition,
                "offset": message.offset,
                "timestamp": message.timestamp,
                "headers": message.headers
            }
            
            # Add to processing queue
            if not self.message_queue.full():
                self.message_queue.put(message_data)
            else:
                logger.warning("Message queue is full, dropping message")
                
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON message: {e}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    def _process_messages(self):
        """Process messages from the queue."""
        batch_size = self.config.get("batch_size", 100)
        batch_timeout = self.config.get("batch_timeout", 5)  # seconds
        
        while self.running:
            try:
                batch = []
                start_time = time.time()
                
                # Collect batch
                while len(batch) < batch_size and (time.time() - start_time) < batch_timeout:
                    try:
                        message = self.message_queue.get(timeout=1)
                        batch.append(message)
                    except:
                        break
                
                if batch:
                    self._process_batch(batch)
                    self.stats["messages_processed"] += len(batch)
                
            except Exception as e:
                logger.error(f"Error processing message batch: {e}")
                self.stats["errors"] += 1
                time.sleep(1)
    
    def _process_batch(self, batch: List[Dict[str, Any]]):
        """Process a batch of messages."""
        if self.message_handler:
            try:
                self.message_handler(batch)
            except Exception as e:
                logger.error(f"Error in message handler: {e}")
                self.stats["errors"] += 1
        else:
            # Default processing: log the batch
            for message in batch:
                logger.info(f"Processed message: {message.get('value', '')}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get consumer statistics."""
        stats = self.stats.copy()
        
        if self.stats["start_time"]:
            uptime = datetime.now() - self.stats["start_time"]
            stats["uptime_seconds"] = uptime.total_seconds()
            
            if stats["messages_processed"] > 0:
                stats["messages_per_second"] = stats["messages_processed"] / uptime.total_seconds()
            else:
                stats["messages_per_second"] = 0
        
        stats["queue_size"] = self.message_queue.qsize()
        stats["running"] = self.running
        
        return stats
    
    def set_message_handler(self, handler: Callable):
        """Set custom message handler."""
        self.message_handler = handler
    
    def is_healthy(self) -> bool:
        """Check if consumer is healthy."""
        if not self.running:
            return False
        
        # Check if we're receiving messages
        if self.stats["last_message_time"]:
            time_since_last = (datetime.now() - self.stats["last_message_time"]).total_seconds()
            if time_since_last > 300:  # 5 minutes
                logger.warning("No messages received in the last 5 minutes")
                return False
        
        return True


class BatchKafkaConsumer(KafkaConsumer):
    """Kafka consumer optimized for batch processing."""
    
    def __init__(self, config: Dict[str, Any], batch_handler: Optional[Callable] = None):
        super().__init__(config, batch_handler)
        self.batch_size = config.get("batch_size", 1000)
        self.batch_timeout = config.get("batch_timeout", 30)  # seconds
        self.batch_buffer = []
        self.last_batch_time = time.time()
    
    def _process_messages(self):
        """Process messages in batches."""
        while self.running:
            try:
                # Collect messages for batch
                while len(self.batch_buffer) < self.batch_size:
                    try:
                        message = self.message_queue.get(timeout=1)
                        self.batch_buffer.append(message)
                    except:
                        break
                
                current_time = time.time()
                
                # Process batch if full or timeout reached
                if (len(self.batch_buffer) >= self.batch_size or 
                    (len(self.batch_buffer) > 0 and 
                     (current_time - self.last_batch_time) >= self.batch_timeout)):
                    
                    if self.batch_buffer:
                        self._process_batch(self.batch_buffer)
                        self.stats["messages_processed"] += len(self.batch_buffer)
                        self.batch_buffer = []
                        self.last_batch_time = current_time
                
            except Exception as e:
                logger.error(f"Error in batch processing: {e}")
                self.stats["errors"] += 1
                time.sleep(1)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batch consumer statistics."""
        stats = super().get_stats()
        stats["batch_buffer_size"] = len(self.batch_buffer)
        stats["time_since_last_batch"] = time.time() - self.last_batch_time
        return stats 