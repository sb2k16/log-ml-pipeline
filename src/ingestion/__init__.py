"""
Log Ingestion Module

Handles log data ingestion from various sources including:
- Kafka streams
- File systems
- HTTP endpoints
- Database queries
"""

from .kafka_consumer import KafkaConsumer
from .file_reader import FileReader
from .log_parser import LogParser

__all__ = ["KafkaConsumer", "FileReader", "LogParser"] 