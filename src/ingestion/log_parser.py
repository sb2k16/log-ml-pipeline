"""
Log Parser Module

Parses various log formats and extracts structured features for anomaly detection.
"""

import re
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ParsedLog:
    """Structured representation of a parsed log entry."""
    timestamp: datetime
    level: str
    message: str
    source: str
    service: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    duration: Optional[float] = None
    status_code: Optional[int] = None
    method: Optional[str] = None
    url: Optional[str] = None
    user_agent: Optional[str] = None
    request_size: Optional[int] = None
    response_size: Optional[int] = None
    error_code: Optional[str] = None
    stack_trace: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class LogParser:
    """Parses various log formats into structured data."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.timestamp_format = config.get("timestamp_format", "%Y-%m-%d %H:%M:%S")
        self.default_levels = config.get("default_levels", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        self.custom_patterns = config.get("custom_patterns", [])
        
        # Compile regex patterns
        self.patterns = self._compile_patterns()
        
    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for log parsing."""
        patterns = {
            # Standard log format: timestamp [level] message
            "standard": re.compile(
                r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?)\s*\[?(\w+)\]?\s*(.+)'
            ),
            
            # JSON log format
            "json": re.compile(r'^\{.*\}$'),
            
            # HTTP request log
            "http_request": re.compile(
                r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+(\w+)\s+(\S+)\s+(\S+)\s+(\d+)\s+(\d+)'
            ),
            
            # Database query log
            "database_query": re.compile(
                r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+DB_QUERY:\s+(\w+)\s+duration=(\d+\.\d+)'
            ),
            
            # Error log with stack trace
            "error_with_stack": re.compile(
                r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+ERROR\s+(.+?)(?:\n\s+at\s+.+)*'
            ),
        }
        
        # Add custom patterns
        for pattern_config in self.custom_patterns:
            name = pattern_config["name"]
            pattern = pattern_config["pattern"]
            patterns[name] = re.compile(pattern)
            
        return patterns
    
    def parse_log_line(self, log_line: str) -> Optional[ParsedLog]:
        """Parse a single log line into structured data."""
        try:
            # Try JSON format first
            if self.patterns["json"].match(log_line.strip()):
                return self._parse_json_log(log_line)
            
            # Try standard format
            match = self.patterns["standard"].match(log_line)
            if match:
                return self._parse_standard_log(log_line, match)
            
            # Try HTTP request format
            match = self.patterns["http_request"].match(log_line)
            if match:
                return self._parse_http_request(log_line, match)
            
            # Try database query format
            match = self.patterns["database_query"].match(log_line)
            if match:
                return self._parse_database_query(log_line, match)
            
            # Try custom patterns
            for pattern_name, pattern in self.patterns.items():
                if pattern_name not in ["standard", "json", "http_request", "database_query", "error_with_stack"]:
                    match = pattern.match(log_line)
                    if match:
                        return self._parse_custom_log(log_line, match, pattern_name)
            
            # Fallback to basic parsing
            return self._parse_basic_log(log_line)
            
        except Exception as e:
            logger.warning(f"Failed to parse log line: {log_line[:100]}... Error: {e}")
            return None
    
    def _parse_json_log(self, log_line: str) -> ParsedLog:
        """Parse JSON formatted log."""
        data = json.loads(log_line)
        
        timestamp_str = data.get("timestamp", data.get("time", data.get("ts")))
        timestamp = self._parse_timestamp(timestamp_str)
        
        level = data.get("level", data.get("severity", "INFO"))
        message = data.get("message", data.get("msg", ""))
        source = data.get("source", data.get("service", "unknown"))
        
        return ParsedLog(
            timestamp=timestamp,
            level=level.upper(),
            message=message,
            source=source,
            service=data.get("service"),
            user_id=data.get("user_id"),
            session_id=data.get("session_id"),
            ip_address=data.get("ip"),
            duration=data.get("duration"),
            status_code=data.get("status_code"),
            method=data.get("method"),
            url=data.get("url"),
            user_agent=data.get("user_agent"),
            request_size=data.get("request_size"),
            response_size=data.get("response_size"),
            error_code=data.get("error_code"),
            stack_trace=data.get("stack_trace"),
            metadata=data
        )
    
    def _parse_standard_log(self, log_line: str, match: re.Match) -> ParsedLog:
        """Parse standard log format."""
        timestamp_str, level, message = match.groups()
        timestamp = self._parse_timestamp(timestamp_str)
        
        return ParsedLog(
            timestamp=timestamp,
            level=level.upper(),
            message=message,
            source="application"
        )
    
    def _parse_http_request(self, log_line: str, match: re.Match) -> ParsedLog:
        """Parse HTTP request log."""
        timestamp_str, ip, method, url, status_code, response_size = match.groups()
        timestamp = self._parse_timestamp(timestamp_str)
        
        return ParsedLog(
            timestamp=timestamp,
            level="INFO",
            message=f"{method} {url} {status_code}",
            source="web_server",
            ip_address=ip,
            method=method,
            url=url,
            status_code=int(status_code),
            response_size=int(response_size)
        )
    
    def _parse_database_query(self, log_line: str, match: re.Match) -> ParsedLog:
        """Parse database query log."""
        timestamp_str, query_type, duration = match.groups()
        timestamp = self._parse_timestamp(timestamp_str)
        
        return ParsedLog(
            timestamp=timestamp,
            level="INFO",
            message=f"DB_QUERY: {query_type}",
            source="database",
            duration=float(duration)
        )
    
    def _parse_custom_log(self, log_line: str, match: re.Match, pattern_name: str) -> ParsedLog:
        """Parse custom pattern log."""
        groups = match.groups()
        timestamp_str = groups[0] if groups else None
        timestamp = self._parse_timestamp(timestamp_str) if timestamp_str else datetime.now()
        
        return ParsedLog(
            timestamp=timestamp,
            level="INFO",
            message=log_line,
            source=pattern_name
        )
    
    def _parse_basic_log(self, log_line: str) -> ParsedLog:
        """Basic fallback parsing."""
        return ParsedLog(
            timestamp=datetime.now(),
            level="INFO",
            message=log_line,
            source="unknown"
        )
    
    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp string to datetime object."""
        if not timestamp_str:
            return datetime.now()
        
        try:
            return datetime.strptime(timestamp_str, self.timestamp_format)
        except ValueError:
            # Try common timestamp formats
            formats = [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%S.%f+00:00"
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(timestamp_str, fmt)
                except ValueError:
                    continue
            
            logger.warning(f"Could not parse timestamp: {timestamp_str}")
            return datetime.now()
    
    def parse_batch(self, log_lines: List[str]) -> List[ParsedLog]:
        """Parse a batch of log lines."""
        parsed_logs = []
        
        for log_line in log_lines:
            parsed_log = self.parse_log_line(log_line.strip())
            if parsed_log:
                parsed_logs.append(parsed_log)
        
        return parsed_logs
    
    def to_dataframe(self, parsed_logs: List[ParsedLog]) -> pd.DataFrame:
        """Convert parsed logs to pandas DataFrame."""
        if not parsed_logs:
            return pd.DataFrame()
        
        data = []
        for log in parsed_logs:
            row = {
                "timestamp": log.timestamp,
                "level": log.level,
                "message": log.message,
                "source": log.source,
                "service": log.service,
                "user_id": log.user_id,
                "session_id": log.session_id,
                "ip_address": log.ip_address,
                "duration": log.duration,
                "status_code": log.status_code,
                "method": log.method,
                "url": log.url,
                "user_agent": log.user_agent,
                "request_size": log.request_size,
                "response_size": log.response_size,
                "error_code": log.error_code,
                "stack_trace": log.stack_trace,
            }
            
            # Add metadata fields
            if log.metadata:
                for key, value in log.metadata.items():
                    if key not in row:
                        row[f"meta_{key}"] = value
            
            data.append(row)
        
        return pd.DataFrame(data) 