#!/usr/bin/env python3
"""
Setup Script for Anomaly Detection Pipeline

Installs dependencies, creates necessary directories, and configures the pipeline.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary directories."""
    directories = [
        "logs",
        "reports", 
        "models",
        "data",
        "config",
        "src/ingestion",
        "src/processing",
        "src/models",
        "src/evaluation",
        "src/api",
        "tests",
        "notebooks",
        "docker",
        "k8s",
        "grafana/dashboards",
        "grafana/datasources",
        "prometheus"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def install_dependencies():
    """Install Python dependencies in a virtual environment."""
    logger.info("Setting up virtual environment and installing dependencies...")
    
    try:
        # Create virtual environment
        venv_path = Path("venv")
        if not venv_path.exists():
            logger.info("Creating virtual environment...")
            subprocess.check_call([sys.executable, "-m", "venv", "venv"])
            logger.info("Virtual environment created successfully")
        
        # Determine the correct pip path
        if os.name == 'nt':  # Windows
            pip_path = venv_path / "Scripts" / "pip"
        else:  # Unix/Linux/macOS
            pip_path = venv_path / "bin" / "pip"
        
        # Install minimal dependencies first
        logger.info("Installing minimal dependencies in virtual environment...")
        try:
            subprocess.check_call([str(pip_path), "install", "-r", "requirements-minimal.txt"])
            logger.info("Minimal dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install minimal dependencies: {e}")
            logger.info("Trying with individual packages...")
            try:
                subprocess.check_call([str(pip_path), "install", "pandas", "numpy", "scikit-learn"])
                logger.info("Core ML packages installed successfully")
            except subprocess.CalledProcessError as e2:
                logger.error(f"Failed to install core packages: {e2}")
                return False
        
        # Try to install additional dependencies if space allows
        try:
            logger.info("Installing additional dependencies...")
            subprocess.check_call([str(pip_path), "install", "-r", "requirements-basic.txt"])
            logger.info("Additional dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Some optional dependencies failed to install: {e}")
            logger.info("Basic functionality is available with minimal dependencies")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        logger.info("You can manually create a virtual environment and install dependencies:")
        logger.info("python3 -m venv venv")
        logger.info("source venv/bin/activate  # On macOS/Linux")
        logger.info("pip install -r requirements.txt")
        return False

def create_config_files():
    """Create default configuration files if they don't exist."""
    config_files = {
        "config/config.yaml": """# Anomaly Detection Pipeline Configuration

# Data Sources
data_sources:
  kafka:
    bootstrap_servers: "localhost:9092"
    topic: "log-stream"
    group_id: "anomaly-detection"
    auto_offset_reset: "latest"
  
  file:
    path: "data/sample_logs.jsonl"
    batch_size: 1000
    max_lines: 100000

# Log Parsing
log_parsing:
  timestamp_format: "%Y-%m-%d %H:%M:%S"
  default_levels: ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# Feature Engineering
feature_engineering:
  time_features:
    - hour_of_day
    - day_of_week
    - is_weekend
    - is_business_hour
  
  text_features:
    - log_level_encoding
    - message_length
    - word_count
    - special_char_count

# ML Models Configuration
models:
  isolation_forest:
    contamination: 0.1
    n_estimators: 100
    max_samples: "auto"
    random_state: 42
  
  one_class_svm:
    kernel: "rbf"
    nu: 0.1
    gamma: "scale"

# Rule-based Methods
rule_based:
  statistical:
    z_score_threshold: 3.0
    iqr_multiplier: 1.5
    rolling_window: 3600
  
  pattern_matching:
    error_patterns:
      - "ERROR"
      - "Exception"
      - "Failed"
      - "Timeout"
    warning_patterns:
      - "WARNING"
      - "Deprecated"
      - "Slow"

# Evaluation Metrics
evaluation:
  metrics:
    - precision
    - recall
    - f1_score
    - roc_auc
    - average_precision
  
  thresholds:
    precision_threshold: 0.8
    recall_threshold: 0.7
    f1_threshold: 0.75

# API Configuration
api:
  host: "0.0.0.0"
  port: 8000
  debug: false
  workers: 4
  timeout: 30

# Logging
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/anomaly_detection.log"
  max_size: "100MB"
  backup_count: 5
""",
        
        "prometheus/prometheus.yml": """global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'anomaly-detection'
    static_configs:
      - targets: ['anomaly-detection:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
""",
        
        "grafana/datasources/prometheus.yml": """apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
""",
        
        "grafana/dashboards/anomaly-detection.json": """{
  "dashboard": {
    "id": null,
    "title": "Anomaly Detection Dashboard",
    "tags": ["anomaly-detection"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Anomaly Detection Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "anomaly_detection_rate",
            "refId": "A"
          }
        ]
      },
      {
        "id": 2,
        "title": "False Positive Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "false_positive_rate",
            "refId": "A"
          }
        ]
      },
      {
        "id": 3,
        "title": "Processing Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "processing_latency_seconds",
            "refId": "A"
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "5s"
  }
}"""
    }
    
    for file_path, content in config_files.items():
        if not Path(file_path).exists():
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w') as f:
                f.write(content)
            logger.info(f"Created config file: {file_path}")

def create_sample_data():
    """Create sample log data if it doesn't exist."""
    sample_logs_path = "data/sample_logs.jsonl"
    
    if not Path(sample_logs_path).exists():
        logger.info("Creating sample log data...")
        
        sample_logs = [
            {"timestamp": "2024-01-15 09:15:23", "level": "INFO", "message": "User login successful", "source": "auth_service", "user_id": "user123", "ip_address": "192.168.1.100"},
            {"timestamp": "2024-01-15 09:15:25", "level": "INFO", "message": "Database query executed", "source": "database", "duration": 0.045, "query_type": "SELECT"},
            {"timestamp": "2024-01-15 09:15:30", "level": "ERROR", "message": "Connection timeout", "source": "api_gateway", "error_code": "TIMEOUT_001"},
            {"timestamp": "2024-01-15 09:15:35", "level": "WARNING", "message": "High memory usage detected", "source": "monitoring", "memory_usage": 85.2},
            {"timestamp": "2024-01-15 09:16:00", "level": "INFO", "message": "HTTP GET /api/users", "source": "web_server", "method": "GET", "url": "/api/users", "status_code": 200, "response_size": 1024},
            {"timestamp": "2024-01-15 09:16:05", "level": "ERROR", "message": "Database connection failed", "source": "database", "error_code": "DB_CONN_001"},
            {"timestamp": "2024-01-15 09:16:10", "level": "INFO", "message": "Cache miss", "source": "cache_service", "cache_key": "user_profile_123"},
            {"timestamp": "2024-01-15 09:16:15", "level": "INFO", "message": "Email sent successfully", "source": "email_service", "recipient": "user@example.com"},
            {"timestamp": "2024-01-15 09:16:20", "level": "ERROR", "message": "Invalid authentication token", "source": "auth_service", "error_code": "AUTH_001"},
            {"timestamp": "2024-01-15 09:16:25", "level": "INFO", "message": "File uploaded", "source": "file_service", "file_size": 2048576, "file_type": "image/jpeg"}
        ]
        
        with open(sample_logs_path, 'w') as f:
            for log in sample_logs:
                f.write(f"{log}\n")
        
        logger.info(f"Created sample log data: {sample_logs_path}")

def run_tests():
    """Run basic tests to verify installation."""
    logger.info("Running basic tests...")
    
    try:
        # Test imports
        import pandas as pd
        import numpy as np
        import sklearn
        import fastapi
        
        logger.info("✓ All required packages imported successfully")
        
        # Test pipeline components
        sys.path.append(str(Path(__file__).parent / "src"))
        
        from src.ingestion.log_parser import LogParser
        from src.processing.feature_engineer import FeatureEngineer
        from src.models.isolation_forest import IsolationForestDetector
        from src.models.rule_based import RuleBasedDetector
        from src.evaluation.metrics import AnomalyDetectionMetrics
        
        logger.info("✓ All pipeline components imported successfully")
        
        return True
        
    except ImportError as e:
        logger.error(f"✗ Import test failed: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        return False

def main():
    """Main setup function."""
    logger.info("Setting up Anomaly Detection Pipeline...")
    
    # Step 1: Create directories
    create_directories()
    
    # Step 2: Install dependencies
    if not install_dependencies():
        logger.error("Failed to install dependencies")
        return False
    
    # Step 3: Create config files
    create_config_files()
    
    # Step 4: Create sample data
    create_sample_data()
    
    # Step 5: Run tests
    if not run_tests():
        logger.error("Tests failed")
        return False
    
    logger.info("=" * 60)
    logger.info("SETUP COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    logger.info("Next steps:")
    logger.info("1. Activate virtual environment: source venv/bin/activate")
    logger.info("2. Run the pipeline: python3 main.py")
    logger.info("3. Test the pipeline: python3 test_pipeline.py")
    logger.info("4. Start the API: python3 src/api/app.py")
    logger.info("5. Run with Docker: docker-compose up -d")
    logger.info("=" * 60)
    logger.info("Note: Always activate the virtual environment before running the pipeline!")
    logger.info("To deactivate: deactivate")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 