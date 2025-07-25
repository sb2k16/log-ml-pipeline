#!/usr/bin/env python3
"""
Simplified Anomaly Detection Pipeline
Works with minimal dependencies for demonstration purposes.
"""

import json
import logging
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleLogParser:
    """Simple log parser for JSON logs."""
    
    def __init__(self):
        self.parsed_logs = []
    
    def parse_logs(self, logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse raw logs into structured format."""
        logger.info(f"Parsing {len(logs)} logs...")
        
        parsed = []
        for log in logs:
            parsed_log = {
                "timestamp": log.get("timestamp", ""),
                "level": log.get("level", "INFO"),
                "message": log.get("message", ""),
                "source": log.get("source", "unknown"),
                "parsed_at": datetime.now().isoformat()
            }
            parsed.append(parsed_log)
        
        self.parsed_logs = parsed
        logger.info(f"✓ Parsed {len(parsed)} logs")
        return parsed

class SimpleFeatureEngineer:
    """Simple feature engineer for log data."""
    
    def __init__(self):
        self.features = []
    
    def extract_features(self, logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract features from parsed logs."""
        logger.info("Extracting features...")
        
        features = []
        for log in logs:
            feature = {
                "timestamp": log["timestamp"],
                "level": log["level"],
                "message_length": len(log["message"]),
                "word_count": len(log["message"].split()),
                "has_error": "ERROR" in log["level"].upper(),
                "has_warning": "WARNING" in log["level"].upper(),
                "has_info": "INFO" in log["level"].upper(),
                "special_chars": sum(1 for c in log["message"] if not c.isalnum() and c != ' '),
                "hour": datetime.fromisoformat(log["timestamp"].replace('Z', '+00:00')).hour if log["timestamp"] else 0,
                "is_business_hours": 9 <= datetime.fromisoformat(log["timestamp"].replace('Z', '+00:00')).hour <= 17 if log["timestamp"] else True,
            }
            features.append(feature)
        
        self.features = features
        logger.info(f"✓ Extracted features from {len(features)} logs")
        return features

class SimpleAnomalyDetector:
    """Simple rule-based anomaly detector."""
    
    def __init__(self):
        self.rules = [
            {"name": "error_messages", "condition": lambda f: f["has_error"], "weight": 0.8},
            {"name": "long_messages", "condition": lambda f: f["message_length"] > 100, "weight": 0.3},
            {"name": "high_word_count", "condition": lambda f: f["word_count"] > 15, "weight": 0.4},
            {"name": "many_special_chars", "condition": lambda f: f["special_chars"] > 10, "weight": 0.2},
            {"name": "non_business_hours", "condition": lambda f: not f["is_business_hours"], "weight": 0.1},
        ]
    
    def detect_anomalies(self, features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect anomalies using rule-based approach."""
        logger.info("Detecting anomalies...")
        
        anomalies = []
        for i, feature in enumerate(features):
            score = 0.0
            triggered_rules = []
            
            for rule in self.rules:
                if rule["condition"](feature):
                    score += rule["weight"]
                    triggered_rules.append(rule["name"])
            
            if score > 0.5:  # Threshold for anomaly
                anomaly = {
                    "index": i,
                    "feature": feature,
                    "score": score,
                    "triggered_rules": triggered_rules,
                    "is_anomaly": True
                }
                anomalies.append(anomaly)
        
        logger.info(f"✓ Detected {len(anomalies)} anomalies")
        return anomalies

class SimpleEvaluator:
    """Simple evaluator for anomaly detection results."""
    
    def __init__(self):
        self.metrics = {}
    
    def evaluate(self, features: List[Dict[str, Any]], anomalies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate the anomaly detection performance."""
        logger.info("Evaluating results...")
        
        total_logs = len(features)
        anomalies_found = len(anomalies)
        anomaly_rate = anomalies_found / total_logs if total_logs > 0 else 0
        
        # Calculate some basic metrics
        error_count = sum(1 for f in features if f["has_error"])
        warning_count = sum(1 for f in features if f["has_warning"])
        
        metrics = {
            "total_logs": total_logs,
            "anomalies_found": anomalies_found,
            "anomaly_rate": anomaly_rate,
            "error_count": error_count,
            "warning_count": warning_count,
            "avg_message_length": sum(f["message_length"] for f in features) / total_logs if total_logs > 0 else 0,
            "avg_word_count": sum(f["word_count"] for f in features) / total_logs if total_logs > 0 else 0,
        }
        
        self.metrics = metrics
        logger.info("✓ Evaluation completed")
        return metrics

def generate_sample_logs(num_logs: int = 100) -> List[Dict[str, Any]]:
    """Generate sample log data for testing."""
    logger.info(f"Generating {num_logs} sample logs...")
    
    logs = []
    base_time = datetime.now() - timedelta(hours=1)
    
    levels = ["INFO", "WARNING", "ERROR"]
    messages = [
        "User login successful",
        "Database query executed",
        "File uploaded successfully",
        "Memory usage at 75%",
        "Network connection established",
        "Cache miss occurred",
        "Database connection timeout",
        "Critical system error detected",
        "Service restart initiated",
        "Backup completed successfully",
        "Authentication failed",
        "Disk space low",
        "API rate limit exceeded",
        "SSL certificate expired",
        "Load balancer health check failed",
    ]
    
    for i in range(num_logs):
        timestamp = base_time + timedelta(minutes=i)
        level = random.choices(levels, weights=[0.7, 0.2, 0.1])[0]
        message = random.choice(messages)
        
        log = {
            "timestamp": timestamp.isoformat(),
            "level": level,
            "message": message,
            "source": f"service-{random.randint(1, 5)}"
        }
        logs.append(log)
    
    logger.info(f"✓ Generated {len(logs)} sample logs")
    return logs

def main():
    """Main pipeline execution."""
    logger.info("=" * 60)
    logger.info("SIMPLIFIED ANOMALY DETECTION PIPELINE")
    logger.info("=" * 60)
    
    try:
        # Step 1: Generate sample data
        logs = generate_sample_logs(50)
        
        # Step 2: Parse logs
        parser = SimpleLogParser()
        parsed_logs = parser.parse_logs(logs)
        
        # Step 3: Extract features
        feature_engineer = SimpleFeatureEngineer()
        features = feature_engineer.extract_features(parsed_logs)
        
        # Step 4: Detect anomalies
        detector = SimpleAnomalyDetector()
        anomalies = detector.detect_anomalies(features)
        
        # Step 5: Evaluate results
        evaluator = SimpleEvaluator()
        metrics = evaluator.evaluate(features, anomalies)
        
        # Step 6: Generate report
        report = {
            "pipeline_info": {
                "name": "Simplified Anomaly Detection Pipeline",
                "version": "1.0.0",
                "timestamp": datetime.now().isoformat()
            },
            "metrics": metrics,
            "anomalies": anomalies[:10],  # Show first 10 anomalies
            "summary": {
                "total_processed": len(logs),
                "anomalies_detected": len(anomalies),
                "detection_rate": f"{metrics['anomaly_rate']:.2%}",
                "most_common_anomaly_type": "Error messages" if anomalies else "None"
            }
        }
        
        # Save results
        os.makedirs("reports", exist_ok=True)
        with open("reports/simple_pipeline_results.json", "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"Total logs processed: {metrics['total_logs']}")
        logger.info(f"Anomalies detected: {metrics['anomalies_found']}")
        logger.info(f"Detection rate: {metrics['anomaly_rate']:.2%}")
        logger.info(f"Results saved to: reports/simple_pipeline_results.json")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 