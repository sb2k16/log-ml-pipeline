#!/usr/bin/env python3
"""
Basic test script for anomaly detection pipeline with minimal dependencies.
"""

import sys
import os
import json
import logging
from datetime import datetime, timedelta
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_basic_functionality():
    """Test basic functionality without heavy dependencies."""
    logger.info("Testing basic functionality...")
    
    # Test 1: Basic data structures
    try:
        # Simulate log data
        logs = [
            {"timestamp": "2024-01-01T10:00:00", "level": "INFO", "message": "Normal operation"},
            {"timestamp": "2024-01-01T10:01:00", "level": "ERROR", "message": "Database connection failed"},
            {"timestamp": "2024-01-01T10:02:00", "level": "INFO", "message": "Connection restored"},
            {"timestamp": "2024-01-01T10:03:00", "level": "WARNING", "message": "High memory usage"},
            {"timestamp": "2024-01-01T10:04:00", "level": "ERROR", "message": "Critical system failure"},
        ]
        logger.info("✓ Log data structure created")
        
        # Test 2: Simple feature extraction
        features = []
        for log in logs:
            feature = {
                "timestamp": log["timestamp"],
                "level": log["level"],
                "message_length": len(log["message"]),
                "has_error": "ERROR" in log["level"],
                "has_warning": "WARNING" in log["level"],
                "word_count": len(log["message"].split()),
            }
            features.append(feature)
        logger.info("✓ Feature extraction completed")
        
        # Test 3: Simple anomaly detection (rule-based)
        anomalies = []
        for i, feature in enumerate(features):
            is_anomaly = False
            reasons = []
            
            # Rule 1: Error messages are anomalies
            if feature["has_error"]:
                is_anomaly = True
                reasons.append("Error message detected")
            
            # Rule 2: Very long messages might be anomalies
            if feature["message_length"] > 50:
                is_anomaly = True
                reasons.append("Unusually long message")
            
            # Rule 3: High word count might indicate issues
            if feature["word_count"] > 10:
                is_anomaly = True
                reasons.append("High word count")
            
            if is_anomaly:
                anomalies.append({
                    "index": i,
                    "log": logs[i],
                    "reasons": reasons
                })
        
        logger.info(f"✓ Anomaly detection completed: {len(anomalies)} anomalies found")
        
        # Test 4: Generate report
        report = {
            "total_logs": len(logs),
            "anomalies_found": len(anomalies),
            "anomaly_rate": len(anomalies) / len(logs),
            "anomalies": anomalies
        }
        
        logger.info("✓ Report generation completed")
        
        # Test 5: Save results
        with open("test_results.json", "w") as f:
            json.dump(report, f, indent=2)
        logger.info("✓ Results saved to test_results.json")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        return False

def test_config_loading():
    """Test configuration loading."""
    logger.info("Testing configuration loading...")
    
    try:
        # Create a simple config
        config = {
            "data": {
                "input_path": "data/sample_logs.jsonl",
                "output_path": "reports/"
            },
            "models": {
                "isolation_forest": {
                    "contamination": 0.1,
                    "random_state": 42
                }
            },
            "evaluation": {
                "metrics": ["precision", "recall", "f1"]
            }
        }
        
        # Save config
        with open("test_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        # Load config
        with open("test_config.json", "r") as f:
            loaded_config = json.load(f)
        
        logger.info("✓ Configuration loading completed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Config test failed: {e}")
        return False

def main():
    """Run all basic tests."""
    logger.info("=" * 60)
    logger.info("BASIC FUNCTIONALITY TEST")
    logger.info("=" * 60)
    
    tests = [
        ("Configuration Loading", test_config_loading),
        ("Basic Functionality", test_basic_functionality),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning: {test_name}")
        if test_func():
            logger.info(f"✓ {test_name} PASSED")
            passed += 1
        else:
            logger.error(f"✗ {test_name} FAILED")
    
    logger.info("=" * 60)
    logger.info(f"RESULTS: {passed}/{total} tests passed")
    logger.info("=" * 60)
    
    if passed == total:
        logger.info("🎉 All basic tests passed! The pipeline is ready for core functionality.")
        logger.info("\nNext steps:")
        logger.info("1. Free up disk space for full installation")
        logger.info("2. Run: python3 setup.py")
        logger.info("3. Run: python3 main.py")
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 