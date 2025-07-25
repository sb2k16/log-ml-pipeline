#!/usr/bin/env python3
"""
Test Script for Anomaly Detection Pipeline

Demonstrates the complete pipeline functionality:
1. Load sample data
2. Engineer features
3. Train models
4. Detect anomalies
5. Evaluate performance
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.ingestion.log_parser import LogParser
from src.processing.feature_engineer import FeatureEngineer
from src.models.isolation_forest import IsolationForestDetector
from src.models.rule_based import RuleBasedDetector
from src.evaluation.metrics import AnomalyDetectionMetrics
import yaml
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration."""
    try:
        with open("config/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {}

def test_log_parsing():
    """Test log parsing functionality."""
    logger.info("Testing log parsing...")
    
    config = load_config()
    parser = LogParser(config.get("log_parsing", {}))
    
    # Sample log lines
    sample_logs = [
        '{"timestamp": "2024-01-15 09:15:23", "level": "INFO", "message": "User login successful", "source": "auth_service"}',
        '{"timestamp": "2024-01-15 09:15:30", "level": "ERROR", "message": "Connection timeout", "source": "api_gateway"}',
        '{"timestamp": "2024-01-15 09:15:35", "level": "WARNING", "message": "High memory usage detected", "source": "monitoring"}'
    ]
    
    # Parse logs
    parsed_logs = parser.parse_batch(sample_logs)
    df = parser.to_dataframe(parsed_logs)
    
    logger.info(f"Parsed {len(df)} log entries")
    logger.info(f"Columns: {list(df.columns)}")
    
    return df

def test_feature_engineering(df):
    """Test feature engineering."""
    logger.info("Testing feature engineering...")
    
    config = load_config()
    feature_engineer = FeatureEngineer(config)
    
    # Engineer features
    df_features = feature_engineer.engineer_features(df)
    
    logger.info(f"Feature engineering complete. Shape: {df_features.shape}")
    logger.info(f"Feature columns: {list(df_features.columns)}")
    
    return df_features

def test_model_training(df):
    """Test model training."""
    logger.info("Testing model training...")
    
    config = load_config()
    models = {}
    
    # Initialize models
    models["isolation_forest"] = IsolationForestDetector(config)
    models["rule_based"] = RuleBasedDetector(config)
    
    # Train models
    for model_name, model in models.items():
        try:
            logger.info(f"Training {model_name}...")
            model.fit(df)
            logger.info(f"{model_name} training completed")
        except Exception as e:
            logger.error(f"Failed to train {model_name}: {e}")
    
    return models

def test_anomaly_detection(df, models):
    """Test anomaly detection."""
    logger.info("Testing anomaly detection...")
    
    predictions = {}
    
    for model_name, model in models.items():
        try:
            logger.info(f"Running predictions with {model_name}...")
            pred = model.predict(df)
            predictions[model_name] = pred
            
            anomaly_count = np.sum(pred)
            logger.info(f"{model_name} detected {anomaly_count} anomalies out of {len(df)} samples")
            
        except Exception as e:
            logger.error(f"Failed to predict with {model_name}: {e}")
    
    return predictions

def test_evaluation(df, predictions):
    """Test model evaluation."""
    logger.info("Testing model evaluation...")
    
    config = load_config()
    metrics = AnomalyDetectionMetrics(config)
    
    # Create synthetic ground truth
    ground_truth = np.zeros(len(df))
    
    # Mark ERROR and CRITICAL logs as anomalies
    if 'level' in df.columns:
        error_mask = df['level'].isin(['ERROR', 'CRITICAL'])
        ground_truth[error_mask] = 1
    
    # Mark unusual patterns as anomalies
    if 'message' in df.columns:
        anomaly_patterns = ['timeout', 'failed', 'crash', 'breach', 'corruption']
        for pattern in anomaly_patterns:
            pattern_mask = df['message'].str.contains(pattern, case=False, na=False)
            ground_truth[pattern_mask] = 1
    
    # Evaluate each model
    results = []
    for model_name, pred in predictions.items():
        try:
            logger.info(f"Evaluating {model_name}...")
            
            # Get scores if available
            scores = None
            if hasattr(models[model_name], 'predict_proba'):
                scores = models[model_name].predict_proba(df)
            elif hasattr(models[model_name], 'get_anomaly_scores'):
                scores = models[model_name].get_anomaly_scores(df)
            
            # Evaluate
            result = metrics.evaluate_model(ground_truth, pred, scores, model_name)
            results.append(result)
            
            # Print key metrics
            basic_metrics = result.get("basic_metrics", {})
            logger.info(f"{model_name} - Precision: {basic_metrics.get('precision', 0):.4f}, "
                       f"Recall: {basic_metrics.get('recall', 0):.4f}, "
                       f"F1: {basic_metrics.get('f1_score', 0):.4f}")
            
        except Exception as e:
            logger.error(f"Failed to evaluate {model_name}: {e}")
    
    return results

def main():
    """Run the complete test pipeline."""
    logger.info("Starting anomaly detection pipeline test...")
    
    try:
        # Step 1: Test log parsing
        df = test_log_parsing()
        
        if df.empty:
            logger.error("No data loaded, exiting")
            return
        
        # Step 2: Test feature engineering
        df_features = test_feature_engineering(df)
        
        # Step 3: Test model training
        models = test_model_training(df_features)
        
        # Step 4: Test anomaly detection
        predictions = test_anomaly_detection(df_features, models)
        
        # Step 5: Test evaluation
        results = test_evaluation(df_features, predictions)
        
        # Step 6: Generate summary
        logger.info("=" * 60)
        logger.info("PIPELINE TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total log entries processed: {len(df)}")
        logger.info(f"Features engineered: {len(df_features.columns)}")
        logger.info(f"Models trained: {len(models)}")
        logger.info(f"Models evaluated: {len(results)}")
        
        # Print best model
        if results:
            best_model = max(results, key=lambda x: x.get("basic_metrics", {}).get("f1_score", 0))
            logger.info(f"Best model: {best_model['model_name']}")
            logger.info(f"Best F1 score: {best_model['basic_metrics'].get('f1_score', 0):.4f}")
        
        logger.info("Pipeline test completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
        raise

if __name__ == "__main__":
    main() 