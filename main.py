#!/usr/bin/env python3
"""
Main Anomaly Detection Pipeline

Orchestrates the complete anomaly detection pipeline:
1. Data ingestion and parsing
2. Feature engineering
3. Model training and prediction
4. Evaluation and benchmarking
5. Visualization and reporting
"""

import logging
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import argparse
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.ingestion.log_parser import LogParser
from src.processing.feature_engineer import FeatureEngineer
from src.models.isolation_forest import IsolationForestDetector
from src.models.rule_based import RuleBasedDetector
from src.evaluation.metrics import AnomalyDetectionMetrics


class AnomalyDetectionPipeline:
    """Main pipeline for anomaly detection in log data."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.setup_logging()
        
        # Initialize components
        self.log_parser = LogParser(self.config.get("log_parsing", {}))
        self.feature_engineer = FeatureEngineer(self.config)
        self.metrics = AnomalyDetectionMetrics(self.config)
        
        # Initialize models
        self.models = {}
        self._initialize_models()
        
        # Results storage
        self.results = {}
        self.evaluation_results = []
        
        logger.info("Anomaly Detection Pipeline initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return {}
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get("logging", {})
        log_level = getattr(logging, log_config.get("level", "INFO"))
        
        logging.basicConfig(
            level=log_level,
            format=log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_config.get("file", "logs/anomaly_detection.log"))
            ]
        )
    
    def _initialize_models(self):
        """Initialize all anomaly detection models."""
        # Isolation Forest
        self.models["isolation_forest"] = IsolationForestDetector(self.config)
        
        # Rule-based detector
        self.models["rule_based"] = RuleBasedDetector(self.config)
        
        logger.info(f"Initialized {len(self.models)} models")
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load and parse log data."""
        logger.info(f"Loading data from {data_path}")
        
        try:
            # Read JSONL file
            with open(data_path, 'r') as f:
                log_lines = f.readlines()
            
            # Parse logs
            parsed_logs = self.log_parser.parse_batch(log_lines)
            
            # Convert to DataFrame
            df = self.log_parser.to_dataframe(parsed_logs)
            
            logger.info(f"Loaded {len(df)} log entries")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data from {data_path}: {e}")
            return pd.DataFrame()
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for anomaly detection."""
        logger.info("Engineering features...")
        
        try:
            df_features = self.feature_engineer.engineer_features(df)
            logger.info(f"Feature engineering complete. Shape: {df_features.shape}")
            return df_features
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            return df
    
    def train_models(self, df: pd.DataFrame):
        """Train all models on the data."""
        logger.info("Training models...")
        
        for model_name, model in self.models.items():
            try:
                logger.info(f"Training {model_name}...")
                model.fit(df)
                logger.info(f"{model_name} training completed")
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
    
    def predict_anomalies(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Predict anomalies using all models."""
        logger.info("Predicting anomalies...")
        
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                logger.info(f"Running predictions with {model_name}...")
                pred = model.predict(df)
                predictions[model_name] = pred
                logger.info(f"{model_name} predictions completed")
            except Exception as e:
                logger.error(f"Failed to predict with {model_name}: {e}")
        
        return predictions
    
    def evaluate_models(self, df: pd.DataFrame, predictions: Dict[str, np.ndarray]):
        """Evaluate model performance."""
        logger.info("Evaluating models...")
        
        # Create synthetic ground truth for demonstration
        # In real scenarios, this would come from labeled data
        ground_truth = self._create_synthetic_ground_truth(df)
        
        for model_name, pred in predictions.items():
            try:
                logger.info(f"Evaluating {model_name}...")
                
                # Get scores if available
                scores = None
                if hasattr(self.models[model_name], 'predict_proba'):
                    scores = self.models[model_name].predict_proba(df)
                elif hasattr(self.models[model_name], 'get_anomaly_scores'):
                    scores = self.models[model_name].get_anomaly_scores(df)
                
                # Evaluate
                result = self.metrics.evaluate_model(
                    ground_truth, pred, scores, model_name
                )
                
                self.evaluation_results.append(result)
                logger.info(f"{model_name} evaluation completed")
                
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {e}")
    
    def _create_synthetic_ground_truth(self, df: pd.DataFrame) -> np.ndarray:
        """Create synthetic ground truth for evaluation."""
        # In a real scenario, this would be actual labeled data
        # For demonstration, we'll create synthetic anomalies based on log levels
        
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
        
        # Mark statistical outliers as anomalies
        if 'message_length' in df.columns:
            q1 = df['message_length'].quantile(0.25)
            q3 = df['message_length'].quantile(0.75)
            iqr = q3 - q1
            outlier_mask = (df['message_length'] < (q1 - 1.5 * iqr)) | (df['message_length'] > (q3 + 1.5 * iqr))
            ground_truth[outlier_mask] = 1
        
        return ground_truth
    
    def generate_report(self):
        """Generate comprehensive evaluation report."""
        logger.info("Generating evaluation report...")
        
        if not self.evaluation_results:
            logger.warning("No evaluation results to report")
            return
        
        # Generate report
        report = self.metrics.generate_report(self.evaluation_results)
        
        # Save report
        report_path = "reports/evaluation_report.txt"
        Path("reports").mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Report saved to {report_path}")
        print("\n" + "="*60)
        print("EVALUATION REPORT")
        print("="*60)
        print(report)
    
    def run_pipeline(self, data_path: str):
        """Run the complete anomaly detection pipeline."""
        logger.info("Starting anomaly detection pipeline...")
        
        try:
            # Step 1: Load and parse data
            df = self.load_data(data_path)
            if df.empty:
                logger.error("No data loaded, exiting pipeline")
                return
            
            # Step 2: Engineer features
            df_features = self.engineer_features(df)
            
            # Step 3: Train models
            self.train_models(df_features)
            
            # Step 4: Predict anomalies
            predictions = self.predict_anomalies(df_features)
            
            # Step 5: Evaluate models
            self.evaluate_models(df_features, predictions)
            
            # Step 6: Generate report
            self.generate_report()
            
            logger.info("Pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Anomaly Detection Pipeline")
    parser.add_argument("--config", default="config/config.yaml", help="Configuration file path")
    parser.add_argument("--data", default="data/sample_logs.jsonl", help="Input data file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    
    # Create and run pipeline
    pipeline = AnomalyDetectionPipeline(args.config)
    pipeline.run_pipeline(args.data)


if __name__ == "__main__":
    main() 