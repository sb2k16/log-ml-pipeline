"""
Anomaly Detection Metrics

Provides comprehensive evaluation metrics for anomaly detection models
including precision, recall, F1-score, ROC curves, and custom metrics.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from sklearn.model_selection import cross_val_score
import time

logger = logging.getLogger(__name__)


class AnomalyDetectionMetrics:
    """Comprehensive metrics for anomaly detection evaluation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("evaluation", {})
        self.metrics = self.config.get("metrics", [])
        self.thresholds = self.config.get("thresholds", {})
        
        # Performance tracking
        self.evaluation_times = {}
        self.metric_history = []
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      y_scores: Optional[np.ndarray] = None,
                      model_name: str = "unknown") -> Dict[str, Any]:
        """Evaluate a model with comprehensive metrics."""
        logger.info(f"Evaluating model: {model_name}")
        
        start_time = time.time()
        
        # Basic classification metrics
        basic_metrics = self._calculate_basic_metrics(y_true, y_pred)
        
        # Advanced metrics
        advanced_metrics = self._calculate_advanced_metrics(y_true, y_pred, y_scores)
        
        # Custom anomaly detection metrics
        custom_metrics = self._calculate_custom_metrics(y_true, y_pred, y_scores)
        
        # Performance metrics
        performance_metrics = self._calculate_performance_metrics(y_true, y_pred)
        
        # Combine all metrics
        results = {
            "model_name": model_name,
            "basic_metrics": basic_metrics,
            "advanced_metrics": advanced_metrics,
            "custom_metrics": custom_metrics,
            "performance_metrics": performance_metrics,
            "evaluation_time": time.time() - start_time
        }
        
        # Store in history
        self.metric_history.append(results)
        
        logger.info(f"Evaluation completed in {results['evaluation_time']:.2f}s")
        
        return results
    
    def _calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate basic classification metrics."""
        try:
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # Calculate accuracy
            accuracy = np.mean(y_true == y_pred)
            
            # Calculate specificity (True Negative Rate)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # Calculate balanced accuracy
            balanced_accuracy = (recall + specificity) / 2
            
            return {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "accuracy": accuracy,
                "specificity": specificity,
                "balanced_accuracy": balanced_accuracy
            }
        except Exception as e:
            logger.error(f"Error calculating basic metrics: {e}")
            return {}
    
    def _calculate_advanced_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  y_scores: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate advanced metrics including ROC and PR curves."""
        advanced_metrics = {}
        
        if y_scores is not None:
            try:
                # ROC AUC
                if len(np.unique(y_true)) > 1:
                    roc_auc = roc_auc_score(y_true, y_scores)
                    advanced_metrics["roc_auc"] = roc_auc
                
                # Average Precision
                avg_precision = average_precision_score(y_true, y_scores)
                advanced_metrics["average_precision"] = avg_precision
                
                # Calculate optimal threshold
                fpr, tpr, thresholds = roc_curve(y_true, y_scores)
                optimal_idx = np.argmax(tpr - fpr)
                optimal_threshold = thresholds[optimal_idx]
                advanced_metrics["optimal_threshold"] = optimal_threshold
                
                # Youden's J statistic
                youden_j = tpr[optimal_idx] - fpr[optimal_idx]
                advanced_metrics["youden_j"] = youden_j
                
            except Exception as e:
                logger.error(f"Error calculating advanced metrics: {e}")
        
        return advanced_metrics
    
    def _calculate_custom_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                y_scores: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate custom metrics specific to anomaly detection."""
        custom_metrics = {}
        
        try:
            # Anomaly detection rate
            anomaly_rate = np.mean(y_true)
            custom_metrics["anomaly_rate"] = anomaly_rate
            
            # Detection rate (recall for anomalies)
            detection_rate = recall_score(y_true, y_pred, zero_division=0)
            custom_metrics["detection_rate"] = detection_rate
            
            # False alarm rate (1 - specificity)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
            custom_metrics["false_alarm_rate"] = false_alarm_rate
            
            # Precision for anomalies
            anomaly_precision = precision_score(y_true, y_pred, zero_division=0)
            custom_metrics["anomaly_precision"] = anomaly_precision
            
            # Matthews Correlation Coefficient
            mcc_numerator = (tp * tn) - (fp * fn)
            mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            mcc = mcc_numerator / mcc_denominator if mcc_denominator > 0 else 0
            custom_metrics["matthews_correlation"] = mcc
            
            # Cohen's Kappa
            po = (tp + tn) / (tp + tn + fp + fn)  # observed agreement
            pe = ((tp + fp) * (tp + fn) + (tn + fp) * (tn + fn)) / ((tp + tn + fp + fn) ** 2)  # expected agreement
            kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0
            custom_metrics["cohens_kappa"] = kappa
            
        except Exception as e:
            logger.error(f"Error calculating custom metrics: {e}")
        
        return custom_metrics
    
    def _calculate_performance_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate performance-related metrics."""
        try:
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            # Performance metrics
            performance = {
                "true_positives": int(tp),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "total_predictions": len(y_true),
                "positive_predictions": int(np.sum(y_pred)),
                "negative_predictions": int(len(y_pred) - np.sum(y_pred))
            }
            
            # Calculate rates
            if (tp + fn) > 0:
                performance["true_positive_rate"] = tp / (tp + fn)
            if (tn + fp) > 0:
                performance["true_negative_rate"] = tn / (tn + fp)
            if (tp + fp) > 0:
                performance["positive_predictive_value"] = tp / (tp + fp)
            if (tn + fn) > 0:
                performance["negative_predictive_value"] = tn / (tn + fn)
            
            return performance
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def compare_models(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Compare multiple models and return comparison DataFrame."""
        comparison_data = []
        
        for result in results:
            model_name = result["model_name"]
            
            # Extract key metrics
            basic = result.get("basic_metrics", {})
            advanced = result.get("advanced_metrics", {})
            custom = result.get("custom_metrics", {})
            
            row = {
                "model_name": model_name,
                "precision": basic.get("precision", 0),
                "recall": basic.get("recall", 0),
                "f1_score": basic.get("f1_score", 0),
                "accuracy": basic.get("accuracy", 0),
                "roc_auc": advanced.get("roc_auc", 0),
                "average_precision": advanced.get("average_precision", 0),
                "detection_rate": custom.get("detection_rate", 0),
                "false_alarm_rate": custom.get("false_alarm_rate", 0),
                "matthews_correlation": custom.get("matthews_correlation", 0),
                "evaluation_time": result.get("evaluation_time", 0)
            }
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def get_best_model(self, results: List[Dict[str, Any]], 
                      metric: str = "f1_score") -> Dict[str, Any]:
        """Get the best model based on a specific metric."""
        if not results:
            return {}
        
        # Find the model with the highest score for the specified metric
        best_score = -1
        best_model = None
        
        for result in results:
            basic_metrics = result.get("basic_metrics", {})
            advanced_metrics = result.get("advanced_metrics", {})
            custom_metrics = result.get("custom_metrics", {})
            
            # Check in different metric categories
            score = (basic_metrics.get(metric, 0) or 
                    advanced_metrics.get(metric, 0) or 
                    custom_metrics.get(metric, 0))
            
            if score > best_score:
                best_score = score
                best_model = result
        
        return best_model
    
    def generate_report(self, results: List[Dict[str, Any]]) -> str:
        """Generate a comprehensive evaluation report."""
        if not results:
            return "No results to report."
        
        report = []
        report.append("=" * 60)
        report.append("ANOMALY DETECTION MODEL EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Model comparison table
        comparison_df = self.compare_models(results)
        report.append("MODEL COMPARISON:")
        report.append("-" * 40)
        report.append(comparison_df.to_string(index=False))
        report.append("")
        
        # Best model
        best_model = self.get_best_model(results)
        if best_model:
            report.append(f"BEST MODEL: {best_model['model_name']}")
            report.append(f"F1 Score: {best_model['basic_metrics'].get('f1_score', 0):.4f}")
            report.append(f"Precision: {best_model['basic_metrics'].get('precision', 0):.4f}")
            report.append(f"Recall: {best_model['basic_metrics'].get('recall', 0):.4f}")
            report.append("")
        
        # Detailed results for each model
        for result in results:
            report.append(f"DETAILED RESULTS - {result['model_name']}:")
            report.append("-" * 40)
            
            basic = result.get("basic_metrics", {})
            report.append(f"Precision: {basic.get('precision', 0):.4f}")
            report.append(f"Recall: {basic.get('recall', 0):.4f}")
            report.append(f"F1 Score: {basic.get('f1_score', 0):.4f}")
            report.append(f"Accuracy: {basic.get('accuracy', 0):.4f}")
            report.append(f"Specificity: {basic.get('specificity', 0):.4f}")
            
            advanced = result.get("advanced_metrics", {})
            if advanced:
                report.append(f"ROC AUC: {advanced.get('roc_auc', 0):.4f}")
                report.append(f"Average Precision: {advanced.get('average_precision', 0):.4f}")
            
            custom = result.get("custom_metrics", {})
            if custom:
                report.append(f"Detection Rate: {custom.get('detection_rate', 0):.4f}")
                report.append(f"False Alarm Rate: {custom.get('false_alarm_rate', 0):.4f}")
                report.append(f"Matthews Correlation: {custom.get('matthews_correlation', 0):.4f}")
            
            report.append(f"Evaluation Time: {result.get('evaluation_time', 0):.2f}s")
            report.append("")
        
        return "\n".join(report)
    
    def save_results(self, results: List[Dict[str, Any]], filepath: str):
        """Save evaluation results to file."""
        import json
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Convert results
        serializable_results = []
        for result in results:
            serializable_result = {}
            for key, value in result.items():
                if key == "model_name":
                    serializable_result[key] = value
                else:
                    serializable_result[key] = convert_numpy(value)
            serializable_results.append(serializable_result)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
    
    def load_results(self, filepath: str) -> List[Dict[str, Any]]:
        """Load evaluation results from file."""
        import json
        
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        logger.info(f"Results loaded from {filepath}")
        return results 