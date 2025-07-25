"""
Rule-Based Anomaly Detector

Implements various rule-based methods for anomaly detection including:
- Statistical thresholds (Z-score, IQR)
- Pattern matching
- Frequency-based rules
- Time-based rules
"""

import logging
import numpy as np
import pandas as pd
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class RuleBasedDetector:
    """Rule-based anomaly detector with multiple detection methods."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("rule_based", {})
        self.is_fitted = False
        
        # Statistical thresholds
        self.statistical_config = self.config.get("statistical", {})
        self.z_score_threshold = self.statistical_config.get("z_score_threshold", 3.0)
        self.iqr_multiplier = self.statistical_config.get("iqr_multiplier", 1.5)
        self.rolling_window = self.statistical_config.get("rolling_window", 3600)
        
        # Pattern matching
        self.pattern_config = self.config.get("pattern_matching", {})
        self.error_patterns = self.pattern_config.get("error_patterns", [])
        self.warning_patterns = self.pattern_config.get("warning_patterns", [])
        
        # Frequency-based rules
        self.frequency_config = self.config.get("frequency_based", {})
        self.max_errors_per_minute = self.frequency_config.get("max_errors_per_minute", 10)
        self.max_warnings_per_minute = self.frequency_config.get("max_warnings_per_minute", 50)
        self.max_requests_per_second = self.frequency_config.get("max_requests_per_second", 1000)
        
        # Statistical baselines
        self.baselines = {}
        self.thresholds = {}
        
        # Performance tracking
        self.detection_counts = defaultdict(int)
        self.rule_performance = {}
    
    def fit(self, df: pd.DataFrame) -> 'RuleBasedDetector':
        """Fit the rule-based detector by establishing baselines."""
        logger.info("Fitting rule-based detector...")
        
        # Calculate statistical baselines
        self._calculate_statistical_baselines(df)
        
        # Calculate frequency baselines
        self._calculate_frequency_baselines(df)
        
        # Calculate pattern baselines
        self._calculate_pattern_baselines(df)
        
        self.is_fitted = True
        logger.info("Rule-based detector fitted successfully")
        
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict anomalies using all rule-based methods."""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before making predictions")
        
        logger.info("Running rule-based anomaly detection...")
        
        # Initialize results
        n_samples = len(df)
        anomaly_scores = np.zeros(n_samples)
        
        # Statistical anomaly detection
        statistical_anomalies = self._detect_statistical_anomalies(df)
        anomaly_scores += statistical_anomalies
        
        # Pattern-based anomaly detection
        pattern_anomalies = self._detect_pattern_anomalies(df)
        anomaly_scores += pattern_anomalies
        
        # Frequency-based anomaly detection
        frequency_anomalies = self._detect_frequency_anomalies(df)
        anomaly_scores += frequency_anomalies
        
        # Time-based anomaly detection
        time_anomalies = self._detect_time_anomalies(df)
        anomaly_scores += time_anomalies
        
        # Convert to binary (any rule triggered = anomaly)
        binary_anomalies = (anomaly_scores > 0).astype(int)
        
        logger.info(f"Detected {np.sum(binary_anomalies)} anomalies out of {n_samples} samples")
        
        return binary_anomalies
    
    def _calculate_statistical_baselines(self, df: pd.DataFrame):
        """Calculate statistical baselines for numerical features."""
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_columns:
            if col in ['timestamp', 'timestamp_epoch']:
                continue
                
            values = df[col].dropna()
            if len(values) == 0:
                continue
            
            # Calculate basic statistics
            mean_val = values.mean()
            std_val = values.std()
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            
            self.baselines[col] = {
                'mean': mean_val,
                'std': std_val,
                'q1': q1,
                'q3': q3,
                'iqr': iqr,
                'min': values.min(),
                'max': values.max()
            }
            
            # Calculate thresholds
            self.thresholds[col] = {
                'z_score_lower': mean_val - self.z_score_threshold * std_val,
                'z_score_upper': mean_val + self.z_score_threshold * std_val,
                'iqr_lower': q1 - self.iqr_multiplier * iqr,
                'iqr_upper': q3 + self.iqr_multiplier * iqr
            }
    
    def _calculate_frequency_baselines(self, df: pd.DataFrame):
        """Calculate frequency baselines for categorical features."""
        if 'timestamp' not in df.columns:
            return
        
        # Calculate error rates
        if 'level' in df.columns:
            error_counts = df[df['level'].isin(['ERROR', 'CRITICAL'])].groupby(
                pd.Grouper(key='timestamp', freq='1min')
            ).size()
            
            self.baselines['error_rate'] = {
                'mean': error_counts.mean(),
                'std': error_counts.std(),
                'max': error_counts.max()
            }
        
        # Calculate request rates
        if 'method' in df.columns:
            request_counts = df.groupby(
                pd.Grouper(key='timestamp', freq='1min')
            ).size()
            
            self.baselines['request_rate'] = {
                'mean': request_counts.mean(),
                'std': request_counts.std(),
                'max': request_counts.max()
            }
    
    def _calculate_pattern_baselines(self, df: pd.DataFrame):
        """Calculate pattern baselines."""
        if 'message' not in df.columns:
            return
        
        # Calculate pattern frequencies
        pattern_counts = {}
        
        for pattern in self.error_patterns + self.warning_patterns:
            count = df['message'].str.contains(pattern, case=False, regex=False).sum()
            pattern_counts[pattern] = count
        
        self.baselines['patterns'] = pattern_counts
    
    def _detect_statistical_anomalies(self, df: pd.DataFrame) -> np.ndarray:
        """Detect anomalies using statistical thresholds."""
        anomalies = np.zeros(len(df))
        
        for col, thresholds in self.thresholds.items():
            if col not in df.columns:
                continue
            
            values = df[col].fillna(0)
            
            # Z-score based detection
            if self.baselines[col]['std'] > 0:
                z_scores = np.abs((values - self.baselines[col]['mean']) / self.baselines[col]['std'])
                z_score_anomalies = (z_scores > self.z_score_threshold)
                anomalies += z_score_anomalies.astype(int)
            
            # IQR based detection
            iqr_anomalies = ((values < thresholds['iqr_lower']) | 
                            (values > thresholds['iqr_upper']))
            anomalies += iqr_anomalies.astype(int)
        
        return anomalies
    
    def _detect_pattern_anomalies(self, df: pd.DataFrame) -> np.ndarray:
        """Detect anomalies using pattern matching."""
        anomalies = np.zeros(len(df))
        
        if 'message' not in df.columns:
            return anomalies
        
        # Check for error patterns
        for pattern in self.error_patterns:
            pattern_matches = df['message'].str.contains(pattern, case=False, regex=False)
            anomalies += pattern_matches.astype(int)
        
        # Check for warning patterns
        for pattern in self.warning_patterns:
            pattern_matches = df['message'].str.contains(pattern, case=False, regex=False)
            anomalies += pattern_matches.astype(int)
        
        return anomalies
    
    def _detect_frequency_anomalies(self, df: pd.DataFrame) -> np.ndarray:
        """Detect anomalies based on frequency thresholds."""
        anomalies = np.zeros(len(df))
        
        if 'timestamp' not in df.columns:
            return anomalies
        
        # Error frequency detection
        if 'level' in df.columns:
            error_counts = df[df['level'].isin(['ERROR', 'CRITICAL'])].groupby(
                pd.Grouper(key='timestamp', freq='1min')
            ).size()
            
            # Mark periods with high error rates
            high_error_periods = error_counts[error_counts > self.max_errors_per_minute]
            
            for period in high_error_periods.index:
                period_mask = (df['timestamp'] >= period) & (df['timestamp'] < period + timedelta(minutes=1))
                anomalies += period_mask.astype(int)
        
        # Request frequency detection
        request_counts = df.groupby(
            pd.Grouper(key='timestamp', freq='1min')
        ).size()
        
        high_request_periods = request_counts[request_counts > self.max_requests_per_second * 60]
        
        for period in high_request_periods.index:
            period_mask = (df['timestamp'] >= period) & (df['timestamp'] < period + timedelta(minutes=1))
            anomalies += period_mask.astype(int)
        
        return anomalies
    
    def _detect_time_anomalies(self, df: pd.DataFrame) -> np.ndarray:
        """Detect anomalies based on time-based rules."""
        anomalies = np.zeros(len(df))
        
        if 'timestamp' not in df.columns:
            return anomalies
        
        # Business hours anomaly detection
        if 'hour_of_day' in df.columns:
            # Detect unusual activity outside business hours
            non_business_hours = ~((df['hour_of_day'] >= 9) & (df['hour_of_day'] <= 17))
            
            # Only flag if there's significant activity outside business hours
            if non_business_hours.sum() > len(df) * 0.1:  # More than 10% of activity
                anomalies += non_business_hours.astype(int)
        
        # Weekend anomaly detection
        if 'day_of_week' in df.columns:
            weekend_activity = df['day_of_week'].isin([5, 6])  # Saturday, Sunday
            
            # Only flag if there's significant weekend activity
            if weekend_activity.sum() > len(df) * 0.05:  # More than 5% of activity
                anomalies += weekend_activity.astype(int)
        
        return anomalies
    
    def get_rule_breakdown(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get detailed breakdown of which rules were triggered."""
        breakdown = {
            'statistical': self._detect_statistical_anomalies(df),
            'pattern': self._detect_pattern_anomalies(df),
            'frequency': self._detect_frequency_anomalies(df),
            'time': self._detect_time_anomalies(df)
        }
        
        # Count anomalies by rule type
        rule_counts = {}
        for rule_type, anomalies in breakdown.items():
            rule_counts[rule_type] = int(np.sum(anomalies))
        
        return {
            'breakdown': breakdown,
            'counts': rule_counts,
            'total_anomalies': int(np.sum(any(breakdown.values())))
        }
    
    def get_baseline_info(self) -> Dict[str, Any]:
        """Get information about calculated baselines."""
        return {
            'statistical_baselines': self.baselines,
            'thresholds': self.thresholds,
            'config': {
                'z_score_threshold': self.z_score_threshold,
                'iqr_multiplier': self.iqr_multiplier,
                'error_patterns': self.error_patterns,
                'warning_patterns': self.warning_patterns,
                'frequency_limits': {
                    'max_errors_per_minute': self.max_errors_per_minute,
                    'max_warnings_per_minute': self.max_warnings_per_minute,
                    'max_requests_per_second': self.max_requests_per_second
                }
            }
        }
    
    def update_baselines(self, new_data: pd.DataFrame):
        """Update baselines with new data."""
        logger.info("Updating rule-based detector baselines...")
        
        # Combine with existing baselines if available
        if self.baselines:
            # This would require more sophisticated baseline updating logic
            # For now, we'll refit with the new data
            self.fit(new_data)
        else:
            self.fit(new_data)
    
    def add_custom_rule(self, rule_name: str, rule_function: callable):
        """Add a custom rule function."""
        if not hasattr(self, 'custom_rules'):
            self.custom_rules = {}
        
        self.custom_rules[rule_name] = rule_function
        logger.info(f"Added custom rule: {rule_name}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics."""
        return {
            "model_type": "Rule-Based Detector",
            "is_fitted": self.is_fitted,
            "n_baselines": len(self.baselines),
            "n_thresholds": len(self.thresholds),
            "config": self.config,
            "baseline_features": list(self.baselines.keys())
        } 