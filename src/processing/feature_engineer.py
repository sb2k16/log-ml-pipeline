"""
Feature Engineering Module

Extracts and engineers features from log data for anomaly detection.
Includes time-based, text-based, and statistical features.
"""

import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import hashlib

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Engineers features from log data for anomaly detection."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.feature_config = config.get("feature_engineering", {})
        
        # Initialize encoders and vectorizers
        self.label_encoders = {}
        self.tfidf_vectorizer = None
        self.scaler = StandardScaler()
        
        # Feature cache for performance
        self.feature_cache = {}
        
        # Statistical aggregations
        self.rolling_stats = {}
        self.global_stats = {}
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer all features for the dataset."""
        if df.empty:
            return df
        
        logger.info(f"Engineering features for {len(df)} log entries")
        
        # Create a copy to avoid modifying original
        df_features = df.copy()
        
        # Time-based features
        df_features = self._add_time_features(df_features)
        
        # Text-based features
        df_features = self._add_text_features(df_features)
        
        # Statistical features
        df_features = self._add_statistical_features(df_features)
        
        # Categorical features
        df_features = self._add_categorical_features(df_features)
        
        # Network features
        df_features = self._add_network_features(df_features)
        
        # Performance features
        df_features = self._add_performance_features(df_features)
        
        # Behavioral features
        df_features = self._add_behavioral_features(df_features)
        
        logger.info(f"Feature engineering complete. Final shape: {df_features.shape}")
        
        return df_features
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        if 'timestamp' not in df.columns:
            logger.warning("No timestamp column found, skipping time features")
            return df
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Basic time features
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['year'] = df['timestamp'].dt.year
        df['is_weekend'] = df['timestamp'].dt.dayofweek.isin([5, 6]).astype(int)
        df['is_business_hour'] = ((df['timestamp'].dt.hour >= 9) & 
                                 (df['timestamp'].dt.hour <= 17)).astype(int)
        
        # Time since epoch (for numerical representation)
        df['timestamp_epoch'] = df['timestamp'].astype(np.int64) // 10**9
        
        # Cyclical time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Time intervals
        df['time_since_midnight'] = (df['timestamp'] - 
                                   df['timestamp'].dt.normalize()).dt.total_seconds()
        
        return df
    
    def _add_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add text-based features from log messages."""
        if 'message' not in df.columns:
            logger.warning("No message column found, skipping text features")
            return df
        
        # Basic text features
        df['message_length'] = df['message'].str.len()
        df['word_count'] = df['message'].str.split().str.len()
        df['char_count'] = df['message'].str.len()
        
        # Special character counts
        df['special_char_count'] = df['message'].str.count(r'[^a-zA-Z0-9\s]')
        df['digit_count'] = df['message'].str.count(r'\d')
        df['uppercase_count'] = df['message'].str.count(r'[A-Z]')
        df['lowercase_count'] = df['message'].str.count(r'[a-z]')
        
        # URL and IP detection
        url_pattern = r'https?://[^\s]+'
        ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        
        df['url_count'] = df['message'].str.count(url_pattern)
        df['ip_count'] = df['message'].str.count(ip_pattern)
        
        # Error pattern detection
        error_patterns = ['error', 'exception', 'failed', 'timeout', 'crash']
        for pattern in error_patterns:
            df[f'contains_{pattern}'] = df['message'].str.contains(
                pattern, case=False, regex=False
            ).astype(int)
        
        # Log level encoding
        if 'level' in df.columns:
            level_mapping = {
                'DEBUG': 0, 'INFO': 1, 'WARNING': 2, 
                'ERROR': 3, 'CRITICAL': 4
            }
            df['level_encoded'] = df['level'].map(level_mapping).fillna(1)
        
        # Message hash for uniqueness
        df['message_hash'] = df['message'].apply(
            lambda x: int(hashlib.md5(x.encode()).hexdigest()[:8], 16)
        )
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features based on rolling windows."""
        if df.empty:
            return df
        
        # Sort by timestamp for rolling calculations
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Rolling statistics for different time windows
        windows = [60, 300, 3600]  # 1min, 5min, 1hour in seconds
        
        for window in windows:
            window_name = f"{window}s"
            
            # Rolling counts
            df[f'rolling_count_{window_name}'] = df.groupby(
                pd.Grouper(key='timestamp', freq=f'{window}S')
            ).size().reindex(df.index, method='ffill').fillna(0)
            
            # Rolling error rates
            if 'level' in df.columns:
                error_mask = df['level'].isin(['ERROR', 'CRITICAL'])
                df[f'rolling_error_rate_{window_name}'] = error_mask.rolling(
                    window=window, min_periods=1
                ).mean()
            
            # Rolling message length statistics
            if 'message_length' in df.columns:
                df[f'rolling_mean_length_{window_name}'] = df['message_length'].rolling(
                    window=window, min_periods=1
                ).mean()
                df[f'rolling_std_length_{window_name}'] = df['message_length'].rolling(
                    window=window, min_periods=1
                ).std()
        
        return df
    
    def _add_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add categorical features with encoding."""
        categorical_columns = ['source', 'service', 'method', 'ip_address']
        
        for col in categorical_columns:
            if col in df.columns and df[col].notna().any():
                # Create label encoder if not exists
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    # Fit on all unique values
                    unique_values = df[col].dropna().unique()
                    self.label_encoders[col].fit(unique_values)
                
                # Encode values
                df[f'{col}_encoded'] = self.label_encoders[col].transform(
                    df[col].fillna('unknown')
                )
                
                # One-hot encoding for top categories
                top_categories = df[col].value_counts().head(10).index
                for category in top_categories:
                    df[f'{col}_{category}'] = (df[col] == category).astype(int)
        
        return df
    
    def _add_network_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add network-related features."""
        if 'ip_address' not in df.columns:
            return df
        
        # IP address features
        df['ip_is_private'] = df['ip_address'].apply(self._is_private_ip)
        df['ip_is_localhost'] = (df['ip_address'] == '127.0.0.1').astype(int)
        
        # Extract IP octets
        ip_octets = df['ip_address'].str.extract(r'(\d+)\.(\d+)\.(\d+)\.(\d+)')
        if not ip_octets.empty:
            for i in range(4):
                df[f'ip_octet_{i+1}'] = pd.to_numeric(ip_octets[i], errors='coerce')
        
        return df
    
    def _add_performance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add performance-related features."""
        # Duration features
        if 'duration' in df.columns:
            df['duration_log'] = np.log1p(df['duration'].fillna(0))
            df['duration_squared'] = df['duration'].fillna(0) ** 2
        
        # Status code features
        if 'status_code' in df.columns:
            df['is_error_status'] = (df['status_code'] >= 400).astype(int)
            df['is_success_status'] = (df['status_code'] >= 200) & (df['status_code'] < 300)
            df['is_redirect_status'] = (df['status_code'] >= 300) & (df['status_code'] < 400)
        
        # Request/response size features
        for size_col in ['request_size', 'response_size']:
            if size_col in df.columns:
                df[f'{size_col}_log'] = np.log1p(df[size_col].fillna(0))
                df[f'{size_col}_missing'] = df[size_col].isna().astype(int)
        
        return df
    
    def _add_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add behavioral features based on user/session patterns."""
        # Session-based features
        if 'session_id' in df.columns and 'user_id' in df.columns:
            # Session duration
            session_stats = df.groupby('session_id').agg({
                'timestamp': ['min', 'max', 'count']
            }).reset_index()
            session_stats.columns = ['session_id', 'session_start', 'session_end', 'session_count']
            session_stats['session_duration'] = (
                session_stats['session_end'] - session_stats['session_start']
            ).dt.total_seconds()
            
            # Merge back to main dataframe
            df = df.merge(session_stats, on='session_id', how='left')
        
        # User-based features
        if 'user_id' in df.columns:
            user_stats = df.groupby('user_id').agg({
                'timestamp': 'count',
                'level': lambda x: (x == 'ERROR').sum()
            }).reset_index()
            user_stats.columns = ['user_id', 'user_total_requests', 'user_error_count']
            user_stats['user_error_rate'] = user_stats['user_error_count'] / user_stats['user_total_requests']
            
            df = df.merge(user_stats, on='user_id', how='left')
        
        return df
    
    def _is_private_ip(self, ip: str) -> bool:
        """Check if IP address is private."""
        if pd.isna(ip):
            return False
        
        try:
            octets = [int(x) for x in ip.split('.')]
            return (octets[0] == 10 or 
                   (octets[0] == 172 and 16 <= octets[1] <= 31) or
                   (octets[0] == 192 and octets[1] == 168))
        except:
            return False
    
    def get_feature_names(self) -> List[str]:
        """Get list of engineered feature names."""
        # This would be implemented based on the actual features created
        # For now, return common feature names
        return [
            'hour_of_day', 'day_of_week', 'is_weekend', 'is_business_hour',
            'message_length', 'word_count', 'special_char_count',
            'level_encoded', 'rolling_count_60s', 'rolling_error_rate_60s'
        ]
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit the feature engineer and transform the data."""
        # Fit scalers and encoders
        self._fit_encoders(df)
        
        # Transform the data
        return self.engineer_features(df)
    
    def _fit_encoders(self, df: pd.DataFrame):
        """Fit label encoders on the training data."""
        categorical_columns = ['source', 'service', 'method', 'ip_address']
        
        for col in categorical_columns:
            if col in df.columns and df[col].notna().any():
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    unique_values = df[col].dropna().unique()
                    self.label_encoders[col].fit(unique_values)
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted encoders."""
        return self.engineer_features(df) 