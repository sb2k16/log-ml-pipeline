"""
Isolation Forest Anomaly Detector

Implements Isolation Forest algorithm for unsupervised anomaly detection.
Isolation Forest is efficient for high-dimensional data and works well
with log data by isolating anomalies in fewer steps.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import joblib
import os

logger = logging.getLogger(__name__)


class IsolationForestDetector:
    """Isolation Forest based anomaly detector."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("isolation_forest", {})
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False
        
        # Model parameters
        self.contamination = self.config.get("contamination", 0.1)
        self.n_estimators = self.config.get("n_estimators", 100)
        self.max_samples = self.config.get("max_samples", "auto")
        self.random_state = self.config.get("random_state", 42)
        
        # Performance tracking
        self.training_time = 0
        self.inference_times = []
        
    def fit(self, X: pd.DataFrame) -> 'IsolationForestDetector':
        """Fit the Isolation Forest model."""
        import time
        start_time = time.time()
        
        logger.info("Training Isolation Forest model...")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Prepare features (select numerical columns)
        X_numerical = self._prepare_features(X)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_numerical)
        
        # Initialize and train model
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=-1  # Use all available cores
        )
        
        self.model.fit(X_scaled)
        
        self.training_time = time.time() - start_time
        self.is_fitted = True
        
        logger.info(f"Isolation Forest training completed in {self.training_time:.2f}s")
        logger.info(f"Model parameters: contamination={self.contamination}, "
                   f"n_estimators={self.n_estimators}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict anomalies in the data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        import time
        start_time = time.time()
        
        # Prepare features
        X_numerical = self._prepare_features(X)
        
        # Scale features
        X_scaled = self.scaler.transform(X_numerical)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        # Convert to binary (1 for anomaly, 0 for normal)
        # Isolation Forest returns -1 for anomalies, 1 for normal
        anomaly_scores = (predictions == -1).astype(int)
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        logger.debug(f"Prediction completed in {inference_time:.4f}s")
        
        return anomaly_scores
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict anomaly probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Prepare features
        X_numerical = self._prepare_features(X)
        X_scaled = self.scaler.transform(X_numerical)
        
        # Get decision function (lower values indicate more anomalous)
        decision_scores = self.model.decision_function(X_scaled)
        
        # Convert to probability-like scores (0 to 1, where 1 is more anomalous)
        # Normalize decision scores to [0, 1] range
        min_score = np.min(decision_scores)
        max_score = np.max(decision_scores)
        
        if max_score > min_score:
            proba_scores = (decision_scores - min_score) / (max_score - min_score)
        else:
            proba_scores = np.zeros_like(decision_scores)
        
        return proba_scores
    
    def get_anomaly_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Get raw anomaly scores (decision function values)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting scores")
        
        X_numerical = self._prepare_features(X)
        X_scaled = self.scaler.transform(X_numerical)
        
        return self.model.decision_function(X_scaled)
    
    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for the model."""
        # Select numerical columns
        numerical_columns = X.select_dtypes(include=[np.number]).columns
        
        if len(numerical_columns) == 0:
            raise ValueError("No numerical features found in the dataset")
        
        # Fill missing values
        X_numerical = X[numerical_columns].fillna(0)
        
        # Remove infinite values
        X_numerical = X_numerical.replace([np.inf, -np.inf], 0)
        
        return X_numerical
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Optimize hyperparameters using grid search."""
        logger.info("Optimizing Isolation Forest hyperparameters...")
        
        # Prepare features
        X_numerical = self._prepare_features(X)
        X_scaled = self.scaler.fit_transform(X_numerical)
        
        # Define parameter grid
        param_grid = {
            'contamination': [0.05, 0.1, 0.15, 0.2],
            'n_estimators': [50, 100, 200],
            'max_samples': ['auto', 100, 200]
        }
        
        # Initialize base model
        base_model = IsolationForest(random_state=self.random_state)
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_scaled)
        
        # Update model parameters
        best_params = grid_search.best_params_
        self.contamination = best_params['contamination']
        self.n_estimators = best_params['n_estimators']
        self.max_samples = best_params['max_samples']
        
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best score: {grid_search.best_score_:.4f}")
        
        return best_params
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        # Isolation Forest doesn't provide direct feature importance
        # We can estimate it by measuring the impact of each feature
        # on the anomaly scores
        
        importance_scores = {}
        
        if hasattr(self.model, 'estimators_'):
            # Calculate feature importance based on tree structure
            n_features = len(self.feature_names)
            feature_importance = np.zeros(n_features)
            
            for estimator in self.model.estimators_:
                # Count feature usage in the tree
                for node in estimator.tree_.children_left:
                    if node != -1:  # Not a leaf node
                        feature_idx = estimator.tree_.feature[node]
                        if feature_idx >= 0:
                            feature_importance[feature_idx] += 1
            
            # Normalize importance scores
            if np.sum(feature_importance) > 0:
                feature_importance = feature_importance / np.sum(feature_importance)
            
            # Map to feature names
            for i, feature_name in enumerate(self.feature_names):
                importance_scores[feature_name] = float(feature_importance[i])
        
        return importance_scores
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics."""
        info = {
            "model_type": "Isolation Forest",
            "is_fitted": self.is_fitted,
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "parameters": {
                "contamination": self.contamination,
                "n_estimators": self.n_estimators,
                "max_samples": self.max_samples,
                "random_state": self.random_state
            },
            "performance": {
                "training_time": self.training_time,
                "avg_inference_time": np.mean(self.inference_times) if self.inference_times else 0,
                "n_predictions": len(self.inference_times)
            }
        }
        
        return info
    
    def save_model(self, filepath: str):
        """Save the trained model to disk."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "config": self.config,
            "is_fitted": self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.feature_names = model_data["feature_names"]
        self.config = model_data["config"]
        self.is_fitted = model_data["is_fitted"]
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_threshold_scores(self, X: pd.DataFrame, contamination: float = 0.1) -> float:
        """Get threshold score for a given contamination level."""
        scores = self.get_anomaly_scores(X)
        threshold = np.percentile(scores, (1 - contamination) * 100)
        return threshold
    
    def detect_anomalies_with_threshold(self, X: pd.DataFrame, threshold: float) -> np.ndarray:
        """Detect anomalies using a custom threshold."""
        scores = self.get_anomaly_scores(X)
        return (scores < threshold).astype(int) 