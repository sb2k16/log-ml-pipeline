"""
Advanced ML Models for Anomaly Detection
Implements various unsupervised learning models for anomaly detection.
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)

class IsolationForestDetector:
    """Isolation Forest for anomaly detection."""
    
    def __init__(self, contamination=0.1, random_state=42, n_estimators=100):
        self.contamination = contamination
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.model = None
        self.is_fitted = False
        
    def fit(self, features: np.ndarray) -> 'IsolationForestDetector':
        """Fit the Isolation Forest model."""
        try:
            from sklearn.ensemble import IsolationForest
            
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state,
                n_estimators=self.n_estimators
            )
            
            self.model.fit(features)
            self.is_fitted = True
            logger.info(f"✓ Isolation Forest fitted with {len(features)} samples")
            
        except ImportError:
            logger.error("scikit-learn not available. Install with: pip install scikit-learn")
            raise
        
        return self
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict anomalies (-1 for anomalies, 1 for normal)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict(features)
    
    def score_samples(self, features: np.ndarray) -> np.ndarray:
        """Get anomaly scores (lower values = more anomalous)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        
        return self.model.score_samples(features)
    
    def get_anomalies(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get anomaly predictions and scores."""
        predictions = self.predict(features)
        scores = self.score_samples(features)
        
        # Convert to binary (1 for anomaly, 0 for normal)
        anomalies = (predictions == -1).astype(int)
        
        return anomalies, scores

class OneClassSVMDetector:
    """One-Class SVM for anomaly detection."""
    
    def __init__(self, nu=0.1, kernel='rbf', gamma='scale'):
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.model = None
        self.is_fitted = False
        
    def fit(self, features: np.ndarray) -> 'OneClassSVMDetector':
        """Fit the One-Class SVM model."""
        try:
            from sklearn.svm import OneClassSVM
            
            self.model = OneClassSVM(
                nu=self.nu,
                kernel=self.kernel,
                gamma=self.gamma
            )
            
            self.model.fit(features)
            self.is_fitted = True
            logger.info(f"✓ One-Class SVM fitted with {len(features)} samples")
            
        except ImportError:
            logger.error("scikit-learn not available. Install with: pip install scikit-learn")
            raise
        
        return self
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict anomalies (-1 for anomalies, 1 for normal)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict(features)
    
    def score_samples(self, features: np.ndarray) -> np.ndarray:
        """Get anomaly scores (lower values = more anomalous)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        
        return self.model.score_samples(features)
    
    def get_anomalies(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get anomaly predictions and scores."""
        predictions = self.predict(features)
        scores = self.score_samples(features)
        
        # Convert to binary (1 for anomaly, 0 for normal)
        anomalies = (predictions == -1).astype(int)
        
        return anomalies, scores

class LocalOutlierFactorDetector:
    """Local Outlier Factor for anomaly detection."""
    
    def __init__(self, n_neighbors=20, contamination=0.1):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.model = None
        self.is_fitted = False
        
    def fit(self, features: np.ndarray) -> 'LocalOutlierFactorDetector':
        """Fit the LOF model."""
        try:
            from sklearn.neighbors import LocalOutlierFactor
            
            self.model = LocalOutlierFactor(
                n_neighbors=self.n_neighbors,
                contamination=self.contamination
            )
            
            # LOF doesn't need separate fit for prediction
            self.is_fitted = True
            logger.info(f"✓ Local Outlier Factor configured for {len(features)} samples")
            
        except ImportError:
            logger.error("scikit-learn not available. Install with: pip install scikit-learn")
            raise
        
        return self
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict anomalies (-1 for anomalies, 1 for normal)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.fit_predict(features)
    
    def score_samples(self, features: np.ndarray) -> np.ndarray:
        """Get anomaly scores (lower values = more anomalous)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        
        return self.model.score_samples(features)
    
    def get_anomalies(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get anomaly predictions and scores."""
        predictions = self.predict(features)
        scores = self.score_samples(features)
        
        # Convert to binary (1 for anomaly, 0 for normal)
        anomalies = (predictions == -1).astype(int)
        
        return anomalies, scores

class EnsembleDetector:
    """Ensemble of multiple anomaly detection models."""
    
    def __init__(self, detectors: List, weights: Optional[List[float]] = None):
        self.detectors = detectors
        self.weights = weights or [1/len(detectors)] * len(detectors)
        self.is_fitted = False
        
        if len(self.weights) != len(self.detectors):
            raise ValueError("Number of weights must match number of detectors")
    
    def fit(self, features: np.ndarray) -> 'EnsembleDetector':
        """Fit all detectors in the ensemble."""
        for detector in self.detectors:
            detector.fit(features)
        
        self.is_fitted = True
        logger.info(f"✓ Ensemble fitted with {len(self.detectors)} detectors")
        return self
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Get ensemble predictions using weighted voting."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        predictions = []
        for detector in self.detectors:
            pred = detector.predict(features)
            # Convert to binary (1 for anomaly, 0 for normal)
            binary_pred = (pred == -1).astype(int)
            predictions.append(binary_pred)
        
        # Weighted voting
        weighted_pred = np.zeros(len(features))
        for pred, weight in zip(predictions, self.weights):
            weighted_pred += pred * weight
        
        # Threshold at 0.5 for final prediction
        ensemble_pred = (weighted_pred > 0.5).astype(int)
        
        return ensemble_pred
    
    def score_samples(self, features: np.ndarray) -> np.ndarray:
        """Get ensemble scores (average of all detector scores)."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before scoring")
        
        scores = []
        for detector in self.detectors:
            score = detector.score_samples(features)
            scores.append(score)
        
        # Average scores
        ensemble_scores = np.mean(scores, axis=0)
        return ensemble_scores
    
    def get_anomalies(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get ensemble anomaly predictions and scores."""
        predictions = self.predict(features)
        scores = self.score_samples(features)
        
        return predictions, scores

class ModelManager:
    """Manages multiple ML models for anomaly detection."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def add_model(self, name: str, model) -> None:
        """Add a model to the manager."""
        self.models[name] = model
        logger.info(f"✓ Added model: {name}")
    
    def fit_all(self, features: np.ndarray) -> None:
        """Fit all models with the given features."""
        logger.info(f"Fitting {len(self.models)} models...")
        
        for name, model in self.models.items():
            try:
                model.fit(features)
                logger.info(f"✓ {name} fitted successfully")
            except Exception as e:
                logger.error(f"✗ Failed to fit {name}: {e}")
    
    def predict_all(self, features: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Get predictions from all models."""
        results = {}
        
        for name, model in self.models.items():
            try:
                anomalies, scores = model.get_anomalies(features)
                results[name] = (anomalies, scores)
                logger.info(f"✓ {name} predictions completed")
            except Exception as e:
                logger.error(f"✗ Failed to get predictions from {name}: {e}")
        
        self.results = results
        return results
    
    def compare_models(self, features: np.ndarray, true_labels: Optional[np.ndarray] = None) -> Dict[str, Dict[str, float]]:
        """Compare performance of all models."""
        if not self.results:
            self.predict_all(features)
        
        comparison = {}
        
        for name, (anomalies, scores) in self.results.items():
            metrics = {
                'anomaly_rate': np.mean(anomalies),
                'avg_score': np.mean(scores),
                'std_score': np.std(scores)
            }
            
            if true_labels is not None:
                try:
                    from sklearn.metrics import precision_score, recall_score, f1_score
                    
                    metrics.update({
                        'precision': precision_score(true_labels, anomalies, zero_division=0),
                        'recall': recall_score(true_labels, anomalies, zero_division=0),
                        'f1_score': f1_score(true_labels, anomalies, zero_division=0)
                    })
                except ImportError:
                    logger.warning("scikit-learn not available for detailed metrics")
            
            comparison[name] = metrics
        
        return comparison
    
    def save_results(self, filepath: str) -> None:
        """Save model results to JSON file."""
        if not self.results:
            logger.warning("No results to save. Run predict_all() first.")
            return
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for name, (anomalies, scores) in self.results.items():
            serializable_results[name] = {
                'anomalies': anomalies.tolist(),
                'scores': scores.tolist()
            }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"✓ Results saved to {filepath}")
    
    def load_results(self, filepath: str) -> None:
        """Load model results from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Convert lists back to numpy arrays
        self.results = {}
        for name, result_data in data.items():
            self.results[name] = (
                np.array(result_data['anomalies']),
                np.array(result_data['scores'])
            )
        
        logger.info(f"✓ Results loaded from {filepath}")

def create_feature_matrix(features: List[Dict[str, Any]]) -> np.ndarray:
    """Convert list of feature dictionaries to numpy array."""
    # Extract numerical features
    numerical_features = []
    
    for feature in features:
        numerical_feature = []
        
        # Add all numerical values
        for key, value in feature.items():
            if isinstance(value, (int, float, bool)):
                numerical_feature.append(float(value))
            elif isinstance(value, list):
                # Handle list features (e.g., TF-IDF features)
                numerical_feature.extend([float(x) for x in value])
        
        numerical_features.append(numerical_feature)
    
    return np.array(numerical_features)

def demo_advanced_ml():
    """Demonstrate advanced ML models with sample data."""
    logger.info("=" * 60)
    logger.info("ADVANCED ML MODELS DEMO")
    logger.info("=" * 60)
    
    # Generate sample features
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Create normal data
    normal_data = np.random.normal(0, 1, (n_samples, n_features))
    
    # Add some anomalies
    anomalies = np.random.normal(5, 1, (50, n_features))
    features = np.vstack([normal_data, anomalies])
    
    # Create feature dictionaries (simulating our pipeline)
    feature_dicts = []
    for i, feature_vector in enumerate(features):
        feature_dict = {
            'feature_1': feature_vector[0],
            'feature_2': feature_vector[1],
            'feature_3': feature_vector[2],
            'feature_4': feature_vector[3],
            'feature_5': feature_vector[4],
            'feature_6': feature_vector[5],
            'feature_7': feature_vector[6],
            'feature_8': feature_vector[7],
            'feature_9': feature_vector[8],
            'feature_10': feature_vector[9],
            'is_anomaly': i >= n_samples  # Last 50 are anomalies
        }
        feature_dicts.append(feature_dict)
    
    # Convert to feature matrix
    feature_matrix = create_feature_matrix(feature_dicts)
    
    # Create model manager
    manager = ModelManager()
    
    # Add models
    try:
        manager.add_model('Isolation Forest', IsolationForestDetector(contamination=0.05))
        manager.add_model('One-Class SVM', OneClassSVMDetector(nu=0.05))
        manager.add_model('Local Outlier Factor', LocalOutlierFactorDetector(contamination=0.05))
        
        # Create ensemble
        ensemble_detectors = [
            IsolationForestDetector(contamination=0.05),
            OneClassSVMDetector(nu=0.05),
            LocalOutlierFactorDetector(contamination=0.05)
        ]
        ensemble = EnsembleDetector(ensemble_detectors)
        manager.add_model('Ensemble', ensemble)
        
        # Fit all models
        manager.fit_all(feature_matrix)
        
        # Get predictions
        results = manager.predict_all(feature_matrix)
        
        # Compare models
        true_labels = np.array([1 if i >= n_samples else 0 for i in range(len(feature_dicts))])
        comparison = manager.compare_models(feature_matrix, true_labels)
        
        # Print results
        logger.info("\nModel Comparison:")
        logger.info("-" * 60)
        for name, metrics in comparison.items():
            logger.info(f"{name}:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
            logger.info("")
        
        # Save results
        manager.save_results("reports/advanced_ml_results.json")
        
        logger.info("=" * 60)
        logger.info("ADVANCED ML DEMO COMPLETED!")
        logger.info("=" * 60)
        
    except ImportError as e:
        logger.error(f"Required dependencies not available: {e}")
        logger.info("Install with: pip install scikit-learn")
        logger.info("The basic rule-based pipeline is still working!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    demo_advanced_ml() 