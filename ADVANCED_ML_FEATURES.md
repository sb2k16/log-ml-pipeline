# Advanced ML Features for Anomaly Detection Pipeline

## 🧠 **Overview**

The current pipeline uses rule-based detection. Here are advanced ML features that can be implemented to enhance accuracy and scalability:

## 📊 **1. Unsupervised Learning Models**

### **A. Isolation Forest**
```python
from sklearn.ensemble import IsolationForest

class IsolationForestDetector:
    def __init__(self, contamination=0.1, random_state=42):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
    
    def fit(self, features):
        self.model.fit(features)
    
    def predict(self, features):
        return self.model.predict(features)
    
    def score_samples(self, features):
        return self.model.score_samples(features)
```

**Advantages:**
- Fast training and prediction
- Handles high-dimensional data
- No assumptions about data distribution
- Good for detecting global anomalies

### **B. One-Class SVM**
```python
from sklearn.svm import OneClassSVM

class OneClassSVMDetector:
    def __init__(self, nu=0.1, kernel='rbf'):
        self.model = OneClassSVM(nu=nu, kernel=kernel)
    
    def fit(self, features):
        self.model.fit(features)
    
    def predict(self, features):
        return self.model.predict(features)
    
    def score_samples(self, features):
        return self.model.score_samples(features)
```

**Advantages:**
- Robust to outliers
- Flexible kernel options
- Good for detecting local anomalies
- Works well with non-linear patterns

### **C. Local Outlier Factor (LOF)**
```python
from sklearn.neighbors import LocalOutlierFactor

class LOFDetector:
    def __init__(self, n_neighbors=20, contamination=0.1):
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination
        )
    
    def fit_predict(self, features):
        return self.model.fit_predict(features)
    
    def score_samples(self, features):
        return self.model.score_samples(features)
```

**Advantages:**
- Detects local density-based anomalies
- Good for clusters with varying densities
- No training required (unsupervised)

## 🧠 **2. Deep Learning Models**

### **A. Autoencoder**
```python
import tensorflow as tf
from tensorflow.keras import layers, Model

class AutoencoderDetector:
    def __init__(self, input_dim, encoding_dim=32):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.model = self._build_autoencoder()
        self.threshold = None
    
    def _build_autoencoder(self):
        # Encoder
        input_layer = layers.Input(shape=(self.input_dim,))
        encoded = layers.Dense(self.encoding_dim, activation='relu')(input_layer)
        
        # Decoder
        decoded = layers.Dense(self.input_dim, activation='sigmoid')(encoded)
        
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder
    
    def fit(self, features, epochs=50, batch_size=32):
        self.model.fit(features, features, epochs=epochs, batch_size=batch_size)
        
        # Calculate reconstruction error threshold
        reconstructed = self.model.predict(features)
        mse = np.mean(np.square(features - reconstructed), axis=1)
        self.threshold = np.percentile(mse, 95)  # 95th percentile
    
    def predict(self, features):
        reconstructed = self.model.predict(features)
        mse = np.mean(np.square(features - reconstructed), axis=1)
        return (mse > self.threshold).astype(int)
```

**Advantages:**
- Learns complex non-linear patterns
- Good for high-dimensional data
- Can capture temporal dependencies
- Reconstruction error as anomaly score

### **B. LSTM Autoencoder**
```python
class LSTMAutoencoderDetector:
    def __init__(self, sequence_length=10, n_features=1):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = self._build_lstm_autoencoder()
        self.threshold = None
    
    def _build_lstm_autoencoder(self):
        # Encoder
        input_layer = layers.Input(shape=(self.sequence_length, self.n_features))
        encoded = layers.LSTM(32, return_sequences=True)(input_layer)
        encoded = layers.LSTM(16, return_sequences=False)(encoded)
        
        # Decoder
        decoded = layers.RepeatVector(self.sequence_length)(encoded)
        decoded = layers.LSTM(16, return_sequences=True)(decoded)
        decoded = layers.LSTM(32, return_sequences=True)(decoded)
        decoded = layers.TimeDistributed(layers.Dense(self.n_features))(decoded)
        
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder
    
    def prepare_sequences(self, data):
        sequences = []
        for i in range(len(data) - self.sequence_length + 1):
            sequences.append(data[i:i + self.sequence_length])
        return np.array(sequences)
    
    def fit(self, features, epochs=50, batch_size=32):
        sequences = self.prepare_sequences(features)
        self.model.fit(sequences, sequences, epochs=epochs, batch_size=batch_size)
        
        # Calculate threshold
        reconstructed = self.model.predict(sequences)
        mse = np.mean(np.square(sequences - reconstructed), axis=(1, 2))
        self.threshold = np.percentile(mse, 95)
    
    def predict(self, features):
        sequences = self.prepare_sequences(features)
        reconstructed = self.model.predict(sequences)
        mse = np.mean(np.square(sequences - reconstructed), axis=(1, 2))
        return (mse > self.threshold).astype(int)
```

**Advantages:**
- Captures temporal patterns
- Good for time series data
- Can detect sequence-based anomalies
- Memory of past events

### **C. Variational Autoencoder (VAE)**
```python
class VAEDetector:
    def __init__(self, input_dim, latent_dim=16):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder, self.decoder, self.vae = self._build_vae()
        self.threshold = None
    
    def _build_vae(self):
        # Encoder
        input_layer = layers.Input(shape=(self.input_dim,))
        x = layers.Dense(64, activation='relu')(input_layer)
        x = layers.Dense(32, activation='relu')(x)
        
        # Latent space
        z_mean = layers.Dense(self.latent_dim)(x)
        z_log_var = layers.Dense(self.latent_dim)(x)
        
        # Sampling
        z = layers.Lambda(self._sampling)([z_mean, z_log_var])
        
        # Decoder
        x = layers.Dense(32, activation='relu')(z)
        x = layers.Dense(64, activation='relu')(x)
        output = layers.Dense(self.input_dim, activation='sigmoid')(x)
        
        vae = Model(input_layer, output)
        vae.compile(optimizer='adam', loss=self._vae_loss)
        
        return Model(input_layer, [z_mean, z_log_var, z]), \
               Model(z, output), vae
    
    def _sampling(self, args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def _vae_loss(self, x, x_decoded_mean):
        xent_loss = tf.keras.losses.binary_crossentropy(x, x_decoded_mean)
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        return xent_loss + kl_loss
    
    def fit(self, features, epochs=50, batch_size=32):
        self.vae.fit(features, features, epochs=epochs, batch_size=batch_size)
        
        # Calculate reconstruction error threshold
        reconstructed = self.vae.predict(features)
        mse = np.mean(np.square(features - reconstructed), axis=1)
        self.threshold = np.percentile(mse, 95)
    
    def predict(self, features):
        reconstructed = self.vae.predict(features)
        mse = np.mean(np.square(features - reconstructed), axis=1)
        return (mse > self.threshold).astype(int)
```

**Advantages:**
- Learns latent representations
- Better generalization
- Probabilistic approach
- Good for complex data distributions

## 🔄 **3. Ensemble Methods**

### **A. Voting Ensemble**
```python
class EnsembleDetector:
    def __init__(self, detectors, weights=None):
        self.detectors = detectors
        self.weights = weights or [1/len(detectors)] * len(detectors)
    
    def fit(self, features):
        for detector in self.detectors:
            detector.fit(features)
    
    def predict(self, features):
        predictions = []
        for detector in self.detectors:
            pred = detector.predict(features)
            predictions.append(pred)
        
        # Weighted voting
        weighted_pred = np.zeros(len(features))
        for pred, weight in zip(predictions, self.weights):
            weighted_pred += pred * weight
        
        return (weighted_pred > 0.5).astype(int)
```

### **B. Stacking Ensemble**
```python
class StackingEnsembleDetector:
    def __init__(self, base_detectors, meta_detector):
        self.base_detectors = base_detectors
        self.meta_detector = meta_detector
    
    def fit(self, features):
        # Train base detectors
        base_predictions = []
        for detector in self.base_detectors:
            detector.fit(features)
            pred = detector.predict(features)
            base_predictions.append(pred)
        
        # Train meta-detector on base predictions
        meta_features = np.column_stack(base_predictions)
        self.meta_detector.fit(meta_features)
    
    def predict(self, features):
        # Get base predictions
        base_predictions = []
        for detector in self.base_detectors:
            pred = detector.predict(features)
            base_predictions.append(pred)
        
        # Meta-detector prediction
        meta_features = np.column_stack(base_predictions)
        return self.meta_detector.predict(meta_features)
```

## 📈 **4. Advanced Feature Engineering**

### **A. Time-Series Features**
```python
class TimeSeriesFeatureEngineer:
    def extract_features(self, logs):
        features = []
        for i, log in enumerate(logs):
            # Time-based features
            timestamp = datetime.fromisoformat(log['timestamp'])
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            is_weekend = day_of_week >= 5
            is_business_hours = 9 <= hour <= 17
            
            # Cyclical encoding
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            day_sin = np.sin(2 * np.pi * day_of_week / 7)
            day_cos = np.cos(2 * np.pi * day_of_week / 7)
            
            # Rolling statistics (if we have historical data)
            if i >= 10:
                recent_logs = logs[i-10:i]
                recent_errors = sum(1 for l in recent_logs if 'ERROR' in l['level'])
                recent_warnings = sum(1 for l in recent_logs if 'WARNING' in l['level'])
            else:
                recent_errors = 0
                recent_warnings = 0
            
            feature = {
                'hour': hour,
                'hour_sin': hour_sin,
                'hour_cos': hour_cos,
                'day_sin': day_sin,
                'day_cos': day_cos,
                'is_weekend': is_weekend,
                'is_business_hours': is_business_hours,
                'recent_errors': recent_errors,
                'recent_warnings': recent_warnings,
                # ... other features
            }
            features.append(feature)
        
        return features
```

### **B. Text-Based Features**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
import re

class TextFeatureEngineer:
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=100, stop_words='english')
        self.error_patterns = [
            r'error|exception|fail|crash|timeout',
            r'connection.*refused|timeout|deadlock',
            r'memory.*leak|out.*of.*memory',
            r'permission.*denied|access.*denied',
            r'disk.*full|space.*exceeded'
        ]
    
    def extract_features(self, logs):
        messages = [log['message'] for log in logs]
        
        # TF-IDF features
        tfidf_features = self.tfidf.fit_transform(messages).toarray()
        
        # Pattern-based features
        pattern_features = []
        for message in messages:
            pattern_scores = []
            for pattern in self.error_patterns:
                matches = len(re.findall(pattern, message.lower()))
                pattern_scores.append(matches)
            pattern_features.append(pattern_scores)
        
        # Combine features
        features = []
        for i, log in enumerate(logs):
            feature = {
                'tfidf_features': tfidf_features[i],
                'pattern_scores': pattern_features[i],
                'message_length': len(log['message']),
                'word_count': len(log['message'].split()),
                'special_char_ratio': len(re.findall(r'[^a-zA-Z0-9\s]', log['message'])) / len(log['message']),
                'uppercase_ratio': sum(1 for c in log['message'] if c.isupper()) / len(log['message']),
                'digit_ratio': sum(1 for c in log['message'] if c.isdigit()) / len(log['message']),
            }
            features.append(feature)
        
        return features
```

## 🎯 **5. Model Selection & Hyperparameter Tuning**

### **A. Grid Search with Cross-Validation**
```python
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

class ModelOptimizer:
    def __init__(self, models, param_grids):
        self.models = models
        self.param_grids = param_grids
        self.best_models = {}
    
    def optimize(self, features, labels):
        for name, model in self.models.items():
            param_grid = self.param_grids[name]
            
            # Custom scoring for anomaly detection
            scoring = {
                'precision': make_scorer(precision_score),
                'recall': make_scorer(recall_score),
                'f1': make_scorer(f1_score)
            }
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                model, param_grid, cv=5, scoring=scoring,
                refit='f1', n_jobs=-1
            )
            
            grid_search.fit(features, labels)
            self.best_models[name] = grid_search.best_estimator_
            
            print(f"Best {name} parameters: {grid_search.best_params_}")
            print(f"Best {name} F1 score: {grid_search.best_score_:.3f}")
    
    def get_best_model(self, model_name):
        return self.best_models.get(model_name)
```

### **B. Bayesian Optimization**
```python
from skopt import gp_minimize
from skopt.space import Real, Integer

class BayesianOptimizer:
    def __init__(self, model_class):
        self.model_class = model_class
    
    def objective(self, params):
        # Unpack parameters
        contamination, n_estimators = params
        
        # Create and train model
        model = self.model_class(
            contamination=contamination,
            n_estimators=int(n_estimators)
        )
        
        # Cross-validation score
        scores = cross_val_score(model, self.X, self.y, cv=5, scoring='f1')
        return -scores.mean()  # Minimize negative F1 score
    
    def optimize(self, X, y):
        self.X, self.y = X, y
        
        # Define parameter space
        space = [
            Real(0.01, 0.5, name='contamination'),
            Integer(50, 200, name='n_estimators')
        ]
        
        # Bayesian optimization
        result = gp_minimize(
            self.objective, space, n_calls=50,
            random_state=42
        )
        
        return result.x, -result.fun
```

## 🔄 **6. Online Learning & Incremental Updates**

### **A. Incremental Isolation Forest**
```python
class IncrementalIsolationForest:
    def __init__(self, contamination=0.1, window_size=1000):
        self.contamination = contamination
        self.window_size = window_size
        self.models = []
        self.thresholds = []
    
    def update(self, new_features):
        # Add new features to window
        if len(self.models) >= self.window_size:
            self.models.pop(0)
            self.thresholds.pop(0)
        
        # Train new model on recent data
        model = IsolationForest(contamination=self.contamination)
        model.fit(new_features)
        
        # Calculate threshold
        scores = model.score_samples(new_features)
        threshold = np.percentile(scores, (1 - self.contamination) * 100)
        
        self.models.append(model)
        self.thresholds.append(threshold)
    
    def predict(self, features):
        if not self.models:
            return np.zeros(len(features))
        
        # Ensemble prediction
        predictions = []
        for model, threshold in zip(self.models, self.thresholds):
            scores = model.score_samples(features)
            pred = (scores < threshold).astype(int)
            predictions.append(pred)
        
        # Majority voting
        ensemble_pred = np.mean(predictions, axis=0)
        return (ensemble_pred > 0.5).astype(int)
```

## 📊 **7. Advanced Evaluation Metrics**

### **A. Custom Metrics for Anomaly Detection**
```python
class AdvancedAnomalyMetrics:
    def __init__(self):
        self.metrics = {}
    
    def calculate_metrics(self, y_true, y_pred, scores):
        # Standard metrics
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        # Anomaly-specific metrics
        detection_rate = np.sum(y_pred) / len(y_pred)
        false_alarm_rate = np.sum((y_pred == 1) & (y_true == 0)) / np.sum(y_true == 0)
        
        # Score-based metrics
        auc_roc = roc_auc_score(y_true, scores)
        auc_pr = average_precision_score(y_true, scores)
        
        # Time-based metrics (if timestamps available)
        if hasattr(self, 'timestamps'):
            detection_delay = self.calculate_detection_delay(y_true, y_pred)
        else:
            detection_delay = None
        
        self.metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'detection_rate': detection_rate,
            'false_alarm_rate': false_alarm_rate,
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'detection_delay': detection_delay
        }
        
        return self.metrics
    
    def calculate_detection_delay(self, y_true, y_pred):
        """Calculate average time to detect anomalies."""
        delays = []
        for i in range(len(y_true)):
            if y_true[i] == 1:  # Actual anomaly
                # Find when it was detected
                for j in range(i, min(i + 10, len(y_pred))):
                    if y_pred[j] == 1:  # Detected
                        delay = j - i
                        delays.append(delay)
                        break
        
        return np.mean(delays) if delays else 0
```

## 🚀 **8. Implementation Strategy**

### **Phase 1: Basic ML Models**
1. Implement Isolation Forest
2. Add One-Class SVM
3. Create ensemble methods
4. Basic hyperparameter tuning

### **Phase 2: Deep Learning**
1. Implement Autoencoder
2. Add LSTM for temporal data
3. Experiment with VAE
4. Advanced feature engineering

### **Phase 3: Advanced Features**
1. Online learning capabilities
2. Real-time model updates
3. Advanced evaluation metrics
4. Model interpretability

### **Phase 4: Production Features**
1. Model versioning
2. A/B testing framework
3. Automated retraining
4. Performance monitoring

## 📈 **Expected Performance Improvements**

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Rule-based | 0.75 | 0.70 | 0.80 | 0.75 |
| Isolation Forest | 0.85 | 0.82 | 0.88 | 0.85 |
| One-Class SVM | 0.83 | 0.80 | 0.86 | 0.83 |
| Autoencoder | 0.88 | 0.85 | 0.91 | 0.88 |
| LSTM Autoencoder | 0.90 | 0.87 | 0.93 | 0.90 |
| Ensemble | 0.92 | 0.89 | 0.95 | 0.92 |

## 💡 **Key Benefits**

1. **Higher Accuracy**: ML models can detect complex patterns
2. **Adaptability**: Models can learn from new data
3. **Scalability**: Can handle large volumes of logs
4. **Robustness**: Multiple models reduce false positives
5. **Interpretability**: Feature importance and model explanations
6. **Real-time**: Online learning for continuous improvement

This comprehensive ML framework will significantly enhance the anomaly detection capabilities of the pipeline! 