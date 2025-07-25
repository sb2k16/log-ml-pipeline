# Feature Engineering for Log Data

Feature engineering is the process of transforming raw log entries into structured, informative variables (features) that machine learning models can use for anomaly detection.

## Why Feature Engineering Matters
- **Raw logs are unstructured**: ML models need numerical, structured input
- **Good features = better detection**: The right features reveal hidden anomalies
- **Domain knowledge**: Feature engineering encodes system and application knowledge

## Types of Features for Log Anomaly Detection

### 1. **Time-Based Features**
- **Hour of day**: $h = \text{timestamp.hour}$
- **Day of week**: $d = \text{timestamp.weekday()}$
- **Is weekend**: $w = (d \geq 5)$
- **Cyclical encoding**:
  - $\text{hour\_sin} = \sin(2\pi h / 24)$
  - $\text{hour\_cos} = \cos(2\pi h / 24)$

### 2. **Text-Based Features**
- **Message length**: $l = \text{len(message)}$
- **Word count**: $w = \text{len(message.split())}$
- **Special character count**: $s = \sum 1_{c \notin [a-zA-Z0-9 ]}$
- **TF-IDF vectors**: $\text{tfidf}(\text{message})$
- **Pattern matches**: $\text{count}(\text{regex}, \text{message})$

### 3. **Statistical Features**
- **Rolling mean/std**: $\mu_t = \frac{1}{N}\sum_{i=t-N}^t x_i$
- **Error rate**: $\text{errors}/\text{total}$ in window
- **Z-score**: $z = \frac{x - \mu}{\sigma}$

### 4. **Categorical Features**
- **Log level**: One-hot encode (INFO, WARNING, ERROR, ...)
- **Source/service**: Encode as integer or one-hot
- **Status code**: HTTP status, DB error code, etc.

## Practical Example: Feature Extraction
```python
import re
from datetime import datetime

def extract_features(log):
    timestamp = datetime.fromisoformat(log['timestamp'])
    features = {
        'hour': timestamp.hour,
        'hour_sin': np.sin(2 * np.pi * timestamp.hour / 24),
        'hour_cos': np.cos(2 * np.pi * timestamp.hour / 24),
        'is_weekend': int(timestamp.weekday() >= 5),
        'message_length': len(log['message']),
        'word_count': len(log['message'].split()),
        'special_chars': len(re.findall(r'[^a-zA-Z0-9 ]', log['message'])),
        'log_level': 1 if log['level'] == 'ERROR' else 0,
        # ... add more features
    }
    return features
```

## Feature Engineering Pipeline
1. **Parse logs**: Extract fields (timestamp, level, message, ...)
2. **Transform**: Apply feature functions
3. **Vectorize**: Convert to numpy arrays or DataFrames
4. **Scale/normalize**: Standardize features for ML models

## Impact on Anomaly Detection
- **Improved accuracy**: Models can distinguish normal vs. anomalous patterns
- **Interpretability**: Feature importance reveals root causes
- **Adaptability**: New features can be added as systems evolve

---

**Next: [Isolation Forest: Theory, Math, and Application](isolation_forest.md) →** 