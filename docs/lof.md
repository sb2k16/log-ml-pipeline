# Local Outlier Factor: Theory, Math, and Application

Local Outlier Factor (LOF) is a density-based anomaly detection algorithm that identifies points that have a substantially lower density than their neighbors.

## Theory and Intuition
- **Local density**: Anomalies are points in regions of lower density compared to their neighbors.
- **Relative comparison**: LOF compares the local density of a point to that of its $k$ nearest neighbors.

## Mathematical Formulation
- For each point $p$, compute the $k$-distance (distance to the $k$-th nearest neighbor).
- Compute the local reachability density (LRD):
  $$
  \text{LRD}_k(p) = \left( \frac{\sum_{o \in N_k(p)} \max\{\text{dist}(p, o), k\text{-dist}(o)\}}{|N_k(p)|} \right)^{-1}
  $$
- Compute LOF score:
  $$
  \text{LOF}_k(p) = \frac{\sum_{o \in N_k(p)} \frac{\text{LRD}_k(o)}{\text{LRD}_k(p)}}{|N_k(p)|}
  $$
- $\text{LOF}_k(p) \gg 1$ indicates an outlier.

## Algorithm Steps
1. For each point, find $k$ nearest neighbors.
2. Compute LRD for each point.
3. Compute LOF score for each point.
4. Points with high LOF are anomalies.

## Practical Example (Python)
```python
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd

# Assume df is a DataFrame of engineered log features
model = LocalOutlierFactor(n_neighbors=20, contamination=0.01)
predictions = model.fit_predict(df)  # -1: anomaly, 1: normal
lof_scores = -model.negative_outlier_factor_
```

## Visual Intuition
- Normal points are in dense clusters.
- Anomalies are isolated or in sparse regions.

## Why LOF for Log Anomaly Detection?
- **Detects local, context-dependent anomalies**
- **No need for labeled data**
- **Good for mixed-density data**

## Example Use Case
- Detecting rare error bursts in microservice logs
- Identifying outlier transactions in payment logs

---

**Next: [Autoencoders: Deep Learning for Anomaly Detection](autoencoder.md) →** 