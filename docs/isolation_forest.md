# Isolation Forest: Theory, Math, and Application

Isolation Forest is a tree-based, unsupervised anomaly detection algorithm that isolates anomalies instead of profiling normal data. It is highly effective for high-dimensional, large-scale log data.

## Theory and Intuition
- **Anomalies are few and different**: They are easier to isolate than normal points.
- **Random partitioning**: The algorithm recursively splits data using random features and split values.
- **Isolation path length**: Anomalies are isolated faster (shorter path) than normal points.

## Mathematical Formulation
- Build $t$ random trees (isolation trees) on subsamples of the data.
- For each point $x$, compute the average path length $h(x)$ across all trees.
- The anomaly score is:
  $$
  s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}
  $$
  where $E(h(x))$ is the average path length, $c(n)$ is the average path length of unsuccessful search in a Binary Search Tree ($c(n) \approx 2H(n-1) - 2(n-1)/n$, $H(i)$ is the $i$-th harmonic number).
- **Interpretation**: $s(x) \approx 1$ means highly anomalous, $s(x) \ll 0.5$ means normal.

## Algorithm Steps
1. Randomly select a feature and split value to partition the data.
2. Recursively repeat until each point is isolated or max depth is reached.
3. Compute path lengths for all points.
4. Aggregate scores across all trees.

## Practical Example (Python)
```python
from sklearn.ensemble import IsolationForest
import pandas as pd

# Assume df is a DataFrame of engineered log features
model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
model.fit(df)
scores = model.decision_function(df)
predictions = model.predict(df)  # -1: anomaly, 1: normal
```

## Visual Intuition
- Anomalies are isolated in fewer splits (shorter tree paths).
- Normal points require more splits to be isolated.

## Why Isolation Forest for Log Anomaly Detection?
- **No need for labeled data**
- **Scalable to millions of logs**
- **Works with high-dimensional, sparse features**
- **Robust to irrelevant features**

## Example Use Case
- Detecting rare error patterns in web server logs
- Identifying outlier user sessions in authentication logs

---

**Next: [One-Class SVM: Theory, Math, and Application](one_class_svm.md) →** 