# One-Class SVM: Theory, Math, and Application

One-Class Support Vector Machine (SVM) is an unsupervised algorithm that learns the boundary of normal data and identifies points that fall outside as anomalies.

## Theory and Intuition
- **Support Vector Machines**: Find a hyperplane that separates data points from the origin in a high-dimensional space.
- **One-Class SVM**: Tries to include most data in a region (the "normal" class) and flags points outside as anomalies.
- **Kernel trick**: Allows modeling of non-linear boundaries.

## Mathematical Formulation
Given data $X = \{x_1, ..., x_n\}$, solve:
$$
\min_{w, \rho, \xi} \frac{1}{2} \|w\|^2 + \frac{1}{\nu n} \sum_{i=1}^n \xi_i - \rho
$$
subject to:
$$
(w \cdot \phi(x_i)) \geq \rho - \xi_i, \quad \xi_i \geq 0
$$
where $\nu$ controls the fraction of outliers, $\phi$ is the kernel mapping.

## Algorithm Steps
1. Map data to high-dimensional space using kernel (e.g., RBF).
2. Find the smallest region that contains most data.
3. Points outside this region are anomalies.

## Practical Example (Python)
```python
from sklearn.svm import OneClassSVM
import pandas as pd

# Assume df is a DataFrame of engineered log features
model = OneClassSVM(kernel='rbf', nu=0.01, gamma='scale')
model.fit(df)
predictions = model.predict(df)  # -1: anomaly, 1: normal
```

## Visual Intuition
- Normal data forms a "cloud"; SVM finds a boundary around it.
- Anomalies fall outside the boundary.

## Why One-Class SVM for Log Anomaly Detection?
- **Effective for non-linear, complex patterns**
- **No need for labeled anomalies**
- **Works well with moderate-sized datasets**

## Example Use Case
- Detecting unusual API usage patterns
- Identifying rare error sequences in application logs

---

**Next: [Local Outlier Factor: Theory, Math, and Application](lof.md) →** 