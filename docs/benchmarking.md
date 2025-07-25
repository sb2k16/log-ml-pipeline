# Benchmarking ML Models for Anomaly Detection

Benchmarking is the process of evaluating and comparing the performance of anomaly detection models using quantitative metrics and standardized datasets.

## Why Benchmark?
- **Objective comparison**: Identify the best model for your use case
- **Understand trade-offs**: Precision vs. recall, speed vs. accuracy
- **Guide model selection and tuning**

## Key Evaluation Metrics
- **Precision**: $\frac{TP}{TP + FP}$
- **Recall**: $\frac{TP}{TP + FN}$
- **F1-score**: $2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}$
- **ROC AUC**: Area under the Receiver Operating Characteristic curve
- **Average Precision (AP)**: Area under the Precision-Recall curve
- **Confusion Matrix**: TP, FP, TN, FN counts
- **Custom metrics**: Detection rate, false alarm rate, Matthews correlation, Cohen's kappa

## Benchmarking Methodology
1. **Prepare labeled test data** (if possible)
2. **Split data**: Train/test or cross-validation
3. **Run each model**: Collect predictions and scores
4. **Compute metrics**: For each model
5. **Visualize results**: ROC, PR curves, confusion matrices
6. **Compare and interpret**: Identify strengths/weaknesses

## Practical Example (Python)
```python
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

y_true = [0, 1, 0, 0, 1]  # 1: anomaly, 0: normal
y_pred = [0, 1, 0, 1, 1]

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)
```

## What Insights Does Benchmarking Provide?
- **Best model for your data**
- **Sensitivity to rare anomalies**
- **Robustness to noise and drift**
- **Operational trade-offs** (e.g., false alarms vs. missed anomalies)

## Model Comparison Table (Example)
| Model              | Precision | Recall | F1   | ROC AUC |
|--------------------|-----------|--------|------|---------|
| Isolation Forest   | 0.92      | 0.85   | 0.88 | 0.93    |
| One-Class SVM      | 0.89      | 0.80   | 0.84 | 0.91    |
| LOF                | 0.85      | 0.78   | 0.81 | 0.89    |
| Autoencoder        | 0.94      | 0.88   | 0.91 | 0.95    |
| LSTM               | 0.96      | 0.90   | 0.93 | 0.97    |

---

**Next: [Cloud-Native Infrastructure: From Ingestion to Detection](infrastructure.md) →** 