# Plugging ML Models into the Pipeline

Integrating ML models into a log anomaly detection pipeline requires careful design for scalability, maintainability, and observability.

## Model Serving Approaches
- **Batch inference**: Process logs in batches (e.g., hourly, daily)
- **Real-time inference**: Score logs as they arrive (streaming)
- **Hybrid**: Combine both for flexibility

## Model API Design
- **REST API**: Expose model as a web service (e.g., FastAPI)
- **gRPC**: For high-performance, low-latency needs
- **Input/Output schema**: Standardize feature vectors and prediction format

## Example: FastAPI Model Endpoint
```python
from fastapi import FastAPI
import numpy as np

app = FastAPI()

@app.post('/predict')
def predict(features: list):
    # features: list of floats
    prediction = model.predict(np.array([features]))
    return {'anomaly': int(prediction[0])}
```

## Model Versioning and Registry
- **Track model versions**: Metadata, performance, deployment date
- **Rollback**: Quickly revert to previous model if needed
- **A/B testing**: Compare models in production

## Monitoring and Logging
- **Prediction logs**: Store inputs, outputs, and metadata
- **Performance metrics**: Latency, throughput, error rates
- **Model drift detection**: Monitor for changes in data or performance

## Integration Patterns
- **Microservice**: Each model as a separate service
- **Sidecar**: Model runs alongside main app
- **Centralized scoring**: All models in a single service

## Diagram: Model Integration
```mermaid
graph LR
  A[Log Ingestion] --> B[Feature Engineering]
  B --> C[Model API Service]
  C --> D[Anomaly Results]
  D --> E[Alerting/Storage]
```

---

**Next: [Case Studies & Practical Examples](case_studies.md) →** 