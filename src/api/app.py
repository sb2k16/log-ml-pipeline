"""
FastAPI Web Service for Anomaly Detection Pipeline

Provides REST API endpoints for:
- Data ingestion
- Model training
- Anomaly detection
- Results retrieval
- Health monitoring
"""

import logging
import json
import yaml
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.ingestion.log_parser import LogParser
from src.processing.feature_engineer import FeatureEngineer
from src.models.isolation_forest import IsolationForestDetector
from src.models.rule_based import RuleBasedDetector
from src.evaluation.metrics import AnomalyDetectionMetrics

logger = logging.getLogger(__name__)

# Pydantic models for API
class LogEntry(BaseModel):
    timestamp: str
    level: str
    message: str
    source: str
    service: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    duration: Optional[float] = None
    status_code: Optional[int] = None
    method: Optional[str] = None
    url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class AnomalyResult(BaseModel):
    timestamp: str
    log_entry: LogEntry
    is_anomaly: bool
    anomaly_score: float
    model_name: str
    confidence: float

class ModelInfo(BaseModel):
    model_name: str
    model_type: str
    is_fitted: bool
    n_features: int
    training_time: float
    last_updated: str

class PipelineStatus(BaseModel):
    status: str
    models_loaded: int
    total_predictions: int
    last_prediction_time: Optional[str]
    uptime_seconds: float

# Initialize FastAPI app
app = FastAPI(
    title="Anomaly Detection Pipeline API",
    description="REST API for log anomaly detection using ML models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
config = {}
log_parser = None
feature_engineer = None
models = {}
metrics = None
pipeline_stats = {
    "start_time": datetime.now(),
    "total_predictions": 0,
    "last_prediction_time": None
}

def load_config():
    """Load configuration from file."""
    global config
    try:
        config_path = Path("config/config.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info("Configuration loaded successfully")
        else:
            config = {}
            logger.warning("Config file not found, using defaults")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        config = {}

def initialize_components():
    """Initialize all pipeline components."""
    global log_parser, feature_engineer, models, metrics
    
    try:
        # Initialize components
        log_parser = LogParser(config.get("log_parsing", {}))
        feature_engineer = FeatureEngineer(config)
        metrics = AnomalyDetectionMetrics(config)
        
        # Initialize models
        models["isolation_forest"] = IsolationForestDetector(config)
        models["rule_based"] = RuleBasedDetector(config)
        
        logger.info("Pipeline components initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting Anomaly Detection Pipeline API...")
    load_config()
    initialize_components()

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Anomaly Detection Pipeline API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(models),
        "uptime_seconds": (datetime.now() - pipeline_stats["start_time"]).total_seconds()
    }

@app.post("/ingest/logs")
async def ingest_logs(logs: List[LogEntry]):
    """Ingest log entries for processing."""
    try:
        logger.info(f"Ingesting {len(logs)} log entries")
        
        # Convert to DataFrame
        log_data = []
        for log in logs:
            log_dict = log.dict()
            log_data.append(log_dict)
        
        df = pd.DataFrame(log_data)
        
        # Parse logs
        parsed_logs = log_parser.parse_batch([json.dumps(log) for log in log_data])
        df_parsed = log_parser.to_dataframe(parsed_logs)
        
        # Engineer features
        df_features = feature_engineer.engineer_features(df_parsed)
        
        return {
            "message": f"Successfully ingested {len(logs)} log entries",
            "parsed_entries": len(df_parsed),
            "features_engineered": len(df_features.columns),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to ingest logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/logs")
async def upload_logs(file: UploadFile = File(...)):
    """Upload log file for processing."""
    try:
        logger.info(f"Uploading log file: {file.filename}")
        
        # Read file content
        content = await file.read()
        log_lines = content.decode('utf-8').splitlines()
        
        # Parse logs
        parsed_logs = log_parser.parse_batch(log_lines)
        df = log_parser.to_dataframe(parsed_logs)
        
        # Engineer features
        df_features = feature_engineer.engineer_features(df)
        
        return {
            "message": f"Successfully uploaded and processed {len(log_lines)} log lines",
            "parsed_entries": len(df),
            "features_engineered": len(df_features.columns),
            "filename": file.filename,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to upload logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def train_models(background_tasks: BackgroundTasks):
    """Train all models."""
    try:
        logger.info("Starting model training...")
        
        # This would typically load training data
        # For now, we'll use sample data
        sample_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
            'level': np.random.choice(['INFO', 'WARNING', 'ERROR'], 100),
            'message': [f"Sample log {i}" for i in range(100)],
            'source': np.random.choice(['web_server', 'database', 'auth_service'], 100)
        })
        
        # Engineer features
        df_features = feature_engineer.engineer_features(sample_data)
        
        # Train models
        for model_name, model in models.items():
            try:
                logger.info(f"Training {model_name}...")
                model.fit(df_features)
                logger.info(f"{model_name} training completed")
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
        
        return {
            "message": "Model training completed",
            "models_trained": len(models),
            "training_data_size": len(df_features),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to train models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect")
async def detect_anomalies(logs: List[LogEntry]):
    """Detect anomalies in log entries."""
    try:
        logger.info(f"Detecting anomalies in {len(logs)} log entries")
        
        # Convert to DataFrame
        log_data = []
        for log in logs:
            log_dict = log.dict()
            log_data.append(log_dict)
        
        df = pd.DataFrame(log_data)
        
        # Parse and engineer features
        parsed_logs = log_parser.parse_batch([json.dumps(log) for log in log_data])
        df_parsed = log_parser.to_dataframe(parsed_logs)
        df_features = feature_engineer.engineer_features(df_parsed)
        
        # Detect anomalies with all models
        results = []
        for model_name, model in models.items():
            try:
                if hasattr(model, 'predict') and model.is_fitted:
                    predictions = model.predict(df_features)
                    scores = None
                    
                    if hasattr(model, 'predict_proba'):
                        scores = model.predict_proba(df_features)
                    elif hasattr(model, 'get_anomaly_scores'):
                        scores = model.get_anomaly_scores(df_features)
                    
                    for i, (log, pred) in enumerate(zip(logs, predictions)):
                        result = AnomalyResult(
                            timestamp=datetime.now().isoformat(),
                            log_entry=log,
                            is_anomaly=bool(pred),
                            anomaly_score=float(scores[i]) if scores is not None else 0.0,
                            model_name=model_name,
                            confidence=abs(scores[i] - 0.5) * 2 if scores is not None else 0.0
                        )
                        results.append(result)
                        
            except Exception as e:
                logger.error(f"Failed to detect anomalies with {model_name}: {e}")
        
        # Update statistics
        pipeline_stats["total_predictions"] += len(logs)
        pipeline_stats["last_prediction_time"] = datetime.now().isoformat()
        
        return {
            "message": f"Anomaly detection completed for {len(logs)} log entries",
            "results": results,
            "total_anomalies": sum(1 for r in results if r.is_anomaly),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to detect anomalies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def get_models():
    """Get information about all models."""
    try:
        model_info = []
        for model_name, model in models.items():
            info = ModelInfo(
                model_name=model_name,
                model_type=getattr(model, 'model_type', 'unknown'),
                is_fitted=getattr(model, 'is_fitted', False),
                n_features=len(getattr(model, 'feature_names', [])),
                training_time=getattr(model, 'training_time', 0.0),
                last_updated=datetime.now().isoformat()
            )
            model_info.append(info)
        
        return {
            "models": model_info,
            "total_models": len(model_info)
        }
        
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    """Get pipeline status."""
    try:
        status = PipelineStatus(
            status="running",
            models_loaded=len([m for m in models.values() if getattr(m, 'is_fitted', False)]),
            total_predictions=pipeline_stats["total_predictions"],
            last_prediction_time=pipeline_stats["last_prediction_time"],
            uptime_seconds=(datetime.now() - pipeline_stats["start_time"]).total_seconds()
        )
        
        return status
        
    except Exception as e:
        logger.error(f"Failed to get status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get pipeline metrics."""
    try:
        return {
            "total_predictions": pipeline_stats["total_predictions"],
            "uptime_seconds": (datetime.now() - pipeline_stats["start_time"]).total_seconds(),
            "models_loaded": len([m for m in models.values() if getattr(m, 'is_fitted', False)]),
            "last_prediction_time": pipeline_stats["last_prediction_time"]
        }
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 