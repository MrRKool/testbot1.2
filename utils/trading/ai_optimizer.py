import logging
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import numpy as np
import torch
from prometheus_client import Counter, Gauge, Histogram
from tenacity import retry, stop_after_attempt, wait_exponential
import mlflow
from mlflow.tracking import MlflowClient

# Prometheus metrics
MODEL_PREDICTIONS = Counter('model_predictions_total', 'Total number of model predictions')
MODEL_LATENCY = Histogram('model_prediction_latency_seconds', 'Model prediction latency')
MODEL_ACCURACY = Gauge('model_accuracy', 'Model prediction accuracy')
MODEL_DRIFT = Gauge('model_drift_score', 'Model drift detection score')
API_ERRORS = Counter('api_errors_total', 'Total number of API errors')
CIRCUIT_BREAKER = Gauge('circuit_breaker_state', 'Circuit breaker state (0=closed, 1=open)')

@dataclass
class ModelMetrics:
    """Container for model performance metrics."""
    accuracy: float
    latency: float
    drift_score: float
    predictions: int
    errors: int
    last_updated: datetime

class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure_time = None
        self.is_open = False

    def record_failure(self):
        """Record a failure and potentially open the circuit."""
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.failure_threshold:
            self.is_open = True
            CIRCUIT_BREAKER.set(1)

    def record_success(self):
        """Record a success and potentially close the circuit."""
        self.failures = 0
        self.is_open = False
        CIRCUIT_BREAKER.set(0)

    def can_execute(self) -> bool:
        """Check if the circuit is closed or can be reset."""
        if not self.is_open:
            return True
        if time.time() - self.last_failure_time > self.reset_timeout:
            self.is_open = False
            CIRCUIT_BREAKER.set(0)
            return True
        return False

class ModelVersionManager:
    """Manages model versions and A/B testing."""
    def __init__(self, experiment_name: str = "trading_bot"):
        self.client = MlflowClient()
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        self.current_model = None
        self.model_versions = {}

    def register_model(self, model: Any, version: str, metrics: Dict[str, float]):
        """Register a new model version."""
        with mlflow.start_run():
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            mlflow.pytorch.log_model(model, f"model_{version}")
            self.model_versions[version] = {
                'model': model,
                'metrics': metrics,
                'timestamp': datetime.now()
            }

    def get_best_model(self) -> Any:
        """Get the best performing model version."""
        if not self.model_versions:
            return None
        return max(self.model_versions.items(), 
                  key=lambda x: x[1]['metrics'].get('accuracy', 0))[1]['model']

class DriftDetector:
    """Detects model drift using statistical methods."""
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.predictions = []
        self.reference_distribution = None

    def update(self, prediction: float):
        """Update the drift detection with a new prediction."""
        self.predictions.append(prediction)
        if len(self.predictions) > self.window_size:
            self.predictions.pop(0)
        
        if len(self.predictions) == self.window_size and self.reference_distribution is None:
            self.reference_distribution = np.array(self.predictions)

    def detect_drift(self) -> float:
        """Calculate drift score using KL divergence."""
        if len(self.predictions) < self.window_size:
            return 0.0
        
        current_distribution = np.array(self.predictions)
        drift_score = self._calculate_kl_divergence(
            self.reference_distribution, 
            current_distribution
        )
        MODEL_DRIFT.set(drift_score)
        return drift_score

    def _calculate_kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Calculate KL divergence between two distributions."""
        p = np.clip(p, 1e-10, 1)
        q = np.clip(q, 1e-10, 1)
        return np.sum(p * np.log(p / q))

class AIOptimizer:
    """Main class for AI optimization and monitoring."""
    def __init__(self, config_path: str = 'config/ai_config.json'):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.circuit_breaker = CircuitBreaker()
        self.model_manager = ModelVersionManager()
        self.drift_detector = DriftDetector()
        self.metrics = ModelMetrics(
            accuracy=0.0,
            latency=0.0,
            drift_score=0.0,
            predictions=0,
            errors=0,
            last_updated=datetime.now()
        )

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration with error handling."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return {}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def predict(self, data: np.ndarray) -> np.ndarray:
        """Make predictions with retry logic and monitoring."""
        if not self.circuit_breaker.can_execute():
            raise Exception("Circuit breaker is open")

        start_time = time.time()
        try:
            model = self.model_manager.get_best_model()
            if model is None:
                raise Exception("No model available")

            with torch.no_grad():
                predictions = model(torch.from_numpy(data).float())
                predictions = predictions.numpy()

            # Update metrics
            latency = time.time() - start_time
            MODEL_LATENCY.observe(latency)
            MODEL_PREDICTIONS.inc()
            
            # Update drift detection
            self.drift_detector.update(predictions.mean())
            drift_score = self.drift_detector.detect_drift()
            
            # Update metrics
            self.metrics.latency = latency
            self.metrics.drift_score = drift_score
            self.metrics.predictions += 1
            self.metrics.last_updated = datetime.now()

            self.circuit_breaker.record_success()
            return predictions

        except Exception as e:
            self.circuit_breaker.record_failure()
            API_ERRORS.inc()
            self.metrics.errors += 1
            self.logger.error(f"Prediction error: {str(e)}")
            raise

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics in JSON format."""
        return {
            'accuracy': self.metrics.accuracy,
            'latency': self.metrics.latency,
            'drift_score': self.metrics.drift_score,
            'predictions': self.metrics.predictions,
            'errors': self.metrics.errors,
            'last_updated': self.metrics.last_updated.isoformat()
        }

    def update_model(self, new_model: Any, metrics: Dict[str, float]):
        """Update the model with versioning."""
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_manager.register_model(new_model, version, metrics)
        self.logger.info(f"Model updated to version {version}")

    def check_health(self) -> Dict[str, Any]:
        """Check the health of the AI system."""
        return {
            'circuit_breaker_state': 'open' if self.circuit_breaker.is_open else 'closed',
            'model_available': self.model_manager.get_best_model() is not None,
            'drift_detected': self.metrics.drift_score > self.config.get('drift_threshold', 0.1),
            'metrics': self.get_metrics()
        } 