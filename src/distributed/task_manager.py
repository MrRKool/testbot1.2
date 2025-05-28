import logging
from typing import Dict, List, Any, Optional
import asyncio
import aiohttp
import json
from datetime import datetime
import uuid
from concurrent.futures import ThreadPoolExecutor

class TaskManager:
    """Beheert taken in het distributed computing cluster."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Task settings
        self.max_workers = config.get('max_workers', 4)
        self.task_timeout = config.get('task_timeout', 300)
        self.retry_attempts = config.get('retry_attempts', 3)
        self.retry_delay = config.get('retry_delay', 5)
        
        # Initialize state
        self.tasks = {}
        self.workers = ThreadPoolExecutor(max_workers=self.max_workers)
        self.is_running = False
        
    async def start(self):
        """Start de task manager."""
        try:
            self.is_running = True
            self.logger.info("Task manager started")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting task manager: {str(e)}")
            return False
            
    async def stop(self):
        """Stop de task manager."""
        try:
            self.is_running = False
            self.workers.shutdown(wait=True)
            self.logger.info("Task manager stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping task manager: {str(e)}")
            return False
            
    async def submit_task(self, task_type: str, task_data: Dict) -> str:
        """Submit een nieuwe taak."""
        try:
            # Generate task ID
            task_id = str(uuid.uuid4())
            
            # Create task
            task = {
                'id': task_id,
                'type': task_type,
                'data': task_data,
                'status': 'pending',
                'created_at': datetime.now(),
                'attempts': 0,
                'result': None,
                'error': None
            }
            
            # Store task
            self.tasks[task_id] = task
            
            # Submit task to worker
            self.workers.submit(self._execute_task, task)
            
            return task_id
            
        except Exception as e:
            self.logger.error(f"Error submitting task: {str(e)}")
            raise
            
    def _execute_task(self, task: Dict):
        """Execute een taak in een worker thread."""
        try:
            # Update task status
            task['status'] = 'running'
            task['attempts'] += 1
            
            # Execute task based on type
            if task['type'] == 'model_training':
                result = self._execute_model_training(task['data'])
            elif task['type'] == 'data_processing':
                result = self._execute_data_processing(task['data'])
            elif task['type'] == 'feature_engineering':
                result = self._execute_feature_engineering(task['data'])
            else:
                raise ValueError(f"Unknown task type: {task['type']}")
                
            # Update task with result
            task['status'] = 'completed'
            task['result'] = result
            task['completed_at'] = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error executing task: {str(e)}")
            
            # Handle retry
            if task['attempts'] < self.retry_attempts:
                task['status'] = 'pending'
                task['error'] = str(e)
                asyncio.sleep(self.retry_delay)
                self.workers.submit(self._execute_task, task)
            else:
                task['status'] = 'failed'
                task['error'] = str(e)
                task['failed_at'] = datetime.now()
                
    def _execute_model_training(self, task_data: Dict) -> Dict:
        """Execute model training task."""
        try:
            # Import here to avoid circular imports
            from src.ai.models.model_pipeline import ModelPipeline
            
            # Initialize model pipeline
            pipeline = ModelPipeline(self.config)
            
            # Train model
            success = pipeline.train_model(
                task_data['model_type'],
                task_data['X'],
                task_data['y']
            )
            
            return {
                'success': success,
                'metrics': pipeline.model_metrics.get(task_data['model_type'], {})
            }
            
        except Exception as e:
            self.logger.error(f"Error executing model training: {str(e)}")
            raise
            
    def _execute_data_processing(self, task_data: Dict) -> Dict:
        """Execute data processing task."""
        try:
            # Import here to avoid circular imports
            from src.ai.features.feature_engineering import FeatureEngineering
            
            # Initialize feature engineering
            fe = FeatureEngineering(self.config)
            
            # Process data
            processed_data = fe.preprocess_features(task_data['data'])
            
            return {
                'success': True,
                'processed_data': processed_data
            }
            
        except Exception as e:
            self.logger.error(f"Error executing data processing: {str(e)}")
            raise
            
    def _execute_feature_engineering(self, task_data: Dict) -> Dict:
        """Execute feature engineering task."""
        try:
            # Import here to avoid circular imports
            from src.ai.features.feature_engineering import FeatureEngineering
            
            # Initialize feature engineering
            fe = FeatureEngineering(self.config)
            
            # Create features
            features = fe.create_technical_indicators(task_data['data'])
            
            return {
                'success': True,
                'features': features
            }
            
        except Exception as e:
            self.logger.error(f"Error executing feature engineering: {str(e)}")
            raise
            
    def get_task_status(self, task_id: str) -> Dict:
        """Get de status van een taak."""
        try:
            if task_id not in self.tasks:
                raise ValueError(f"Task not found: {task_id}")
                
            return self.tasks[task_id]
            
        except Exception as e:
            self.logger.error(f"Error getting task status: {str(e)}")
            raise
            
    def get_all_tasks(self) -> List[Dict]:
        """Get alle taken."""
        try:
            return list(self.tasks.values())
            
        except Exception as e:
            self.logger.error(f"Error getting all tasks: {str(e)}")
            raise
            
    def cancel_task(self, task_id: str) -> bool:
        """Cancel een taak."""
        try:
            if task_id not in self.tasks:
                raise ValueError(f"Task not found: {task_id}")
                
            task = self.tasks[task_id]
            if task['status'] in ['pending', 'running']:
                task['status'] = 'cancelled'
                task['cancelled_at'] = datetime.now()
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Error cancelling task: {str(e)}")
            raise
            
    def clear_completed_tasks(self, max_age_days: int = 7) -> int:
        """Clear voltooide taken ouder dan max_age_days."""
        try:
            now = datetime.now()
            cleared = 0
            
            for task_id, task in list(self.tasks.items()):
                if task['status'] in ['completed', 'failed', 'cancelled']:
                    age = (now - task.get('completed_at', task.get('failed_at', task.get('cancelled_at', now)))).days
                    if age > max_age_days:
                        del self.tasks[task_id]
                        cleared += 1
                        
            return cleared
            
        except Exception as e:
            self.logger.error(f"Error clearing completed tasks: {str(e)}")
            raise 