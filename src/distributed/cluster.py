import logging
from typing import Dict, List, Any, Optional
import asyncio
import aiohttp
import json
from datetime import datetime
import psutil
import socket
import uuid

class ClusterManager:
    """Beheert een cluster van nodes voor distributed computing."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Cluster settings
        self.node_id = str(uuid.uuid4())
        self.role = config.get('role', 'worker')
        self.master_url = config.get('master_url')
        self.worker_urls = config.get('worker_urls', [])
        
        # Performance settings
        self.max_workers = config.get('max_workers', psutil.cpu_count())
        self.task_timeout = config.get('task_timeout', 300)
        self.heartbeat_interval = config.get('heartbeat_interval', 30)
        
        # Initialize state
        self.nodes = {}
        self.tasks = {}
        self.is_running = False
        
    async def start(self):
        """Start de cluster manager."""
        try:
            self.is_running = True
            
            if self.role == 'master':
                await self._start_master()
            else:
                await self._start_worker()
                
            # Start heartbeat
            asyncio.create_task(self._heartbeat())
            
            self.logger.info(f"Cluster manager started as {self.role}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting cluster manager: {str(e)}")
            return False
            
    async def stop(self):
        """Stop de cluster manager."""
        try:
            self.is_running = False
            
            if self.role == 'master':
                await self._stop_master()
            else:
                await self._stop_worker()
                
            self.logger.info("Cluster manager stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping cluster manager: {str(e)}")
            return False
            
    async def _start_master(self):
        """Start de master node."""
        try:
            # Initialize master state
            self.nodes = {
                self.node_id: {
                    'role': 'master',
                    'status': 'active',
                    'last_heartbeat': datetime.now(),
                    'resources': self._get_node_resources()
                }
            }
            
            # Start HTTP server
            self.server = await asyncio.start_server(
                self._handle_request,
                '0.0.0.0',
                int(self.master_url.split(':')[-1])
            )
            
            async with self.server:
                await self.server.serve_forever()
                
        except Exception as e:
            self.logger.error(f"Error starting master: {str(e)}")
            raise
            
    async def _start_worker(self):
        """Start de worker node."""
        try:
            # Register with master
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.master_url}/register",
                    json={
                        'node_id': self.node_id,
                        'role': 'worker',
                        'resources': self._get_node_resources()
                    }
                ) as response:
                    if response.status != 200:
                        raise Exception("Failed to register with master")
                        
        except Exception as e:
            self.logger.error(f"Error starting worker: {str(e)}")
            raise
            
    async def _stop_master(self):
        """Stop de master node."""
        try:
            # Stop HTTP server
            if hasattr(self, 'server'):
                self.server.close()
                await self.server.wait_closed()
                
            # Notify workers
            for worker_url in self.worker_urls:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(f"{worker_url}/shutdown") as response:
                            if response.status != 200:
                                self.logger.warning(f"Failed to notify worker {worker_url}")
                except Exception as e:
                    self.logger.warning(f"Error notifying worker {worker_url}: {str(e)}")
                    
        except Exception as e:
            self.logger.error(f"Error stopping master: {str(e)}")
            raise
            
    async def _stop_worker(self):
        """Stop de worker node."""
        try:
            # Unregister from master
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.master_url}/unregister",
                    json={'node_id': self.node_id}
                ) as response:
                    if response.status != 200:
                        self.logger.warning("Failed to unregister from master")
                        
        except Exception as e:
            self.logger.error(f"Error stopping worker: {str(e)}")
            raise
            
    async def _heartbeat(self):
        """Send periodic heartbeat to master."""
        while self.is_running:
            try:
                if self.role == 'worker':
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            f"{self.master_url}/heartbeat",
                            json={
                                'node_id': self.node_id,
                                'resources': self._get_node_resources()
                            }
                        ) as response:
                            if response.status != 200:
                                self.logger.warning("Failed to send heartbeat")
                                
            except Exception as e:
                self.logger.error(f"Error sending heartbeat: {str(e)}")
                
            await asyncio.sleep(self.heartbeat_interval)
            
    async def _handle_request(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle incoming requests."""
        try:
            # Read request
            data = await reader.read(1024)
            request = json.loads(data.decode())
            
            # Handle request
            if request['type'] == 'register':
                response = await self._handle_register(request)
            elif request['type'] == 'unregister':
                response = await self._handle_unregister(request)
            elif request['type'] == 'heartbeat':
                response = await self._handle_heartbeat(request)
            elif request['type'] == 'task':
                response = await self._handle_task(request)
            elif request['type'] == 'shutdown':
                response = await self._handle_shutdown(request)
            else:
                response = {'status': 'error', 'message': 'Unknown request type'}
                
            # Send response
            writer.write(json.dumps(response).encode())
            await writer.drain()
            writer.close()
            
        except Exception as e:
            self.logger.error(f"Error handling request: {str(e)}")
            writer.write(json.dumps({
                'status': 'error',
                'message': str(e)
            }).encode())
            await writer.drain()
            writer.close()
            
    async def _handle_register(self, request: Dict) -> Dict:
        """Handle node registration."""
        try:
            node_id = request['node_id']
            self.nodes[node_id] = {
                'role': request['role'],
                'status': 'active',
                'last_heartbeat': datetime.now(),
                'resources': request['resources']
            }
            
            return {'status': 'success'}
            
        except Exception as e:
            self.logger.error(f"Error handling register: {str(e)}")
            return {'status': 'error', 'message': str(e)}
            
    async def _handle_unregister(self, request: Dict) -> Dict:
        """Handle node unregistration."""
        try:
            node_id = request['node_id']
            if node_id in self.nodes:
                del self.nodes[node_id]
                
            return {'status': 'success'}
            
        except Exception as e:
            self.logger.error(f"Error handling unregister: {str(e)}")
            return {'status': 'error', 'message': str(e)}
            
    async def _handle_heartbeat(self, request: Dict) -> Dict:
        """Handle node heartbeat."""
        try:
            node_id = request['node_id']
            if node_id in self.nodes:
                self.nodes[node_id].update({
                    'last_heartbeat': datetime.now(),
                    'resources': request['resources']
                })
                
            return {'status': 'success'}
            
        except Exception as e:
            self.logger.error(f"Error handling heartbeat: {str(e)}")
            return {'status': 'error', 'message': str(e)}
            
    async def _handle_task(self, request: Dict) -> Dict:
        """Handle task execution."""
        try:
            task_id = request['task_id']
            task = request['task']
            
            # Execute task
            result = await self._execute_task(task)
            
            # Store result
            self.tasks[task_id] = {
                'status': 'completed',
                'result': result
            }
            
            return {'status': 'success', 'result': result}
            
        except Exception as e:
            self.logger.error(f"Error handling task: {str(e)}")
            return {'status': 'error', 'message': str(e)}
            
    async def _handle_shutdown(self, request: Dict) -> Dict:
        """Handle shutdown request."""
        try:
            await self.stop()
            return {'status': 'success'}
            
        except Exception as e:
            self.logger.error(f"Error handling shutdown: {str(e)}")
            return {'status': 'error', 'message': str(e)}
            
    async def _execute_task(self, task: Dict) -> Any:
        """Execute a task."""
        try:
            # Get task details
            task_type = task['type']
            task_data = task['data']
            
            # Execute task based on type
            if task_type == 'model_training':
                result = await self._execute_model_training(task_data)
            elif task_type == 'data_processing':
                result = await self._execute_data_processing(task_data)
            elif task_type == 'feature_engineering':
                result = await self._execute_feature_engineering(task_data)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing task: {str(e)}")
            raise
            
    async def _execute_model_training(self, task_data: Dict) -> Dict:
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
            
    async def _execute_data_processing(self, task_data: Dict) -> Dict:
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
            
    async def _execute_feature_engineering(self, task_data: Dict) -> Dict:
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
            
    def _get_node_resources(self) -> Dict:
        """Get node resource information."""
        try:
            return {
                'cpu_count': psutil.cpu_count(),
                'cpu_percent': psutil.cpu_percent(),
                'memory_total': psutil.virtual_memory().total,
                'memory_available': psutil.virtual_memory().available,
                'disk_total': psutil.disk_usage('/').total,
                'disk_free': psutil.disk_usage('/').free,
                'hostname': socket.gethostname()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting node resources: {str(e)}")
            return {} 