import logging
from typing import Dict, List, Any, Optional
import asyncio
import aiohttp
import json
from datetime import datetime
import random
import numpy as np

class LoadBalancer:
    """Verdeelt taken over nodes in het cluster."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Load balancing settings
        self.strategy = config.get('strategy', 'round_robin')
        self.health_check_interval = config.get('health_check_interval', 30)
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 5)
        
        # Initialize state
        self.nodes = {}
        self.current_index = 0
        self.is_running = False
        
    async def start(self):
        """Start de load balancer."""
        try:
            self.is_running = True
            
            # Start health check
            asyncio.create_task(self._health_check())
            
            self.logger.info("Load balancer started")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting load balancer: {str(e)}")
            return False
            
    async def stop(self):
        """Stop de load balancer."""
        try:
            self.is_running = False
            self.logger.info("Load balancer stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping load balancer: {str(e)}")
            return False
            
    def add_node(self, node_id: str, node_info: Dict):
        """Voeg een node toe aan de load balancer."""
        try:
            self.nodes[node_id] = {
                'info': node_info,
                'status': 'active',
                'last_health_check': datetime.now(),
                'metrics': {
                    'cpu_usage': 0,
                    'memory_usage': 0,
                    'active_tasks': 0,
                    'total_tasks': 0,
                    'success_rate': 1.0
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error adding node: {str(e)}")
            raise
            
    def remove_node(self, node_id: str):
        """Verwijder een node uit de load balancer."""
        try:
            if node_id in self.nodes:
                del self.nodes[node_id]
                
        except Exception as e:
            self.logger.error(f"Error removing node: {str(e)}")
            raise
            
    def get_next_node(self) -> Optional[str]:
        """Get de volgende node voor taak uitvoering."""
        try:
            if not self.nodes:
                return None
                
            # Filter active nodes
            active_nodes = {
                node_id: node
                for node_id, node in self.nodes.items()
                if node['status'] == 'active'
            }
            
            if not active_nodes:
                return None
                
            # Select node based on strategy
            if self.strategy == 'round_robin':
                return self._round_robin(active_nodes)
            elif self.strategy == 'least_loaded':
                return self._least_loaded(active_nodes)
            elif self.strategy == 'weighted_random':
                return self._weighted_random(active_nodes)
            else:
                return self._round_robin(active_nodes)
                
        except Exception as e:
            self.logger.error(f"Error getting next node: {str(e)}")
            return None
            
    def _round_robin(self, nodes: Dict) -> str:
        """Round-robin node selection."""
        try:
            node_ids = list(nodes.keys())
            if not node_ids:
                return None
                
            node_id = node_ids[self.current_index]
            self.current_index = (self.current_index + 1) % len(node_ids)
            
            return node_id
            
        except Exception as e:
            self.logger.error(f"Error in round-robin selection: {str(e)}")
            return None
            
    def _least_loaded(self, nodes: Dict) -> str:
        """Least-loaded node selection."""
        try:
            if not nodes:
                return None
                
            # Calculate load score for each node
            load_scores = {}
            for node_id, node in nodes.items():
                metrics = node['metrics']
                load_score = (
                    0.4 * metrics['cpu_usage'] +
                    0.4 * metrics['memory_usage'] +
                    0.2 * metrics['active_tasks']
                )
                load_scores[node_id] = load_score
                
            # Select node with lowest load
            return min(load_scores.items(), key=lambda x: x[1])[0]
            
        except Exception as e:
            self.logger.error(f"Error in least-loaded selection: {str(e)}")
            return None
            
    def _weighted_random(self, nodes: Dict) -> str:
        """Weighted random node selection."""
        try:
            if not nodes:
                return None
                
            # Calculate weights based on node metrics
            weights = []
            node_ids = []
            
            for node_id, node in nodes.items():
                metrics = node['metrics']
                weight = (
                    0.3 * (1 - metrics['cpu_usage']) +
                    0.3 * (1 - metrics['memory_usage']) +
                    0.2 * (1 - metrics['active_tasks']) +
                    0.2 * metrics['success_rate']
                )
                weights.append(weight)
                node_ids.append(node_id)
                
            # Normalize weights
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            # Select node based on weights
            return np.random.choice(node_ids, p=weights)
            
        except Exception as e:
            self.logger.error(f"Error in weighted random selection: {str(e)}")
            return None
            
    async def _health_check(self):
        """Periodieke health check van nodes."""
        while self.is_running:
            try:
                for node_id, node in list(self.nodes.items()):
                    try:
                        # Check node health
                        async with aiohttp.ClientSession() as session:
                            async with session.get(f"{node['info']['url']}/health") as response:
                                if response.status == 200:
                                    # Update node metrics
                                    metrics = await response.json()
                                    node['metrics'].update(metrics)
                                    node['last_health_check'] = datetime.now()
                                else:
                                    # Mark node as inactive
                                    node['status'] = 'inactive'
                                    
                    except Exception as e:
                        self.logger.warning(f"Error checking node {node_id}: {str(e)}")
                        node['status'] = 'inactive'
                        
            except Exception as e:
                self.logger.error(f"Error in health check: {str(e)}")
                
            await asyncio.sleep(self.health_check_interval)
            
    def update_node_metrics(self, node_id: str, metrics: Dict):
        """Update metrics voor een node."""
        try:
            if node_id in self.nodes:
                self.nodes[node_id]['metrics'].update(metrics)
                
        except Exception as e:
            self.logger.error(f"Error updating node metrics: {str(e)}")
            raise
            
    def get_node_metrics(self, node_id: str) -> Optional[Dict]:
        """Get metrics voor een node."""
        try:
            if node_id in self.nodes:
                return self.nodes[node_id]['metrics']
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting node metrics: {str(e)}")
            return None
            
    def get_all_node_metrics(self) -> Dict[str, Dict]:
        """Get metrics voor alle nodes."""
        try:
            return {
                node_id: node['metrics']
                for node_id, node in self.nodes.items()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting all node metrics: {str(e)}")
            return {} 