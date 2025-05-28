import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from collections import deque
import random
import gc
from typing import List, Tuple, Dict, Any
import logging
from dataclasses import dataclass
import time
from torch.utils.data import DataLoader, TensorDataset
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import os
from scipy import stats
from scipy.fft import fft
import pywt
import pandas as pd

@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    losses: List[float] = None
    rewards: List[float] = None
    epsilons: List[float] = None
    q_values: List[float] = None
    gradients: List[float] = None
    memory_usage: List[float] = None
    training_time: float = 0.0
    
    def __post_init__(self):
        self.losses = []
        self.rewards = []
        self.epsilons = []
        self.q_values = []
        self.gradients = []
        self.memory_usage = []

class PrioritizedReplayBuffer:
    """Optimized replay buffer with prioritized sampling."""
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self._max_priority = 1.0
        self.position = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Push experience to buffer with priority."""
        max_priority = max(self.priorities) if self.priorities else self._max_priority
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(max_priority)
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
            self.priorities[self.position] = max_priority
            self.position = (self.position + 1) % self.capacity
            
    def sample(self, batch_size: int) -> Tuple:
        """Sample batch with importance sampling weights."""
        if len(self.buffer) < batch_size:
            return None
            
        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]
        
        # Convert to tensors efficiently
        states = torch.FloatTensor(np.array([e[0] for e in experiences])).to(self.device)
        actions = torch.LongTensor(np.array([e[1] for e in experiences])).to(self.device)
        rewards = torch.FloatTensor(np.array([e[2] for e in experiences])).to(self.device)
        next_states = torch.FloatTensor(np.array([e[3] for e in experiences])).to(self.device)
        dones = torch.FloatTensor(np.array([e[4] for e in experiences])).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        return states, actions, rewards, next_states, dones, weights, indices
        
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities for experiences."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self._max_priority = max(self._max_priority, priority)
            
    def clear(self):
        """Clear buffer and free memory."""
        self.buffer.clear()
        self.priorities.clear()
        self.position = 0
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class FourierFeatures(nn.Module):
    """Fourier feature mapping for better function approximation."""
    def __init__(self, input_dim: int, output_dim: int, sigma: float = 1.0):
        super().__init__()
        self.B = torch.randn(input_dim, output_dim // 2) * sigma
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = torch.matmul(x, self.B)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)

class WaveletFeatures(nn.Module):
    """Wavelet transform features for signal analysis."""
    def __init__(self, input_dim: int, wavelet: str = 'db4', level: int = 3):
        super().__init__()
        self.wavelet = wavelet
        self.level = level
        self.output_dim = input_dim * (2 ** level - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coeffs = pywt.wavedec(x.cpu().numpy(), self.wavelet, level=self.level)
        features = np.concatenate([c.flatten() for c in coeffs])
        return torch.FloatTensor(features).to(x.device)

class KalmanFilter(nn.Module):
    """Kalman filter for noise reduction and state estimation."""
    def __init__(self, state_dim: int, measurement_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        
        # State transition matrix
        self.F = nn.Parameter(torch.eye(state_dim))
        # Measurement matrix
        self.H = nn.Parameter(torch.randn(measurement_dim, state_dim))
        # Process noise covariance
        self.Q = nn.Parameter(torch.eye(state_dim))
        # Measurement noise covariance
        self.R = nn.Parameter(torch.eye(measurement_dim))
        
        self.state = None
        self.covariance = None

    def forward(self, measurement: torch.Tensor) -> torch.Tensor:
        if self.state is None:
            self.state = torch.zeros(self.state_dim).to(measurement.device)
            self.covariance = torch.eye(self.state_dim).to(measurement.device)
        
        # Predict
        predicted_state = torch.matmul(self.F, self.state)
        predicted_covariance = torch.matmul(torch.matmul(self.F, self.covariance), self.F.t()) + self.Q
        
        # Update
        kalman_gain = torch.matmul(
            torch.matmul(predicted_covariance, self.H.t()),
            torch.inverse(torch.matmul(torch.matmul(self.H, predicted_covariance), self.H.t()) + self.R)
        )
        
        self.state = predicted_state + torch.matmul(kalman_gain, measurement - torch.matmul(self.H, predicted_state))
        self.covariance = predicted_covariance - torch.matmul(
            torch.matmul(kalman_gain, self.H), predicted_covariance
        )
        
        return self.state

class DQN(nn.Module):
    """Enhanced DQN network with advanced mathematical features."""
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super(DQN, self).__init__()
        
        # Fourier feature mapping
        self.fourier = FourierFeatures(state_size, hidden_size)
        
        # Wavelet features
        self.wavelet = WaveletFeatures(state_size)
        
        # Kalman filter for state estimation
        self.kalman = KalmanFilter(state_size, state_size)
        
        # Feature extraction layers with batch normalization
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_size + self.wavelet.output_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Dueling architecture
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )
        
        # Initialize weights using Kaiming initialization
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with enhanced feature extraction."""
        # Apply Kalman filter for noise reduction
        x = self.kalman(x)
        
        # Extract Fourier features
        fourier_features = self.fourier(x)
        
        # Extract wavelet features
        wavelet_features = self.wavelet(x)
        
        # Combine features
        combined_features = torch.cat([fourier_features, wavelet_features], dim=-1)
        
        # Process through feature extractor
        features = self.feature_extractor(combined_features)
        
        # Dueling architecture
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage streams
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

class TradingEnvironment:
    def __init__(self, data: pd.DataFrame, indicators: List[str], initial_balance: float = 10000, transaction_fee: float = 0.001):
        """Initialize trading environment."""
        self.data = data
        self.indicators = indicators
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.reset()
        
    def reset(self):
        """Reset environment to initial state."""
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.total_value = self.balance
        self.done = False
        return self._get_state()
        
    def step(self, action: int):
        """Take action in environment."""
        if self.done:
            return self._get_state(), 0, True, {}
            
        # Get current price and indicators
        current_data = self.data.iloc[self.current_step]
        current_price = current_data['close']
        
        # Execute action
        reward = 0
        if action == 1:  # Buy
            if self.position <= 0:
                # Calculate cost including fees
                cost = current_price * (1 + self.transaction_fee)
                if cost <= self.balance:
                    self.position = 1
                    self.balance -= cost
                    reward = 0.1  # Small positive reward for successful buy
        elif action == 2:  # Sell
            if self.position >= 0:
                # Calculate proceeds including fees
                proceeds = current_price * (1 - self.transaction_fee)
                self.position = -1
                self.balance += proceeds
                reward = 0.1  # Small positive reward for successful sell
                
        # Calculate total value
        self.total_value = self.balance + (self.position * current_price)
        
        # Calculate reward based on value change and indicators
        if self.current_step > 0:
            prev_data = self.data.iloc[self.current_step - 1]
            price_change = (current_price - prev_data['close']) / prev_data['close']
            
            # Base reward on price change and position
            reward += price_change * self.position
            
            # Add indicator-based rewards
            if 'RSI' in current_data:
                if current_data['RSI'] < 30 and self.position > 0:
                    reward += 0.05  # Reward for buying oversold
                elif current_data['RSI'] > 70 and self.position < 0:
                    reward += 0.05  # Reward for selling overbought
                    
            if 'MACD' in current_data:
                if current_data['MACD'] > 0 and self.position > 0:
                    reward += 0.05  # Reward for buying in uptrend
                elif current_data['MACD'] < 0 and self.position < 0:
                    reward += 0.05  # Reward for selling in downtrend
                    
            if all(x in current_data for x in ['BB_upper', 'BB_lower']):
                if current_price < current_data['BB_lower'] and self.position > 0:
                    reward += 0.05  # Reward for buying at lower band
                elif current_price > current_data['BB_upper'] and self.position < 0:
                    reward += 0.05  # Reward for selling at upper band
                    
            if all(x in current_data for x in ['KVO', 'KVO_signal']):
                if current_data['KVO'] > current_data['KVO_signal'] and self.position > 0:
                    reward += 0.05  # Reward for buying on KVO crossover
                elif current_data['KVO'] < current_data['KVO_signal'] and self.position < 0:
                    reward += 0.05  # Reward for selling on KVO crossover
                    
            if all(x in current_data for x in ['VI_plus', 'VI_minus']):
                if current_data['VI_plus'] > current_data['VI_minus'] and self.position > 0:
                    reward += 0.05  # Reward for buying on VI crossover
                elif current_data['VI_plus'] < current_data['VI_minus'] and self.position < 0:
                    reward += 0.05  # Reward for selling on VI crossover
                    
            if 'CMF' in current_data:
                if current_data['CMF'] > 0.2 and self.position > 0:
                    reward += 0.05  # Reward for buying on strong money flow
                elif current_data['CMF'] < -0.2 and self.position < 0:
                    reward += 0.05  # Reward for selling on weak money flow
                    
            if all(x in current_data for x in ['DC_upper', 'DC_lower']):
                if current_price < current_data['DC_lower'] and self.position > 0:
                    reward += 0.05  # Reward for buying at lower channel
                elif current_price > current_data['DC_upper'] and self.position < 0:
                    reward += 0.05  # Reward for selling at upper channel
                    
            if 'DPO' in current_data:
                if current_data['DPO'] > 0 and self.position > 0:
                    reward += 0.05  # Reward for buying on positive DPO
                elif current_data['DPO'] < 0 and self.position < 0:
                    reward += 0.05  # Reward for selling on negative DPO
                    
            if 'WILLR' in current_data:
                if current_data['WILLR'] < -80 and self.position > 0:
                    reward += 0.05  # Reward for buying oversold
                elif current_data['WILLR'] > -20 and self.position < 0:
                    reward += 0.05  # Reward for selling overbought
                    
            if 'STC' in current_data:
                if current_data['STC'] > 75 and self.position > 0:
                    reward += 0.05  # Reward for buying on strong trend
                elif current_data['STC'] < 25 and self.position < 0:
                    reward += 0.05  # Reward for selling on weak trend
                    
        # Move to next step
        self.current_step += 1
        self.done = self.current_step >= len(self.data) - 1
        
        return self._get_state(), reward, self.done, {
            'total_value': self.total_value,
            'balance': self.balance,
            'position': self.position
        }
        
    def _get_state(self):
        """Get current state representation."""
        if self.current_step >= len(self.data):
            return np.zeros(len(self.indicators) + 2)  # +2 for position and balance
            
        # Get current data point
        current_data = self.data.iloc[self.current_step]
        
        # Create state array
        state = []
        
        # Add indicator values
        for indicator in self.indicators:
            if indicator == 'RSI':
                state.append(current_data['RSI'] / 100)  # Normalize to [0, 1]
            elif indicator == 'MACD':
                state.append(current_data['MACD'])
            elif indicator == 'BB':
                price = current_data['close']
                bb_upper = current_data['BB_upper']
                bb_lower = current_data['BB_lower']
                bb_middle = current_data['BB_middle']
                # Normalize price position within bands
                state.append((price - bb_lower) / (bb_upper - bb_lower))
                state.append((bb_middle - bb_lower) / (bb_upper - bb_lower))
            elif indicator == 'KVO':
                state.append(current_data['KVO'])
                state.append(current_data['KVO_signal'])
            elif indicator == 'VI':
                state.append(current_data['VI_plus'])
                state.append(current_data['VI_minus'])
            elif indicator == 'CMF':
                state.append(current_data['CMF'])
            elif indicator == 'DC':
                price = current_data['close']
                dc_upper = current_data['DC_upper']
                dc_lower = current_data['DC_lower']
                dc_middle = current_data['DC_middle']
                # Normalize price position within channels
                state.append((price - dc_lower) / (dc_upper - dc_lower))
                state.append((dc_middle - dc_lower) / (dc_upper - dc_lower))
            elif indicator == 'DPO':
                state.append(current_data['DPO'])
            elif indicator == 'WILLR':
                state.append(current_data['WILLR'] / 100)  # Normalize to [0, 1]
            elif indicator == 'STC':
                state.append(current_data['STC'] / 100)  # Normalize to [0, 1]
                
        # Add position and balance
        state.append(self.position)
        state.append(self.balance / self.initial_balance)  # Normalize balance
        
        return np.array(state, dtype=np.float32)

class RLAgent:
    """Enhanced RL agent with advanced optimization methods."""
    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 0.001, gamma: float = 0.99, epsilon: float = 0.1):
        """Initialize RL agent."""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Initialize neural network
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        # Initialize replay buffer
        self.replay_buffer = []
        self.max_buffer_size = 10000
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.target_model.to(self.device)
        
    def _build_model(self):
        """Build neural network model."""
        model = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, self.action_dim)
        )
        return model
        
    def update_target_model(self):
        """Update target model with current model weights."""
        self.target_model.load_state_dict(self.model.state_dict())
        
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        if len(self.replay_buffer) >= self.max_buffer_size:
            self.replay_buffer.pop(0)
        self.replay_buffer.append((state, action, reward, next_state, done))
        
    def act(self, state, training: bool = True):
        """Choose action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
            
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            return q_values.argmax().item()
            
    def train(self, batch_size: int):
        """Train the model on a batch of experiences."""
        if len(self.replay_buffer) < batch_size:
            return
            
        # Sample batch
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Get current Q values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Get next Q values from target model
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
        # Compute loss and update
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        
    def save(self, path: str):
        """Save model to file."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
        
    def load(self, path: str):
        """Load model from file."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    def train_episode(self, env: TradingEnvironment, max_steps: int = 1000) -> float:
        """Train for one episode with improved efficiency."""
        state = env.reset()
        total_reward = 0
        start_time = time.time()
        
        for step in range(max_steps):
            action = self.act(state)
            next_state, reward, done, info = env.step(action)
            
            self.store_experience(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            # Train on batch
            if len(self.replay_buffer) >= batch_size:
                loss = self.train(batch_size)
                
            if done:
                break
                
        # Update metrics
        self.metrics.rewards.append(total_reward)
        self.metrics.training_time += time.time() - start_time
        
        # Update exploration rate
        self.epsilon = max(0.01, self.epsilon * 0.995)
        self.metrics.epsilons.append(self.epsilon)
        
        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return total_reward
        
    def evaluate(self, env: TradingEnvironment, episodes: int = 10) -> Dict[str, float]:
        """Evaluate agent performance."""
        total_rewards = []
        total_profits = []
        max_drawdowns = []
        
        for _ in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = self.act(state, training=False)
                next_state, reward, done, info = env.step(action)
                state = next_state
                episode_reward += reward
                
            total_rewards.append(episode_reward)
            total_profits.append(env.total_value - env.initial_balance)
            max_drawdowns.append(env.total_value / env.initial_balance - 1)
            
        return {
            'mean_reward': np.mean(total_rewards),
            'mean_profit': np.mean(total_profits),
            'mean_drawdown': np.mean(max_drawdowns),
            'std_reward': np.std(total_rewards),
            'std_profit': np.std(total_profits),
            'std_drawdown': np.std(max_drawdowns)
        } 