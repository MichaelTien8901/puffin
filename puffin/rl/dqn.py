"""
Deep Q-Network (DQN) and Double DQN implementations for reinforcement learning trading.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Optional
from collections import deque
import random


class DQNetwork(nn.Module):
    """
    Deep Q-Network architecture.

    Parameters
    ----------
    state_dim : int
        Dimension of state/observation space
    action_dim : int
        Number of discrete actions
    hidden_dims : List[int], default=[128, 64]
        Hidden layer dimensions
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = None
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers = []
        prev_dim = state_dim

        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.

        Parameters
        ----------
        x : torch.Tensor
            State tensor of shape (batch_size, state_dim)

        Returns
        -------
        torch.Tensor
            Q-values of shape (batch_size, action_dim)
        """
        return self.network(x)


class ReplayBuffer:
    """
    Experience replay buffer for DQN.

    Parameters
    ----------
    capacity : int
        Maximum buffer size
    """

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Store a transition in the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Sample a batch of transitions.

        Returns
        -------
        Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network agent.

    Parameters
    ----------
    state_dim : int
        Dimension of state space
    action_dim : int
        Number of discrete actions
    lr : float, default=1e-4
        Learning rate
    gamma : float, default=0.99
        Discount factor
    buffer_size : int, default=10000
        Replay buffer capacity
    batch_size : int, default=64
        Training batch size
    target_update : int, default=100
        Steps between target network updates
    hidden_dims : List[int], optional
        Hidden layer dimensions
    device : str, optional
        Device to use ('cuda' or 'cpu')
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-4,
        gamma: float = 0.99,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update: int = 100,
        hidden_dims: Optional[List[int]] = None,
        device: Optional[str] = None
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update

        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Networks
        self.online_network = DQNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network = DQNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Training step counter
        self.step_count = 0

    def choose_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Choose action using epsilon-greedy policy.

        Parameters
        ----------
        state : np.ndarray
            Current state
        epsilon : float, default=0.0
            Exploration rate

        Returns
        -------
        int
            Selected action
        """
        if np.random.random() < epsilon:
            return np.random.randint(0, self.action_dim)

        # Greedy action
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.online_network(state_tensor)
            return int(q_values.argmax(dim=1).item())

    def update(self) -> Optional[float]:
        """
        Perform one gradient step on a batch from replay buffer.

        Returns
        -------
        Optional[float]
            Loss value if update was performed, None otherwise
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        current_q_values = self.online_network(states).gather(1, actions.unsqueeze(1))

        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(dim=1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_network.load_state_dict(self.online_network.state_dict())

        return loss.item()

    def train(
        self,
        env,
        episodes: int = 500,
        max_steps: Optional[int] = None,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        verbose: bool = False
    ) -> List[float]:
        """
        Train the DQN agent.

        Parameters
        ----------
        env : gymnasium.Env
            Training environment
        episodes : int, default=500
            Number of training episodes
        max_steps : Optional[int], default=None
            Maximum steps per episode
        epsilon_start : float, default=1.0
            Initial exploration rate
        epsilon_end : float, default=0.01
            Minimum exploration rate
        epsilon_decay : float, default=0.995
            Epsilon decay factor per episode
        verbose : bool, default=False
            Whether to print progress

        Returns
        -------
        List[float]
            List of total rewards per episode
        """
        episode_rewards = []
        epsilon = epsilon_start

        for episode in range(episodes):
            state, info = env.reset()
            if isinstance(state, tuple):
                state = state[0]
            state = np.array(state, dtype=np.float32)

            total_reward = 0.0
            done = False
            step = 0

            while not done:
                # Choose and take action
                action = self.choose_action(state, epsilon)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                if isinstance(next_state, tuple):
                    next_state = next_state[0]
                next_state = np.array(next_state, dtype=np.float32)

                # Store transition
                self.replay_buffer.store(state, action, reward, next_state, done)

                # Update network
                self.update()

                state = next_state
                total_reward += reward
                step += 1

                if max_steps is not None and step >= max_steps:
                    break

            episode_rewards.append(total_reward)

            # Decay epsilon
            epsilon = max(epsilon_end, epsilon * epsilon_decay)

            # Verbose output
            if verbose and (episode + 1) % 50 == 0:
                avg_reward = np.mean(episode_rewards[-50:])
                print(f"Episode {episode + 1}/{episodes}, "
                      f"Avg Reward (last 50): {avg_reward:.2f}, "
                      f"Epsilon: {epsilon:.3f}, "
                      f"Buffer: {len(self.replay_buffer)}")

        return episode_rewards

    def save(self, path: str) -> None:
        """Save agent networks to file."""
        torch.save({
            'online_network': self.online_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step_count': self.step_count
        }, path)

    def load(self, path: str) -> None:
        """Load agent networks from file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.online_network.load_state_dict(checkpoint['online_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.step_count = checkpoint['step_count']


class DDQNAgent(DQNAgent):
    """
    Double Deep Q-Network agent.

    Uses online network to select actions and target network to evaluate them,
    reducing overestimation bias.

    Parameters are the same as DQNAgent.
    """

    def update(self) -> Optional[float]:
        """
        Perform one gradient step using Double Q-learning update.

        Returns
        -------
        Optional[float]
            Loss value if update was performed, None otherwise
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        current_q_values = self.online_network(states).gather(1, actions.unsqueeze(1))

        # Double Q-learning target
        with torch.no_grad():
            # Online network selects action
            next_actions = self.online_network(next_states).argmax(dim=1)
            # Target network evaluates action
            next_q_values = self.target_network(next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze()
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_network.load_state_dict(self.online_network.state_dict())

        return loss.item()
