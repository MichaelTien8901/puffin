---
layout: default
title: "Part 21: Deep Reinforcement Learning"
nav_order: 22
permalink: /21-deep-rl/
---

# Deep Reinforcement Learning for Trading

Deep Reinforcement Learning (RL) offers a powerful framework for developing adaptive trading strategies that learn from market interactions. This chapter covers the implementation of RL agents for algorithmic trading.

## Table of Contents
- [Introduction to Reinforcement Learning](#introduction-to-reinforcement-learning)
- [Tabular Q-Learning](#tabular-q-learning)
- [Deep Q-Networks (DQN)](#deep-q-networks-dqn)
- [Custom Trading Environment](#custom-trading-environment)
- [Proximal Policy Optimization (PPO)](#proximal-policy-optimization-ppo)
- [Agent Evaluation](#agent-evaluation)
- [Best Practices](#best-practices)

## Introduction to Reinforcement Learning

Reinforcement Learning is a machine learning paradigm where an agent learns to make decisions by interacting with an environment. The agent receives rewards or penalties based on its actions and learns to maximize cumulative rewards.

### Key Concepts

**State (s)**: Current market condition (prices, indicators, portfolio state)
**Action (a)**: Trading decision (buy, sell, hold)
**Reward (r)**: Immediate feedback (profit/loss, risk-adjusted return)
**Policy (π)**: Strategy mapping states to actions
**Value Function (Q)**: Expected cumulative reward for state-action pairs

### Trading as RL Problem

In algorithmic trading, we can formulate the problem as:
- **State**: Market features + portfolio state (position, cash, P&L)
- **Action**: Trading decisions (discrete: buy/hold/sell, or continuous: position size)
- **Reward**: Returns, Sharpe ratio, or risk-adjusted metrics
- **Goal**: Learn policy that maximizes long-term portfolio returns

## Tabular Q-Learning

Q-learning is a foundational RL algorithm that learns action-value functions through temporal difference learning.

### Algorithm Overview

Q-learning updates the Q-table using the Bellman equation:

```
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
```

Where:
- α (alpha): Learning rate
- γ (gamma): Discount factor
- r: Immediate reward
- s': Next state

### Implementation

```python
from puffin.rl import QLearningAgent, discretize_state
import numpy as np
import gymnasium as gym

# Create simple discrete environment
env = gym.make('FrozenLake-v1', is_slippery=False)

# Initialize Q-learning agent
agent = QLearningAgent(
    n_states=16,           # Number of discrete states
    n_actions=4,           # Number of actions
    lr=0.1,                # Learning rate
    gamma=0.99,            # Discount factor
    epsilon=1.0,           # Initial exploration rate
    epsilon_decay=0.995    # Decay factor
)

# Train the agent
rewards = agent.train(env, episodes=1000, verbose=True)

# Get learned policy
policy = agent.get_policy()
print("Learned policy:", policy)

# Save the Q-table
agent.save("q_table.npy")
```

### State Discretization

For continuous state spaces, we need to discretize observations:

```python
from puffin.rl import discretize_state
import numpy as np

# Define bins for each dimension
obs = np.array([0.52, 1.23, -0.45])
bins = [
    np.linspace(0, 1, 10),    # 10 bins for dimension 1
    np.linspace(0, 2, 10),    # 10 bins for dimension 2
    np.linspace(-1, 1, 10)    # 10 bins for dimension 3
]

# Convert to discrete state index
state_index = discretize_state(obs, bins)
print(f"Discrete state: {state_index}")
```

### Trading Example

```python
from puffin.rl import QLearningAgent, TradingEnvironment, discretize_state
import numpy as np
import pandas as pd

# Load price data
prices = pd.read_csv('data/prices.csv')['close'].values

# Create bins for price discretization
price_bins = [np.percentile(prices, q) for q in range(0, 101, 10)]
bins = [np.array(price_bins)]

# Create wrapper for discretization
class DiscreteWrapper:
    def __init__(self, env, bins):
        self.env = env
        self.bins = bins

    def reset(self):
        obs, info = self.env.reset()
        return discretize_state(obs[:1], self.bins), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        discrete_obs = discretize_state(obs[:1], self.bins)
        return discrete_obs, reward, terminated, truncated, info

# Create trading environment
base_env = TradingEnvironment(prices, discrete_actions=True)
env = DiscreteWrapper(base_env, bins)

# Train Q-learning agent
n_states = (len(bins[0]) - 1)
agent = QLearningAgent(n_states=n_states, n_actions=3)
rewards = agent.train(env, episodes=500)

print(f"Mean reward (last 100): {np.mean(rewards[-100:]):.2f}")
```

## Deep Q-Networks (DQN)

DQN extends Q-learning to high-dimensional state spaces using neural networks.

### Key Innovations

1. **Experience Replay**: Store transitions in buffer, sample randomly for training
2. **Target Network**: Separate network for stable target Q-values
3. **Neural Network**: Approximate Q-function instead of tabular representation

### Architecture

```python
from puffin.rl import DQNAgent, DQNetwork
import torch
import numpy as np

# Create DQN agent
agent = DQNAgent(
    state_dim=10,              # Dimension of state space
    action_dim=3,              # Number of discrete actions
    lr=1e-4,                   # Learning rate
    gamma=0.99,                # Discount factor
    buffer_size=10000,         # Replay buffer size
    batch_size=64,             # Training batch size
    target_update=100,         # Steps between target updates
    hidden_dims=[128, 64]      # Network architecture
)

# Check network architecture
print(agent.online_network)
```

### Training DQN

```python
from puffin.rl import DQNAgent, TradingEnvironment
import numpy as np
import yfinance as yf

# Download market data
data = yf.download('SPY', start='2020-01-01', end='2023-12-31')
prices = data['Close'].values

# Create features (returns, moving averages, etc.)
returns = np.diff(prices) / prices[:-1]
ma_20 = pd.Series(prices).rolling(20).mean().fillna(method='bfill').values
ma_50 = pd.Series(prices).rolling(50).mean().fillna(method='bfill').values

features = np.column_stack([
    returns,
    (prices[1:] - ma_20[1:]) / prices[1:],
    (prices[1:] - ma_50[1:]) / prices[1:]
])

# Create trading environment
env = TradingEnvironment(
    prices=prices[1:],
    features=features,
    initial_cash=100000,
    commission=0.001,
    reward_type='sharpe'
)

# Initialize DQN agent
state_dim = env.observation_space.shape[0]
agent = DQNAgent(
    state_dim=state_dim,
    action_dim=3,
    lr=1e-4,
    gamma=0.99,
    buffer_size=10000,
    batch_size=64
)

# Train agent
episode_rewards = agent.train(
    env,
    episodes=500,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.995,
    verbose=True
)

# Save trained model
agent.save('models/dqn_trading.pt')
```

### Double DQN (DDQN)

Double DQN reduces overestimation bias by using separate networks for action selection and evaluation:

```python
from puffin.rl import DDQNAgent, TradingEnvironment

# Create DDQN agent (same API as DQN)
agent = DDQNAgent(
    state_dim=state_dim,
    action_dim=3,
    lr=1e-4,
    gamma=0.99,
    buffer_size=10000,
    batch_size=64
)

# Train DDQN
episode_rewards = agent.train(env, episodes=500, verbose=True)

# DDQN uses double Q-learning update internally
# Action selection: argmax_a Q_online(s', a)
# Action evaluation: Q_target(s', argmax_a)
```

### Comparing DQN vs DDQN

```python
from puffin.rl import DQNAgent, DDQNAgent, compare_agents, TradingEnvironment

# Create environment
env = TradingEnvironment(prices, features)

# Train both agents
dqn = DQNAgent(state_dim, action_dim=3)
dqn_rewards = dqn.train(env, episodes=300)

ddqn = DDQNAgent(state_dim, action_dim=3)
ddqn_rewards = ddqn.train(env, episodes=300)

# Compare performance
agents = {
    'DQN': dqn,
    'DDQN': ddqn
}

comparison = compare_agents(agents, env, n_episodes=50)
print(comparison)
```

## Custom Trading Environment

Our custom Gymnasium environment enables realistic backtesting of RL agents.

### Environment Features

- **Observation Space**: Market features + portfolio state
- **Action Space**: Discrete (buy/hold/sell) or continuous (position sizing)
- **Reward Types**: P&L, Sharpe ratio, or risk-adjusted returns
- **Commission**: Transaction costs
- **Position Limits**: Maximum position size

### Creating Environment

```python
from puffin.rl import TradingEnvironment
import numpy as np

# Simple price series
prices = np.random.randn(1000).cumsum() + 100

# Create environment with default settings
env = TradingEnvironment(
    prices=prices,
    initial_cash=100000,
    commission=0.001,
    max_position=1.0,
    reward_type='pnl',
    discrete_actions=True
)

# Reset environment
obs, info = env.reset()
print(f"Initial observation shape: {obs.shape}")
print(f"Info: {info}")

# Take actions
for _ in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break

    env.render()
```

### Advanced Environment Configuration

```python
from puffin.rl import TradingEnvironment
import pandas as pd
import numpy as np

# Load real market data
data = pd.read_csv('data/market_data.csv')
prices = data['close'].values

# Create technical indicators as features
data['rsi'] = calculate_rsi(data['close'])
data['macd'] = calculate_macd(data['close'])
data['bb_upper'], data['bb_lower'] = calculate_bollinger_bands(data['close'])

features = data[['rsi', 'macd', 'bb_upper', 'bb_lower']].values

# Create advanced environment
env = TradingEnvironment(
    prices=prices,
    features=features,
    initial_cash=100000,
    commission=0.001,          # 0.1% commission
    max_position=1.0,          # 100% portfolio
    reward_type='risk_adjusted',  # Penalize volatility
    discrete_actions=False,    # Continuous actions
    lookback=5                 # Include 5-step history
)

print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")
```

### Reward Functions

```python
# P&L reward (simple)
env_pnl = TradingEnvironment(prices, reward_type='pnl')

# Sharpe ratio (risk-adjusted)
env_sharpe = TradingEnvironment(prices, reward_type='sharpe')

# Custom risk-adjusted (volatility + drawdown penalty)
env_risk = TradingEnvironment(prices, reward_type='risk_adjusted')
```

## Proximal Policy Optimization (PPO)

PPO is a state-of-the-art policy gradient method that balances sample efficiency and stability.

### Why PPO for Trading?

- **Stability**: Clipped objective prevents destructive policy updates
- **Sample Efficiency**: Reuses data through multiple epochs
- **Continuous Actions**: Naturally handles position sizing
- **Proven Performance**: Works well across diverse domains

### Using stable-baselines3

```python
from puffin.rl import PPOTrader, TradingEnvironment, TradingCallback
import numpy as np

# Create trading environment
prices = np.random.randn(2000).cumsum() + 100
env = TradingEnvironment(
    prices=prices,
    discrete_actions=True,
    reward_type='sharpe'
)

# Create PPO agent
agent = PPOTrader(
    env=env,
    policy='MlpPolicy',
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,  # Encourage exploration
    verbose=1
)

# Train with callback
callback = TradingCallback(verbose=1, log_freq=1000)
agent.train(total_timesteps=100000, callback=callback)

# Save model
agent.save('models/ppo_trading.zip')
```

### Evaluating PPO Agent

```python
from puffin.rl import PPOTrader, TradingEnvironment

# Load trained agent
agent = PPOTrader.load_from_path('models/ppo_trading.zip', env)

# Evaluate on test data
test_prices = np.random.randn(500).cumsum() + 100
test_env = TradingEnvironment(prices=test_prices)

results = agent.evaluate(test_env, n_episodes=10, deterministic=True)

print(f"Mean Reward: {results['mean_reward']:.2f}")
print(f"Mean Return: {results['mean_return']:.2%}")
print(f"Mean Trades: {results['mean_trades']:.1f}")
```

### Custom PPO Policy

```python
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import gymnasium as gym

class CustomTradingNetwork(BaseFeaturesExtractor):
    """Custom feature extractor for trading."""

    def __init__(self, observation_space: gym.Space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        input_dim = observation_space.shape[0]

        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        return self.network(observations)

# Use custom network with PPO
from stable_baselines3.common.policies import ActorCriticPolicy

policy_kwargs = dict(
    features_extractor_class=CustomTradingNetwork,
    features_extractor_kwargs=dict(features_dim=128),
)

agent = PPOTrader(
    env=env,
    policy='MlpPolicy',
    policy_kwargs=policy_kwargs,
    learning_rate=3e-4
)
```

## Agent Evaluation

Comprehensive evaluation is crucial for understanding agent performance.

### Basic Evaluation

```python
from puffin.rl import evaluate_agent, DQNAgent, TradingEnvironment
import numpy as np

# Create environment and agent
prices = np.random.randn(1000).cumsum() + 100
env = TradingEnvironment(prices)
agent = DQNAgent(state_dim=4, action_dim=3)

# Evaluate agent
results = evaluate_agent(
    agent=agent,
    env=env,
    n_episodes=100,
    deterministic=True
)

print(f"Mean Reward: {results['mean_reward']:.2f}")
print(f"Std Reward: {results['std_reward']:.2f}")
print(f"Cumulative P&L: {results['cumulative_pnl']:.2f}")
print(f"Mean Return: {results['mean_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
print(f"Win Rate: {results['win_rate']:.2%}")
```

### Visualization

```python
from puffin.rl import (
    plot_episode_rewards,
    plot_cumulative_pnl,
    plot_drawdown,
    DQNAgent,
    TradingEnvironment
)
import matplotlib.pyplot as plt

# Train agent and collect rewards
agent = DQNAgent(state_dim=4, action_dim=3)
env = TradingEnvironment(prices)
rewards = agent.train(env, episodes=500)

# Plot training progress
fig = plot_episode_rewards(rewards, window=50)
plt.savefig('results/training_rewards.png')
plt.show()

# Collect episode history
env.reset()
history = []
done = False
while not done:
    action = agent.choose_action(env._get_observation(), epsilon=0.0)
    obs, reward, terminated, truncated, info = env.step(action)
    history.append(info)
    done = terminated or truncated

# Plot cumulative P&L
fig = plot_cumulative_pnl(history)
plt.savefig('results/cumulative_pnl.png')
plt.show()

# Plot drawdown
portfolio_values = [info['portfolio_value'] for info in history]
fig = plot_drawdown(portfolio_values)
plt.savefig('results/drawdown.png')
plt.show()
```

### Comparing Multiple Agents

```python
from puffin.rl import (
    compare_agents,
    plot_comparison,
    QLearningAgent,
    DQNAgent,
    DDQNAgent,
    PPOTrader,
    TradingEnvironment
)
import matplotlib.pyplot as plt

# Create environment
env = TradingEnvironment(prices)

# Train multiple agents
agents = {}

# Q-learning (with discretization wrapper)
q_agent = QLearningAgent(n_states=100, n_actions=3)
# ... train with discrete wrapper ...
agents['Q-Learning'] = q_agent

# DQN
dqn_agent = DQNAgent(state_dim=4, action_dim=3)
dqn_agent.train(env, episodes=300)
agents['DQN'] = dqn_agent

# DDQN
ddqn_agent = DDQNAgent(state_dim=4, action_dim=3)
ddqn_agent.train(env, episodes=300)
agents['DDQN'] = ddqn_agent

# PPO
ppo_agent = PPOTrader(env)
ppo_agent.train(total_timesteps=50000)
agents['PPO'] = ppo_agent

# Compare all agents
comparison = compare_agents(agents, env, n_episodes=50)
print(comparison)

# Plot comparison
fig = plot_comparison(comparison, metric='mean_return')
plt.savefig('results/agent_comparison.png')
plt.show()
```

## Best Practices

### 1. Environment Design

```python
# Use realistic transaction costs
env = TradingEnvironment(
    prices=prices,
    commission=0.001,      # 0.1% per trade
    max_position=1.0       # Limit position size
)

# Choose appropriate reward function
# - P&L: Simple, but ignores risk
# - Sharpe: Risk-adjusted returns
# - Risk-adjusted: Penalizes volatility and drawdowns
env = TradingEnvironment(prices, reward_type='risk_adjusted')
```

### 2. Feature Engineering

```python
import pandas as pd

def create_features(data):
    """Create comprehensive feature set."""
    features = pd.DataFrame()

    # Price-based features
    features['returns'] = data['close'].pct_change()
    features['log_returns'] = np.log(data['close'] / data['close'].shift(1))

    # Moving averages
    features['sma_20'] = data['close'].rolling(20).mean()
    features['sma_50'] = data['close'].rolling(50).mean()
    features['sma_ratio'] = features['sma_20'] / features['sma_50']

    # Volatility
    features['volatility'] = data['close'].pct_change().rolling(20).std()

    # Technical indicators
    features['rsi'] = calculate_rsi(data['close'], 14)
    features['macd'], features['signal'] = calculate_macd(data['close'])

    # Volume features
    features['volume_ma'] = data['volume'].rolling(20).mean()
    features['volume_ratio'] = data['volume'] / features['volume_ma']

    return features.fillna(method='bfill').values

features = create_features(data)
env = TradingEnvironment(prices, features=features)
```

### 3. Hyperparameter Tuning

```python
from puffin.rl import DQNAgent, TradingEnvironment
import optuna

def objective(trial):
    """Objective function for hyperparameter optimization."""
    # Sample hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    gamma = trial.suggest_uniform('gamma', 0.95, 0.999)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    buffer_size = trial.suggest_categorical('buffer_size', [5000, 10000, 20000])

    # Create and train agent
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=3,
        lr=lr,
        gamma=gamma,
        batch_size=batch_size,
        buffer_size=buffer_size
    )

    rewards = agent.train(env, episodes=100, verbose=False)

    # Return mean reward of last 20 episodes
    return np.mean(rewards[-20:])

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print("Best hyperparameters:", study.best_params)
```

### 4. Training Strategies

```python
# Use curriculum learning: start with simple environments
simple_env = TradingEnvironment(prices[:500])  # Shorter series
agent = DQNAgent(state_dim, action_dim=3)
agent.train(simple_env, episodes=100)

# Then move to complex environment
complex_env = TradingEnvironment(prices, features=features)
agent.train(complex_env, episodes=400)

# Implement early stopping
best_reward = -np.inf
patience = 50
patience_counter = 0

for episode in range(1000):
    # ... training loop ...
    avg_reward = np.mean(recent_rewards)

    if avg_reward > best_reward:
        best_reward = avg_reward
        patience_counter = 0
        agent.save('best_model.pt')
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping triggered")
        break
```

### 5. Validation and Testing

```python
# Split data into train/validation/test
train_prices = prices[:int(0.6 * len(prices))]
val_prices = prices[int(0.6 * len(prices)):int(0.8 * len(prices))]
test_prices = prices[int(0.8 * len(prices)):]

# Train on training set
train_env = TradingEnvironment(train_prices)
agent.train(train_env, episodes=500)

# Validate during training
val_env = TradingEnvironment(val_prices)
val_results = evaluate_agent(agent, val_env, n_episodes=10)
print(f"Validation return: {val_results['mean_return']:.2%}")

# Final test on held-out data
test_env = TradingEnvironment(test_prices)
test_results = evaluate_agent(agent, test_env, n_episodes=20)
print(f"Test return: {test_results['mean_return']:.2%}")
print(f"Test Sharpe: {test_results['sharpe']:.2f}")
```

### 6. Risk Management

```python
# Add position limits
env = TradingEnvironment(
    prices=prices,
    max_position=0.3  # Max 30% of portfolio
)

# Use risk-adjusted rewards
env = TradingEnvironment(
    prices=prices,
    reward_type='risk_adjusted'  # Penalizes volatility
)

# Implement stop-loss in environment
class TradingEnvWithStopLoss(TradingEnvironment):
    def __init__(self, *args, stop_loss_pct=0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop_loss_pct = stop_loss_pct

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # Check stop-loss
        if info['cumulative_return'] < -self.stop_loss_pct:
            terminated = True
            reward -= 10  # Penalty for hitting stop-loss

        return obs, reward, terminated, truncated, info
```

## Summary

Deep reinforcement learning provides a powerful framework for developing adaptive trading strategies:

1. **Q-Learning**: Foundation for discrete state/action spaces
2. **DQN/DDQN**: Scalable to high-dimensional observations
3. **Trading Environment**: Realistic backtesting with transaction costs
4. **PPO**: State-of-the-art policy gradient method
5. **Evaluation**: Comprehensive metrics and visualization

Key takeaways:
- Start with simple environments and gradually increase complexity
- Use appropriate reward functions (Sharpe, risk-adjusted)
- Include transaction costs and position limits
- Tune hyperparameters systematically
- Validate on held-out data
- Implement risk management

Next steps:
- Experiment with multi-asset portfolios
- Incorporate alternative data sources
- Explore hierarchical RL for multi-timeframe trading
- Implement online learning for live trading

## References

- [Sutton & Barto (2018). Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)
- [Mnih et al. (2015). Human-level control through deep reinforcement learning](https://doi.org/10.1038/nature14236)
- van Hasselt et al. (2016). Deep Reinforcement Learning with Double Q-learning
- [Schulman et al. (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- stable-baselines3 documentation: https://stable-baselines3.readthedocs.io/

## Source Code

Browse the implementation: [`puffin/rl/`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/rl)
