---
layout: default
title: "PPO & Trading Environment"
parent: "Part 21: Deep Reinforcement Learning"
nav_order: 3
---

# PPO & Trading Environment

This section covers two tightly coupled topics: the custom Gymnasium trading environment that all RL agents use, and Proximal Policy Optimization (PPO) --- a state-of-the-art policy gradient method that naturally handles continuous action spaces for position sizing.

{: .note }
> PPO was introduced by OpenAI in 2017 and has become the default algorithm for many RL
> applications. Unlike DQN, PPO directly optimizes the policy (actor) and value function
> (critic) simultaneously, making it well-suited for continuous control problems.

## Custom Trading Environment

The `TradingEnvironment` class implements the Gymnasium interface, providing a realistic simulation for training and evaluating RL agents.

### Environment Features

- **Observation Space**: Market features concatenated with portfolio state (position, cash, unrealized P&L)
- **Action Space**: Discrete (buy/hold/sell) or continuous (position sizing from -1 to +1)
- **Reward Types**: P&L, Sharpe ratio, or risk-adjusted returns
- **Commission**: Configurable transaction costs
- **Position Limits**: Maximum position size constraint

### Basic Usage

```python
from puffin.rl.trading_env import TradingEnvironment
import numpy as np

# Simple price series
prices = np.random.randn(1000).cumsum() + 100

# Create environment with default settings
env = TradingEnvironment(
    prices=prices,
    initial_cash=100000,
    commission=0.001,       # 0.1% per trade
    max_position=1.0,       # 100% portfolio max
    reward_type='pnl',
    discrete_actions=True   # buy/hold/sell
)

# Standard Gymnasium interface
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

### Advanced Configuration with Features

For more realistic agents, pass pre-computed technical indicators as additional features.

```python
from puffin.rl.trading_env import TradingEnvironment
import pandas as pd
import numpy as np

# Load real market data
data = pd.read_csv('data/market_data.csv')
prices = data['close'].values

# Create technical indicators as features
features = np.column_stack([
    data['close'].pct_change().fillna(0).values,                    # returns
    (data['close'].rolling(20).mean() / data['close']).fillna(1),   # SMA ratio
    data['close'].pct_change().rolling(20).std().fillna(0).values,  # volatility
    data['volume'].rolling(20).mean().fillna(0).values              # avg volume
])

# Create environment with continuous actions for position sizing
env = TradingEnvironment(
    prices=prices,
    features=features,
    initial_cash=100000,
    commission=0.001,
    max_position=1.0,
    reward_type='risk_adjusted',
    discrete_actions=False,    # Continuous: position size in [-1, 1]
    lookback=5                 # Include 5-step history in observations
)

print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")
```

{: .warning }
> Always include transaction costs (`commission > 0`). Without them, agents learn to
> trade on every tick, which is unrealistic and produces misleading backtest results.

### Reward Functions

The choice of reward function dramatically affects learned behavior. The environment supports three built-in options.

```python
from puffin.rl.trading_env import TradingEnvironment

# P&L reward: simple profit/loss per step
# Pro: Easy to interpret. Con: Encourages excessive risk.
env_pnl = TradingEnvironment(prices, reward_type='pnl')

# Sharpe reward: risk-adjusted returns over a rolling window
# Pro: Penalizes volatility. Con: Requires enough history for stable estimate.
env_sharpe = TradingEnvironment(prices, reward_type='sharpe')

# Risk-adjusted reward: volatility penalty + drawdown penalty
# Pro: Most robust for live trading. Con: More hyperparameters to tune.
env_risk = TradingEnvironment(prices, reward_type='risk_adjusted')
```

{: .tip }
> Start with `reward_type='sharpe'` as a reasonable default. Switch to `risk_adjusted`
> if the agent takes excessive drawdowns during evaluation.

### Custom Environment Extensions

Extend `TradingEnvironment` to add stop-loss logic, multi-asset support, or custom rewards.

```python
from puffin.rl.trading_env import TradingEnvironment

class TradingEnvWithStopLoss(TradingEnvironment):
    """Trading environment with automatic stop-loss."""

    def __init__(self, *args, stop_loss_pct=0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop_loss_pct = stop_loss_pct

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # Terminate episode if cumulative loss exceeds threshold
        if info['cumulative_return'] < -self.stop_loss_pct:
            terminated = True
            reward -= 10  # Penalty for hitting stop-loss

        return obs, reward, terminated, truncated, info

# Use the extended environment
env = TradingEnvWithStopLoss(
    prices=prices,
    stop_loss_pct=0.05,     # 5% max drawdown
    reward_type='sharpe'
)
```

## Proximal Policy Optimization (PPO)

PPO is a policy gradient method that directly optimizes the policy (mapping states to action probabilities) rather than learning Q-values. Its key innovation is a clipped surrogate objective that prevents destructively large policy updates.

### Why PPO for Trading?

- **Stability**: The clipped objective bounds how much the policy can change per update
- **Sample Efficiency**: Reuses data through multiple optimization epochs per batch
- **Continuous Actions**: Naturally handles position sizing without discretization
- **Proven Performance**: Consistently performs well across diverse RL benchmarks

### PPOTrader with stable-baselines3

The `PPOTrader` class wraps stable-baselines3's PPO implementation with trading-specific defaults and evaluation methods.

```python
from puffin.rl.ppo_agent import PPOTrader, TradingCallback
from puffin.rl.trading_env import TradingEnvironment
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
    n_steps=2048,          # Steps per rollout before update
    batch_size=64,         # Minibatch size for optimization
    n_epochs=10,           # Optimization epochs per rollout
    gamma=0.99,            # Discount factor
    gae_lambda=0.95,       # GAE lambda for advantage estimation
    clip_range=0.2,        # PPO clipping parameter
    ent_coef=0.01,         # Entropy bonus for exploration
    verbose=1
)

# Train with callback for logging
callback = TradingCallback(verbose=1, log_freq=1000)
agent.train(total_timesteps=100000, callback=callback)

# Save model
agent.save('models/ppo_trading.zip')
```

### Evaluating PPO Agent

```python
from puffin.rl.ppo_agent import PPOTrader
from puffin.rl.trading_env import TradingEnvironment
import numpy as np

# Load trained agent
agent = PPOTrader.load_from_path('models/ppo_trading.zip', env)

# Evaluate on unseen test data
test_prices = np.random.randn(500).cumsum() + 100
test_env = TradingEnvironment(prices=test_prices)

results = agent.evaluate(test_env, n_episodes=10, deterministic=True)

print(f"Mean Reward: {results['mean_reward']:.2f}")
print(f"Mean Return: {results['mean_return']:.2%}")
print(f"Mean Trades: {results['mean_trades']:.1f}")
```

### Custom Policy Network

For more control over the agent's architecture, define a custom feature extractor.

```python
from puffin.rl.ppo_agent import PPOTrader
from puffin.rl.trading_env import TradingEnvironment
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import gymnasium as gym

class CustomTradingNetwork(BaseFeaturesExtractor):
    """Custom feature extractor with dropout for regularization."""

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

## Comprehensive Agent Evaluation

Evaluate and compare all agent types side by side to understand their strengths and weaknesses.

```python
from puffin.rl.evaluation import (
    evaluate_agent,
    compare_agents,
    plot_episode_rewards,
    plot_drawdown,
    plot_comparison
)
from puffin.rl.dqn import DQNAgent, DDQNAgent
from puffin.rl.ppo_agent import PPOTrader
from puffin.rl.trading_env import TradingEnvironment
import matplotlib.pyplot as plt
import numpy as np

# Create environment
env = TradingEnvironment(prices, features)

# Train multiple agents
dqn = DQNAgent(state_dim=4, action_dim=3)
dqn.train(env, episodes=300)

ddqn = DDQNAgent(state_dim=4, action_dim=3)
ddqn.train(env, episodes=300)

ppo = PPOTrader(env)
ppo.train(total_timesteps=50000)

# Compare all agents
agents = {'DQN': dqn, 'DDQN': ddqn, 'PPO': ppo}
comparison = compare_agents(agents, env, n_episodes=50)
print(comparison)

# Plot comparison
fig = plot_comparison(comparison, metric='mean_return')
plt.savefig('results/agent_comparison.png')
plt.show()
```

### Evaluation Metrics

```python
from puffin.rl.evaluation import evaluate_agent

results = evaluate_agent(
    agent=agent,
    env=env,
    n_episodes=100,
    deterministic=True
)

print(f"Mean Reward:     {results['mean_reward']:.2f}")
print(f"Std Reward:      {results['std_reward']:.2f}")
print(f"Cumulative P&L:  {results['cumulative_pnl']:.2f}")
print(f"Mean Return:     {results['mean_return']:.2%}")
print(f"Sharpe Ratio:    {results['sharpe']:.2f}")
print(f"Max Drawdown:    {results['max_drawdown']:.2%}")
print(f"Win Rate:        {results['win_rate']:.2%}")
```

## Best Practices

### Feature Engineering

Good features are critical for RL agent performance. Combine price-based and volume-based signals.

```python
import pandas as pd
import numpy as np

def create_features(data):
    """Create comprehensive feature set for RL agents."""
    features = pd.DataFrame()

    # Price-based
    features['returns'] = data['close'].pct_change()
    features['log_returns'] = np.log(data['close'] / data['close'].shift(1))

    # Moving averages
    features['sma_20'] = data['close'].rolling(20).mean()
    features['sma_50'] = data['close'].rolling(50).mean()
    features['sma_ratio'] = features['sma_20'] / features['sma_50']

    # Volatility
    features['volatility'] = data['close'].pct_change().rolling(20).std()

    # Volume
    features['volume_ma'] = data['volume'].rolling(20).mean()
    features['volume_ratio'] = data['volume'] / features['volume_ma']

    return features.bfill().values
```

### Training Strategies

```python
from puffin.rl.dqn import DQNAgent
from puffin.rl.trading_env import TradingEnvironment
import numpy as np

# Curriculum learning: start simple, increase complexity
simple_env = TradingEnvironment(prices[:500])
agent = DQNAgent(state_dim, action_dim=3)
agent.train(simple_env, episodes=100)

complex_env = TradingEnvironment(prices, features=features)
agent.train(complex_env, episodes=400)

# Early stopping on validation performance
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

{: .warning }
> Always split data into train/validation/test sets. Train on the first 60%, validate
> during training on the next 20%, and report final results only on the last 20%.
> Never tune hyperparameters on test data.

### Validation and Testing

```python
from puffin.rl.evaluation import evaluate_agent
from puffin.rl.trading_env import TradingEnvironment

# Split data temporally (never shuffle time series!)
train_prices = prices[:int(0.6 * len(prices))]
val_prices = prices[int(0.6 * len(prices)):int(0.8 * len(prices))]
test_prices = prices[int(0.8 * len(prices)):]

# Train
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

### Risk Management

```python
from puffin.rl.trading_env import TradingEnvironment

# Limit position size to prevent concentration risk
env = TradingEnvironment(
    prices=prices,
    max_position=0.3,           # Max 30% of portfolio
    commission=0.001,
    reward_type='risk_adjusted' # Penalizes volatility and drawdowns
)
```

{: .tip }
> Combine position limits with risk-adjusted rewards for the most robust agents.
> An agent trained with `risk_adjusted` rewards and `max_position=0.3` will naturally
> learn conservative position sizing.

## Source Code

| File | Description |
|---|---|
| [`puffin/rl/trading_env.py`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/rl/trading_env.py) | `TradingEnvironment` Gymnasium environment |
| [`puffin/rl/ppo_agent.py`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/rl/ppo_agent.py) | `PPOTrader` and `TradingCallback` classes |
| [`puffin/rl/evaluation.py`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/rl/evaluation.py) | `evaluate_agent`, `compare_agents`, and plotting utilities |
| [`tests/rl/test_q_learning.py`](https://github.com/MichaelTien8901/puffin/tree/main/tests/rl/test_q_learning.py) | Integration tests for RL agents and environment |

## References

- Schulman et al. (2017). Proximal Policy Optimization Algorithms. *arXiv:1707.06347*
- [stable-baselines3 documentation](https://stable-baselines3.readthedocs.io/)
- Brockman et al. (2016). OpenAI Gym. *arXiv:1606.01540*
