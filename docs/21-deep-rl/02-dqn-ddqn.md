---
layout: default
title: "DQN & Double DQN"
parent: "Part 21: Deep Reinforcement Learning"
nav_order: 2
---

# DQN & Double DQN

Deep Q-Networks (DQN) extend tabular Q-learning to high-dimensional state spaces by using neural networks to approximate the Q-function. Instead of storing a value for every state-action pair, a neural network takes the state as input and outputs Q-values for all actions simultaneously.

{: .note }
> DQN was introduced by DeepMind in 2015 and achieved human-level performance on Atari games.
> The same principles apply to trading: the network learns which actions (buy, hold, sell)
> maximize expected returns given market observations.

## Key Innovations

DQN introduces three techniques that stabilize neural network Q-learning:

1. **Experience Replay**: Store transitions `(s, a, r, s')` in a buffer and sample random minibatches for training. This breaks temporal correlations and improves sample efficiency.

2. **Target Network**: Maintain a separate, slowly-updated copy of the Q-network for computing target values. This prevents the moving-target problem where updates chase a shifting objective.

3. **Neural Network Approximation**: Replace the Q-table with a network that generalizes across similar states. A state the agent has never seen can still get reasonable Q-values if it resembles states encountered during training.

## DQNetwork Architecture

The `DQNetwork` class implements the neural network that maps states to Q-values.

```python
from puffin.rl.dqn import DQNetwork
import torch

# Create network
network = DQNetwork(
    state_dim=10,           # Input: 10-dimensional state
    action_dim=3,           # Output: Q-values for 3 actions
    hidden_dims=[128, 64]   # Two hidden layers
)

# Forward pass
state = torch.randn(1, 10)
q_values = network(state)
print(f"Q-values shape: {q_values.shape}")   # torch.Size([1, 3])
print(f"Q-values: {q_values.detach().numpy()}")
```

## DQNAgent

The `DQNAgent` class combines the network, replay buffer, and training loop into a complete agent.

```python
from puffin.rl.dqn import DQNAgent
import torch

# Create DQN agent
agent = DQNAgent(
    state_dim=10,              # Dimension of state space
    action_dim=3,              # Number of discrete actions
    lr=1e-4,                   # Learning rate
    gamma=0.99,                # Discount factor
    buffer_size=10000,         # Replay buffer capacity
    batch_size=64,             # Training batch size
    target_update=100,         # Steps between target network syncs
    hidden_dims=[128, 64]      # Network architecture
)

# Inspect network architecture
print(agent.online_network)
```

{: .tip }
> The replay buffer size should be large enough to hold diverse experiences but small
> enough to fit in memory. For daily trading data, 10,000--50,000 transitions is
> usually sufficient.

## Training DQN on Market Data

Training a DQN agent involves creating features from market data and running episodes through the trading environment.

```python
from puffin.rl.dqn import DQNAgent
from puffin.rl.trading_env import TradingEnvironment
import numpy as np
import pandas as pd
import yfinance as yf

# Download market data
data = yf.download('SPY', start='2020-01-01', end='2023-12-31')
prices = data['Close'].values

# Create features (returns, moving average ratios)
returns = np.diff(prices) / prices[:-1]
ma_20 = pd.Series(prices).rolling(20).mean().bfill().values
ma_50 = pd.Series(prices).rolling(50).mean().bfill().values

features = np.column_stack([
    returns,
    (prices[1:] - ma_20[1:]) / prices[1:],
    (prices[1:] - ma_50[1:]) / prices[1:]
])

# Create trading environment with Sharpe reward
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

# Train agent with epsilon decay
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

## Experience Replay

The `ReplayBuffer` stores transitions and provides random sampling for training. Breaking temporal correlations is critical --- without replay, the network trains on sequential, highly correlated data that causes instability.

```python
from puffin.rl.dqn import ReplayBuffer
import numpy as np

# Create replay buffer
buffer = ReplayBuffer(capacity=10000)

# Store transitions during environment interaction
state = np.random.randn(10)
action = 1  # hold
reward = 0.02
next_state = np.random.randn(10)
done = False

buffer.push(state, action, reward, next_state, done)

# Sample a random minibatch for training
if len(buffer) >= 64:
    batch = buffer.sample(batch_size=64)
    states, actions, rewards, next_states, dones = batch
```

{: .warning }
> Do not start training until the replay buffer has accumulated enough transitions.
> A common rule of thumb is to wait until the buffer contains at least `batch_size`
> transitions (often 1,000--5,000 for stability).

## Double DQN (DDQN)

Standard DQN tends to overestimate Q-values because it uses the same network to both select and evaluate actions. The `max` operation introduces a positive bias: noisy Q-estimates cause the agent to consistently pick overestimated actions.

Double DQN decouples action selection from evaluation:

- **Action selection**: Use the online network to pick the best action: `a* = argmax_a Q_online(s', a)`
- **Action evaluation**: Use the target network to evaluate it: `Q_target(s', a*)`

This simple change significantly reduces overestimation bias.

```python
from puffin.rl.dqn import DDQNAgent
from puffin.rl.trading_env import TradingEnvironment

# Create DDQN agent (same API as DQN)
agent = DDQNAgent(
    state_dim=state_dim,
    action_dim=3,
    lr=1e-4,
    gamma=0.99,
    buffer_size=10000,
    batch_size=64,
    target_update=100
)

# Train DDQN
episode_rewards = agent.train(env, episodes=500, verbose=True)

# Internally, DDQN modifies only the target computation:
# DQN target:  r + gamma * max_a' Q_target(s', a')
# DDQN target: r + gamma * Q_target(s', argmax_a' Q_online(s', a'))
```

## DQN vs DDQN: Overestimation Bias

The key difference between DQN and DDQN is how they compute target Q-values. In practice, DDQN produces more stable training and better final policies, especially in noisy environments like financial markets.

```python
from puffin.rl.dqn import DQNAgent, DDQNAgent
from puffin.rl.evaluation import compare_agents
from puffin.rl.trading_env import TradingEnvironment

# Create environment
env = TradingEnvironment(prices, features)

# Train both agents on the same data
dqn = DQNAgent(state_dim, action_dim=3, lr=1e-4, gamma=0.99)
dqn_rewards = dqn.train(env, episodes=300)

ddqn = DDQNAgent(state_dim, action_dim=3, lr=1e-4, gamma=0.99)
ddqn_rewards = ddqn.train(env, episodes=300)

# Compare performance on held-out evaluation episodes
agents = {
    'DQN': dqn,
    'DDQN': ddqn
}

comparison = compare_agents(agents, env, n_episodes=50)
print(comparison)
```

{: .tip }
> In almost all cases, DDQN performs as well as or better than DQN. Unless you have a
> specific reason to use standard DQN, prefer DDQN as your default value-based agent.

## Hyperparameter Tuning

DQN performance is sensitive to hyperparameters. Use systematic search to find good configurations.

```python
from puffin.rl.dqn import DQNAgent
from puffin.rl.trading_env import TradingEnvironment
import optuna
import numpy as np

def objective(trial):
    """Optuna objective for DQN hyperparameter search."""
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    gamma = trial.suggest_uniform('gamma', 0.95, 0.999)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    buffer_size = trial.suggest_categorical('buffer_size', [5000, 10000, 20000])
    target_update = trial.suggest_categorical('target_update', [50, 100, 200])

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=3,
        lr=lr,
        gamma=gamma,
        batch_size=batch_size,
        buffer_size=buffer_size,
        target_update=target_update
    )

    rewards = agent.train(env, episodes=100, verbose=False)
    return np.mean(rewards[-20:])

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print("Best hyperparameters:", study.best_params)
print("Best reward:", study.best_value)
```

## Visualization

Track training progress and agent behavior to diagnose issues.

```python
from puffin.rl.evaluation import plot_episode_rewards, plot_cumulative_pnl
from puffin.rl.dqn import DQNAgent
from puffin.rl.trading_env import TradingEnvironment
import matplotlib.pyplot as plt

# Train agent and collect rewards
agent = DQNAgent(state_dim=4, action_dim=3)
env = TradingEnvironment(prices)
rewards = agent.train(env, episodes=500)

# Plot training progress with smoothing
fig = plot_episode_rewards(rewards, window=50)
plt.savefig('results/training_rewards.png')
plt.show()

# Collect episode history for P&L analysis
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
```

## Source Code

| File | Description |
|---|---|
| [`puffin/rl/dqn.py`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/rl/dqn.py) | `DQNAgent`, `DDQNAgent`, `DQNetwork`, and `ReplayBuffer` classes |
| [`puffin/rl/evaluation.py`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/rl/evaluation.py) | Evaluation metrics, plotting, and agent comparison utilities |
| [`tests/rl/test_q_learning.py`](https://github.com/MichaelTien8901/puffin/tree/main/tests/rl/test_q_learning.py) | Tests covering DQN and DDQN agents |

## References

- Mnih et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529--533
- van Hasselt et al. (2016). Deep Reinforcement Learning with Double Q-learning. *AAAI*
- Lin (1992). Self-improving reactive agents based on reinforcement learning, planning, and teaching. *Machine Learning*, 8(3), 293--321
