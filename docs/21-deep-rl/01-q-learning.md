---
layout: default
title: "Q-Learning Fundamentals"
parent: "Part 21: Deep Reinforcement Learning"
nav_order: 1
---

# Q-Learning Fundamentals

Q-learning is a foundational reinforcement learning algorithm that learns action-value functions through temporal difference learning. It works by maintaining a table of Q-values --- one entry for every state-action pair --- and updating them toward better estimates as the agent interacts with the environment.

{: .note }
> Q-learning is an *off-policy* algorithm: it learns about the optimal policy regardless of
> the exploration strategy used during training. This makes it sample-efficient compared
> to on-policy methods like SARSA.

## The Bellman Equation

Q-learning updates the Q-table using the Bellman optimality equation:

```
Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
```

Where:
- **alpha**: Learning rate controlling update magnitude (typically 0.01--0.1)
- **gamma**: Discount factor weighting future vs immediate rewards (typically 0.95--0.99)
- **r**: Immediate reward received after taking action `a` in state `s`
- **s'**: Next state after the transition
- **max_a' Q(s',a')**: Best estimated future value from the next state

The term `r + gamma * max_a' Q(s',a') - Q(s,a)` is the **temporal difference (TD) error** --- the gap between the current estimate and a better one.

## QLearningAgent

The `QLearningAgent` class provides a complete tabular Q-learning implementation with epsilon-greedy exploration and configurable decay schedules.

```python
from puffin.rl.q_learning import QLearningAgent
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
    epsilon_decay=0.995    # Decay factor per episode
)

# Train the agent
rewards = agent.train(env, episodes=1000, verbose=True)

# Get learned policy (best action per state)
policy = agent.get_policy()
print("Learned policy:", policy)

# Save the Q-table for later use
agent.save("q_table.npy")
```

{: .tip }
> Start with a high epsilon (1.0) for full exploration and decay it slowly. If epsilon
> decays too fast, the agent may converge to a suboptimal policy before discovering
> better actions.

## Epsilon-Greedy Exploration

The epsilon-greedy strategy balances exploration and exploitation:

- With probability **epsilon**, choose a random action (explore)
- With probability **1 - epsilon**, choose the action with highest Q-value (exploit)

Epsilon typically decays over training so the agent explores broadly early on and refines its policy later:

```python
from puffin.rl.q_learning import QLearningAgent

# Aggressive exploration early, refined exploitation later
agent = QLearningAgent(
    n_states=100,
    n_actions=3,
    epsilon=1.0,           # Start with 100% exploration
    epsilon_decay=0.998,   # Slow decay
    epsilon_min=0.01       # Never fully stop exploring
)

# After 1000 episodes: epsilon ~ 0.01 * 0.998^1000 ~ 0.135
# After 2000 episodes: epsilon ~ 0.01 * 0.998^2000 ~ 0.018
```

## State Discretization

For continuous state spaces (like market prices), observations must be converted to discrete indices. The `discretize_state` function bins each dimension independently and returns a combined index.

```python
from puffin.rl.q_learning import discretize_state
import numpy as np

# Define bins for each observation dimension
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

{: .warning }
> The number of states grows exponentially with dimensions. With 10 bins per dimension
> and 5 dimensions, you get 10^5 = 100,000 states. For high-dimensional observations,
> use DQN instead of tabular Q-learning.

## Trading with Q-Learning

Applying Q-learning to trading requires wrapping the `TradingEnvironment` with a discretization layer. The agent learns a policy over discretized price bins.

```python
from puffin.rl.q_learning import QLearningAgent, discretize_state
from puffin.rl.trading_env import TradingEnvironment
import numpy as np
import pandas as pd

# Load price data
prices = pd.read_csv('data/prices.csv')['close'].values

# Create bins for price discretization using percentiles
price_bins = [np.percentile(prices, q) for q in range(0, 101, 10)]
bins = [np.array(price_bins)]

# Create wrapper for discretization
class DiscreteWrapper:
    """Wraps a continuous environment with state discretization."""

    def __init__(self, env, bins):
        self.env = env
        self.bins = bins
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self):
        obs, info = self.env.reset()
        return discretize_state(obs[:1], self.bins), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        discrete_obs = discretize_state(obs[:1], self.bins)
        return discrete_obs, reward, terminated, truncated, info

# Create trading environment with discrete actions (buy/hold/sell)
base_env = TradingEnvironment(prices, discrete_actions=True)
env = DiscreteWrapper(base_env, bins)

# Train Q-learning agent
n_states = (len(bins[0]) - 1)
agent = QLearningAgent(
    n_states=n_states,
    n_actions=3,        # 0=sell, 1=hold, 2=buy
    lr=0.1,
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.995
)
rewards = agent.train(env, episodes=500)

print(f"Mean reward (last 100): {np.mean(rewards[-100:]):.2f}")
```

## Q-Table Inspection

After training, the Q-table reveals what the agent has learned. Each row is a state, each column is an action, and the values represent expected cumulative rewards.

```python
from puffin.rl.q_learning import QLearningAgent
import numpy as np

# After training...
q_table = agent.q_table

# Find states where the agent strongly prefers one action
for state in range(agent.n_states):
    q_values = q_table[state]
    best_action = np.argmax(q_values)
    action_names = ['sell', 'hold', 'buy']

    if np.max(q_values) > 0:
        print(f"State {state}: best={action_names[best_action]}, "
              f"Q={q_values}")
```

{: .tip }
> Inspect the Q-table after training to verify the agent has learned sensible policies.
> If all Q-values are near zero, the agent may need more training episodes or a
> different learning rate.

## Limitations of Tabular Q-Learning

Tabular Q-learning has fundamental constraints that motivate Deep Q-Networks:

1. **Curse of dimensionality**: The Q-table size is `n_states * n_actions`. With continuous or high-dimensional observations, the table becomes impractically large.

2. **No generalization**: Similar states are treated independently. A Q-value learned for state 42 says nothing about state 43, even if they represent nearly identical market conditions.

3. **Discretization artifacts**: Binning continuous data introduces quantization error. Fine bins create too many states; coarse bins lose information.

4. **Slow convergence**: Every state-action pair must be visited multiple times. In large state spaces, many pairs are rarely or never encountered.

These limitations lead naturally to function approximation --- using neural networks to estimate Q-values, which is the subject of the next section on DQN.

## Source Code

| File | Description |
|---|---|
| [`puffin/rl/q_learning.py`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/rl/q_learning.py) | `QLearningAgent` class and `discretize_state` function |
| [`puffin/rl/trading_env.py`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/rl/trading_env.py) | `TradingEnvironment` Gymnasium wrapper |
| [`tests/rl/test_q_learning.py`](https://github.com/MichaelTien8901/puffin/tree/main/tests/rl/test_q_learning.py) | Unit tests for Q-learning agent |

## References

- Sutton & Barto (2018). *Reinforcement Learning: An Introduction*, Chapter 6: Temporal-Difference Learning
- Watkins & Dayan (1992). Q-learning. *Machine Learning*, 8(3-4), 279--292
