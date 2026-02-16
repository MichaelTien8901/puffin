"""
Reinforcement Learning module for algorithmic trading.

This module provides:
- Tabular Q-learning agents
- Deep Q-Network (DQN) and Double DQN agents
- Custom Gymnasium trading environments
- PPO agents using stable-baselines3
- Evaluation and visualization tools
"""

from puffin.rl.q_learning import QLearningAgent, discretize_state
from puffin.rl.dqn import DQNAgent, DDQNAgent, DQNetwork, ReplayBuffer
from puffin.rl.trading_env import TradingEnvironment
from puffin.rl.ppo_agent import PPOTrader, TradingCallback
from puffin.rl.evaluation import (
    evaluate_agent,
    plot_episode_rewards,
    plot_cumulative_pnl,
    plot_drawdown,
    compare_agents,
    plot_comparison
)

__all__ = [
    # Q-learning
    'QLearningAgent',
    'discretize_state',
    # DQN
    'DQNAgent',
    'DDQNAgent',
    'DQNetwork',
    'ReplayBuffer',
    # Environment
    'TradingEnvironment',
    # PPO
    'PPOTrader',
    'TradingCallback',
    # Evaluation
    'evaluate_agent',
    'plot_episode_rewards',
    'plot_cumulative_pnl',
    'plot_drawdown',
    'compare_agents',
    'plot_comparison',
]
