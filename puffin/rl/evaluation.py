"""
Evaluation and visualization tools for reinforcement learning agents.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional
import gymnasium as gym


def evaluate_agent(
    agent: Any,
    env: gym.Env,
    n_episodes: int = 100,
    deterministic: bool = True,
    render: bool = False
) -> Dict[str, float]:
    """
    Evaluate an RL agent on an environment.

    Parameters
    ----------
    agent : Any
        Agent with predict() or choose_action() method
    env : gym.Env
        Environment to evaluate on
    n_episodes : int, default=100
        Number of episodes to run
    deterministic : bool, default=True
        Whether to use deterministic policy (if supported)
    render : bool, default=False
        Whether to render episodes

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - mean_reward: Mean episode reward
        - std_reward: Std of episode rewards
        - cumulative_pnl: Total P&L across all episodes
        - mean_return: Mean cumulative return
        - sharpe: Sharpe ratio of returns
        - max_drawdown: Maximum drawdown
        - win_rate: Fraction of profitable episodes
        - mean_trades: Mean number of trades per episode

    Examples
    --------
    >>> from puffin.rl.dqn import DQNAgent
    >>> from puffin.rl.trading_env import TradingEnvironment
    >>> import numpy as np
    >>> prices = np.random.randn(1000).cumsum() + 100
    >>> env = TradingEnvironment(prices)
    >>> agent = DQNAgent(state_dim=4, action_dim=3)
    >>> results = evaluate_agent(agent, env, n_episodes=10)
    """
    episode_rewards = []
    episode_returns = []
    episode_trades = []
    portfolio_values = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            # Get action from agent
            if hasattr(agent, 'predict'):
                # stable-baselines3 style
                action = agent.predict(obs, deterministic=deterministic)
                if isinstance(action, tuple):
                    action = action[0]
            elif hasattr(agent, 'choose_action'):
                # Custom agent style (DQN, Q-learning)
                if hasattr(agent, 'online_network'):
                    # DQN-style
                    epsilon = 0.0 if deterministic else 0.1
                    action = agent.choose_action(obs, epsilon=epsilon)
                else:
                    # Q-learning style
                    action = agent.choose_action(obs)
            else:
                raise ValueError("Agent must have predict() or choose_action() method")

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            if render:
                env.render()

        episode_rewards.append(episode_reward)
        episode_returns.append(info.get('cumulative_return', 0.0))
        episode_trades.append(info.get('trades', 0))
        portfolio_values.append(info.get('portfolio_value', 0.0))

    # Calculate metrics
    episode_rewards = np.array(episode_rewards)
    episode_returns = np.array(episode_returns)

    # Sharpe ratio
    if len(episode_returns) > 1 and np.std(episode_returns) > 0:
        sharpe = np.mean(episode_returns) / np.std(episode_returns) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Maximum drawdown
    cumulative_returns = np.cumprod(1 + episode_returns) - 1
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - running_max) / (1 + running_max)
    max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0.0

    # Win rate
    win_rate = np.sum(episode_returns > 0) / len(episode_returns) if len(episode_returns) > 0 else 0.0

    return {
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'cumulative_pnl': float(np.sum(episode_rewards)),
        'mean_return': float(np.mean(episode_returns)),
        'std_return': float(np.std(episode_returns)),
        'sharpe': float(sharpe),
        'max_drawdown': float(max_drawdown),
        'win_rate': float(win_rate),
        'mean_trades': float(np.mean(episode_trades)),
        'mean_portfolio_value': float(np.mean(portfolio_values))
    }


def plot_episode_rewards(
    rewards: List[float],
    window: int = 50,
    title: str = "Episode Rewards",
    figsize: tuple = (12, 6)
) -> plt.Figure:
    """
    Plot episode rewards with moving average.

    Parameters
    ----------
    rewards : List[float]
        List of episode rewards
    window : int, default=50
        Window size for moving average
    title : str, default="Episode Rewards"
        Plot title
    figsize : tuple, default=(12, 6)
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure

    Examples
    --------
    >>> rewards = [10, 15, 12, 18, 20, 25]
    >>> fig = plot_episode_rewards(rewards)
    >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)

    episodes = np.arange(len(rewards))
    rewards_array = np.array(rewards)

    # Plot raw rewards
    ax.plot(episodes, rewards_array, alpha=0.3, label='Episode Reward')

    # Plot moving average
    if len(rewards) >= window:
        moving_avg = pd.Series(rewards_array).rolling(window=window).mean()
        ax.plot(episodes, moving_avg, linewidth=2, label=f'{window}-Episode Moving Avg')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_cumulative_pnl(
    env_history: List[Dict[str, Any]],
    title: str = "Cumulative P&L",
    figsize: tuple = (12, 6)
) -> plt.Figure:
    """
    Plot cumulative P&L from environment history.

    Parameters
    ----------
    env_history : List[Dict[str, Any]]
        List of info dicts from environment steps
    title : str, default="Cumulative P&L"
        Plot title
    figsize : tuple, default=(12, 6)
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure

    Examples
    --------
    >>> history = [
    ...     {'portfolio_value': 100000, 'step': 0},
    ...     {'portfolio_value': 101000, 'step': 1},
    ...     {'portfolio_value': 102000, 'step': 2}
    ... ]
    >>> fig = plot_cumulative_pnl(history)
    >>> plt.show()
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    steps = [info['step'] for info in env_history]
    portfolio_values = [info['portfolio_value'] for info in env_history]
    initial_value = portfolio_values[0] if portfolio_values else 100000

    # Calculate returns
    returns = [(pv - initial_value) / initial_value * 100 for pv in portfolio_values]

    # Portfolio value
    ax1.plot(steps, portfolio_values, linewidth=2)
    ax1.axhline(y=initial_value, color='r', linestyle='--', alpha=0.5, label='Initial Value')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Returns
    ax2.plot(steps, returns, linewidth=2, color='green')
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax2.fill_between(steps, returns, 0, alpha=0.3, color='green')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Return (%)')
    ax2.set_title('Cumulative Returns')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_drawdown(
    portfolio_values: List[float],
    title: str = "Drawdown",
    figsize: tuple = (12, 4)
) -> plt.Figure:
    """
    Plot drawdown over time.

    Parameters
    ----------
    portfolio_values : List[float]
        Portfolio values over time
    title : str, default="Drawdown"
        Plot title
    figsize : tuple, default=(12, 4)
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    portfolio_values = np.array(portfolio_values)
    running_max = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - running_max) / running_max * 100

    steps = np.arange(len(portfolio_values))
    ax.fill_between(steps, drawdown, 0, alpha=0.3, color='red')
    ax.plot(steps, drawdown, linewidth=2, color='darkred')
    ax.set_xlabel('Step')
    ax.set_ylabel('Drawdown (%)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Add max drawdown annotation
    max_dd_idx = np.argmin(drawdown)
    max_dd = drawdown[max_dd_idx]
    ax.annotate(f'Max DD: {max_dd:.2f}%',
                xy=(max_dd_idx, max_dd),
                xytext=(max_dd_idx, max_dd + 5),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=10)

    plt.tight_layout()
    return fig


def compare_agents(
    agents_dict: Dict[str, Any],
    env: gym.Env,
    n_episodes: int = 50,
    metrics: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compare multiple agents on the same environment.

    Parameters
    ----------
    agents_dict : Dict[str, Any]
        Dictionary mapping agent names to agent objects
    env : gym.Env
        Environment to evaluate on
    n_episodes : int, default=50
        Number of episodes per agent
    metrics : Optional[List[str]], default=None
        Metrics to include (None = all metrics)

    Returns
    -------
    pd.DataFrame
        Comparison table with agents as rows and metrics as columns

    Examples
    --------
    >>> from puffin.rl.dqn import DQNAgent, DDQNAgent
    >>> from puffin.rl.trading_env import TradingEnvironment
    >>> import numpy as np
    >>> prices = np.random.randn(1000).cumsum() + 100
    >>> env = TradingEnvironment(prices)
    >>> agents = {
    ...     'DQN': DQNAgent(state_dim=4, action_dim=3),
    ...     'DDQN': DDQNAgent(state_dim=4, action_dim=3)
    ... }
    >>> comparison = compare_agents(agents, env, n_episodes=10)
    """
    results = {}

    for agent_name, agent in agents_dict.items():
        print(f"Evaluating {agent_name}...")
        agent_results = evaluate_agent(agent, env, n_episodes=n_episodes)
        results[agent_name] = agent_results

    # Create DataFrame
    df = pd.DataFrame(results).T

    # Filter metrics if specified
    if metrics is not None:
        df = df[metrics]

    # Sort by mean return
    if 'mean_return' in df.columns:
        df = df.sort_values('mean_return', ascending=False)

    return df


def plot_comparison(
    comparison_df: pd.DataFrame,
    metric: str = 'mean_return',
    title: Optional[str] = None,
    figsize: tuple = (10, 6)
) -> plt.Figure:
    """
    Plot comparison of agents for a specific metric.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Comparison DataFrame from compare_agents()
    metric : str, default='mean_return'
        Metric to plot
    title : Optional[str], default=None
        Plot title (auto-generated if None)
    figsize : tuple, default=(10, 6)
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    if metric not in comparison_df.columns:
        raise ValueError(f"Metric '{metric}' not found in comparison DataFrame")

    fig, ax = plt.subplots(figsize=figsize)

    agents = comparison_df.index.tolist()
    values = comparison_df[metric].values

    # Create bar plot
    bars = ax.bar(agents, values, alpha=0.7, edgecolor='black')

    # Color bars based on positive/negative
    for bar, value in zip(bars, values):
        if value >= 0:
            bar.set_color('green')
        else:
            bar.set_color('red')

    # Add value labels on bars
    for i, (agent, value) in enumerate(zip(agents, values)):
        ax.text(i, value, f'{value:.4f}', ha='center',
                va='bottom' if value >= 0 else 'top')

    ax.set_xlabel('Agent')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title or f'Agent Comparison: {metric.replace("_", " ").title()}')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig
