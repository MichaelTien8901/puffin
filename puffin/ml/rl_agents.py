"""RL agent training utilities using stable-baselines3."""

from pathlib import Path

import numpy as np
import pandas as pd


def train_dqn(env, total_timesteps: int = 50_000, **kwargs) -> tuple:
    """Train a DQN agent on the trading environment.

    Args:
        env: Gymnasium-compatible trading environment.
        total_timesteps: Total training steps.

    Returns:
        Tuple of (trained_model, training_info).
    """
    from stable_baselines3 import DQN

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=kwargs.get("learning_rate", 1e-4),
        buffer_size=kwargs.get("buffer_size", 50_000),
        batch_size=kwargs.get("batch_size", 64),
        verbose=kwargs.get("verbose", 0),
    )
    model.learn(total_timesteps=total_timesteps)

    info = _evaluate_agent(model, env)
    return model, info


def train_ppo(env, total_timesteps: int = 50_000, **kwargs) -> tuple:
    """Train a PPO agent on the trading environment.

    Args:
        env: Gymnasium-compatible trading environment.
        total_timesteps: Total training steps.

    Returns:
        Tuple of (trained_model, training_info).
    """
    from stable_baselines3 import PPO

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=kwargs.get("learning_rate", 3e-4),
        n_steps=kwargs.get("n_steps", 2048),
        batch_size=kwargs.get("batch_size", 64),
        verbose=kwargs.get("verbose", 0),
    )
    model.learn(total_timesteps=total_timesteps)

    info = _evaluate_agent(model, env)
    return model, info


def _evaluate_agent(model, env, n_episodes: int = 5) -> dict:
    """Evaluate a trained agent."""
    episode_rewards = []
    episode_values = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(int(action))
            total_reward += reward
            if done or truncated:
                break
        episode_rewards.append(total_reward)
        episode_values.append(info.get("portfolio_value", 0))

    returns = np.array(episode_values) / env.initial_capital - 1

    return {
        "avg_episode_reward": float(np.mean(episode_rewards)),
        "avg_total_return": float(np.mean(returns)),
        "avg_final_value": float(np.mean(episode_values)),
        "sharpe_estimate": float(np.mean(returns) / np.std(returns)) if np.std(returns) > 0 else 0,
    }


def save_agent(model, path: str):
    """Save a trained RL agent."""
    model.save(path)


def load_agent(path: str, algorithm: str = "dqn"):
    """Load a saved RL agent."""
    if algorithm == "dqn":
        from stable_baselines3 import DQN
        return DQN.load(path)
    elif algorithm == "ppo":
        from stable_baselines3 import PPO
        return PPO.load(path)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
