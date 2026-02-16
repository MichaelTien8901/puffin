"""
Proximal Policy Optimization (PPO) agent using stable-baselines3.
"""

import numpy as np
from typing import Optional, Dict, Any, Callable
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym


class PPOTrader:
    """
    PPO-based trading agent using stable-baselines3.

    Parameters
    ----------
    env : gym.Env
        Trading environment
    policy : str, default='MlpPolicy'
        Policy network type ('MlpPolicy', 'CnnPolicy', etc.)
    learning_rate : float, default=3e-4
        Learning rate for optimizer
    n_steps : int, default=2048
        Number of steps to run for each environment per update
    batch_size : int, default=64
        Minibatch size
    n_epochs : int, default=10
        Number of epochs when optimizing the surrogate loss
    gamma : float, default=0.99
        Discount factor
    gae_lambda : float, default=0.95
        Factor for trade-off of bias vs variance for GAE
    clip_range : float, default=0.2
        Clipping parameter for PPO
    ent_coef : float, default=0.0
        Entropy coefficient for exploration
    vf_coef : float, default=0.5
        Value function coefficient for the loss calculation
    verbose : int, default=0
        Verbosity level (0: no output, 1: info, 2: debug)
    device : str, default='auto'
        Device to use ('auto', 'cpu', 'cuda')

    Attributes
    ----------
    model : PPO
        Underlying PPO model from stable-baselines3
    """

    def __init__(
        self,
        env: gym.Env,
        policy: str = 'MlpPolicy',
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        verbose: int = 0,
        device: str = 'auto'
    ):
        self.env = env
        self.policy = policy

        # Wrap environment in DummyVecEnv for stable-baselines3
        if not isinstance(env, DummyVecEnv):
            self.vec_env = DummyVecEnv([lambda: env])
        else:
            self.vec_env = env

        # Create PPO model
        self.model = PPO(
            policy=policy,
            env=self.vec_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            verbose=verbose,
            device=device
        )

    def train(
        self,
        total_timesteps: int = 100000,
        callback: Optional[BaseCallback] = None,
        log_interval: int = 10,
        tb_log_name: str = "PPOTrader",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False
    ) -> 'PPOTrader':
        """
        Train the PPO agent.

        Parameters
        ----------
        total_timesteps : int, default=100000
            Total number of samples to train on
        callback : Optional[BaseCallback], default=None
            Callback function called at each step
        log_interval : int, default=10
            Number of timesteps between logging
        tb_log_name : str, default="PPOTrader"
            Name for tensorboard logging
        reset_num_timesteps : bool, default=True
            Whether to reset timesteps or continue
        progress_bar : bool, default=False
            Whether to show progress bar

        Returns
        -------
        self
            Returns self for method chaining

        Examples
        --------
        >>> from puffin.rl.trading_env import TradingEnvironment
        >>> import numpy as np
        >>> prices = np.random.randn(1000).cumsum() + 100
        >>> env = TradingEnvironment(prices)
        >>> agent = PPOTrader(env)
        >>> agent.train(total_timesteps=10000)
        """
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar
        )

        return self

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> np.ndarray:
        """
        Predict action for given observation.

        Parameters
        ----------
        observation : np.ndarray
            Current observation
        deterministic : bool, default=True
            Whether to use deterministic or stochastic actions

        Returns
        -------
        np.ndarray
            Predicted action
        """
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action

    def evaluate(
        self,
        env: Optional[gym.Env] = None,
        n_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate the agent.

        Parameters
        ----------
        env : Optional[gym.Env], default=None
            Environment to evaluate on (uses training env if None)
        n_episodes : int, default=10
            Number of episodes to evaluate
        deterministic : bool, default=True
            Whether to use deterministic policy
        render : bool, default=False
            Whether to render episodes

        Returns
        -------
        Dict[str, float]
            Dictionary with evaluation metrics:
            - mean_reward: Mean episode reward
            - std_reward: Std of episode rewards
            - mean_return: Mean cumulative return
            - std_return: Std of cumulative returns
            - mean_trades: Mean number of trades
        """
        if env is None:
            env = self.env

        episode_rewards = []
        episode_returns = []
        episode_trades = []

        for episode in range(n_episodes):
            obs, info = env.reset()
            done = False
            episode_reward = 0.0

            while not done:
                action = self.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward

                if render:
                    env.render()

            episode_rewards.append(episode_reward)
            episode_returns.append(info.get('cumulative_return', 0.0))
            episode_trades.append(info.get('trades', 0))

        return {
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'mean_return': float(np.mean(episode_returns)),
            'std_return': float(np.std(episode_returns)),
            'mean_trades': float(np.mean(episode_trades)),
            'mean_portfolio_value': float(np.mean([
                info.get('portfolio_value', 0.0) for _ in range(n_episodes)
            ]))
        }

    def save(self, path: str) -> None:
        """
        Save the model to file.

        Parameters
        ----------
        path : str
            Path to save the model
        """
        self.model.save(path)

    def load(self, path: str, env: Optional[gym.Env] = None) -> None:
        """
        Load the model from file.

        Parameters
        ----------
        path : str
            Path to load the model from
        env : Optional[gym.Env], default=None
            Environment (uses current env if None)
        """
        if env is not None:
            self.env = env
            if not isinstance(env, DummyVecEnv):
                self.vec_env = DummyVecEnv([lambda: env])
            else:
                self.vec_env = env

        self.model = PPO.load(path, env=self.vec_env)

    @classmethod
    def load_from_path(
        cls,
        path: str,
        env: gym.Env
    ) -> 'PPOTrader':
        """
        Load a PPOTrader from saved model.

        Parameters
        ----------
        path : str
            Path to saved model
        env : gym.Env
            Environment to use

        Returns
        -------
        PPOTrader
            Loaded agent

        Examples
        --------
        >>> from puffin.rl.trading_env import TradingEnvironment
        >>> import numpy as np
        >>> prices = np.random.randn(1000).cumsum() + 100
        >>> env = TradingEnvironment(prices)
        >>> agent = PPOTrader.load_from_path('ppo_model.zip', env)
        """
        agent = cls(env)
        agent.load(path, env)
        return agent


class TradingCallback(BaseCallback):
    """
    Custom callback for logging trading metrics during training.

    Parameters
    ----------
    verbose : int, default=0
        Verbosity level
    log_freq : int, default=1000
        Frequency of logging (in timesteps)
    """

    def __init__(self, verbose: int = 0, log_freq: int = 1000):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_returns = []
        self.episode_trades = []

    def _on_step(self) -> bool:
        """Called at each step."""
        # Log when episode ends
        if self.locals.get('dones', [False])[0]:
            info = self.locals.get('infos', [{}])[0]
            if 'episode' in self.locals:
                episode_reward = self.locals['episode']['r']
                self.episode_rewards.append(episode_reward)

            if 'cumulative_return' in info:
                self.episode_returns.append(info['cumulative_return'])

            if 'trades' in info:
                self.episode_trades.append(info['trades'])

            # Log every log_freq steps
            if self.n_calls % self.log_freq == 0:
                if len(self.episode_rewards) > 0:
                    mean_reward = np.mean(self.episode_rewards[-10:])
                    mean_return = np.mean(self.episode_returns[-10:]) if self.episode_returns else 0
                    mean_trades = np.mean(self.episode_trades[-10:]) if self.episode_trades else 0

                    if self.verbose > 0:
                        print(f"Steps: {self.n_calls}, "
                              f"Mean Reward (last 10): {mean_reward:.2f}, "
                              f"Mean Return: {mean_return:.2%}, "
                              f"Mean Trades: {mean_trades:.1f}")

        return True
