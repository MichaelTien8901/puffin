"""
Custom Gymnasium trading environment for reinforcement learning.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any, Literal


class TradingEnvironment(gym.Env):
    """
    Custom trading environment for RL agents.

    Observation space includes:
    - Features (prices, indicators, etc.)
    - Current position
    - Cash balance
    - Unrealized P&L

    Action space:
    - Discrete(3): 0=sell/short, 1=hold, 2=buy/long
    - Or continuous Box(-1, 1) for position sizing

    Parameters
    ----------
    prices : np.ndarray
        Price series, shape (n_timesteps,)
    features : Optional[np.ndarray], default=None
        Additional features, shape (n_timesteps, n_features)
    initial_cash : float, default=100000
        Starting cash balance
    commission : float, default=0.001
        Commission rate (0.001 = 0.1%)
    max_position : float, default=1.0
        Maximum position size (fraction of portfolio)
    reward_type : str, default='pnl'
        Reward calculation method: 'pnl', 'sharpe', 'risk_adjusted'
    discrete_actions : bool, default=True
        Whether to use discrete (3) or continuous actions
    lookback : int, default=1
        Number of historical steps to include in observation

    Attributes
    ----------
    observation_space : gym.Space
        Observation space definition
    action_space : gym.Space
        Action space definition
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        prices: np.ndarray,
        features: Optional[np.ndarray] = None,
        initial_cash: float = 100000.0,
        commission: float = 0.001,
        max_position: float = 1.0,
        reward_type: Literal['pnl', 'sharpe', 'risk_adjusted'] = 'pnl',
        discrete_actions: bool = True,
        lookback: int = 1
    ):
        super().__init__()

        # Validate inputs
        if len(prices) == 0:
            raise ValueError("prices cannot be empty")
        if features is not None and len(features) != len(prices):
            raise ValueError("features and prices must have same length")

        self.prices = np.array(prices, dtype=np.float32)
        self.features = features
        if self.features is not None:
            self.features = np.array(self.features, dtype=np.float32)
        self.initial_cash = initial_cash
        self.commission = commission
        self.max_position = max_position
        self.reward_type = reward_type
        self.lookback = lookback

        # State variables
        self.current_step = 0
        self.cash = initial_cash
        self.position = 0.0  # Number of shares
        self.entry_price = 0.0
        self.trades = []
        self.portfolio_values = []
        self.returns_history = []

        # Define action space
        if discrete_actions:
            # 0: sell/short, 1: hold, 2: buy/long
            self.action_space = spaces.Discrete(3)
            self.discrete_actions = True
        else:
            # Continuous action in [-1, 1]
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            )
            self.discrete_actions = False

        # Define observation space
        # Features + position + cash + unrealized_pnl
        n_features = 1 if features is None else features.shape[1] + 1
        obs_dim = n_features * lookback + 3  # +3 for position, cash, unrealized_pnl

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

    def _get_observation(self) -> np.ndarray:
        """Construct observation vector."""
        obs = []

        # Historical features and prices
        for i in range(self.lookback):
            step = max(0, self.current_step - i)
            price = self.prices[step]

            if self.features is not None:
                feats = self.features[step]
                obs.extend(feats)

            obs.append(price)

        # Current state
        current_price = self.prices[self.current_step]
        portfolio_value = self.cash + self.position * current_price
        unrealized_pnl = self.position * (current_price - self.entry_price) if self.position != 0 else 0.0

        # Normalize state variables
        obs.append(self.position / self.max_position if self.max_position > 0 else self.position)
        obs.append(self.cash / self.initial_cash)
        obs.append(unrealized_pnl / self.initial_cash)

        return np.array(obs, dtype=np.float32)

    def _calculate_reward(
        self,
        prev_portfolio_value: float,
        current_portfolio_value: float
    ) -> float:
        """Calculate reward based on reward_type."""
        if self.reward_type == 'pnl':
            # Simple P&L change
            return current_portfolio_value - prev_portfolio_value

        elif self.reward_type == 'sharpe':
            # Sharpe-like reward (return / std of returns)
            ret = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value
            self.returns_history.append(ret)

            if len(self.returns_history) < 2:
                return 0.0

            # Use recent window for std calculation
            window = min(len(self.returns_history), 20)
            returns_array = np.array(self.returns_history[-window:])
            std = np.std(returns_array)

            if std < 1e-6:
                return 0.0

            return ret / (std + 1e-6)

        elif self.reward_type == 'risk_adjusted':
            # Return penalized by volatility and drawdown
            ret = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value
            self.returns_history.append(ret)

            # Volatility penalty
            if len(self.returns_history) > 1:
                window = min(len(self.returns_history), 20)
                std = np.std(self.returns_history[-window:])
                vol_penalty = std * 0.5
            else:
                vol_penalty = 0.0

            # Drawdown penalty
            if len(self.portfolio_values) > 0:
                peak = max(self.portfolio_values)
                drawdown = (peak - current_portfolio_value) / peak if peak > 0 else 0.0
                dd_penalty = drawdown * 0.5
            else:
                dd_penalty = 0.0

            return ret - vol_penalty - dd_penalty

        else:
            return 0.0

    def _execute_action(self, action: int | float) -> None:
        """Execute trading action."""
        current_price = self.prices[self.current_step]
        prev_position = self.position

        if self.discrete_actions:
            # Discrete actions
            if action == 0:  # Sell/Short
                target_position = -self.max_position
            elif action == 1:  # Hold
                target_position = self.position
            else:  # action == 2, Buy/Long
                target_position = self.max_position
        else:
            # Continuous action
            action = float(action) if isinstance(action, np.ndarray) else action
            target_position = np.clip(action, -self.max_position, self.max_position)

        # Calculate trade
        position_change = target_position - self.position

        if abs(position_change) > 1e-6:
            # Calculate cost including commission
            trade_value = abs(position_change * current_price)
            commission_cost = trade_value * self.commission

            # Check if we have enough cash for the trade
            if position_change > 0:  # Buying
                cost = trade_value + commission_cost
                if cost <= self.cash:
                    self.cash -= cost
                    self.position += position_change
                    # Update entry price (weighted average)
                    if prev_position != 0:
                        total_value = (prev_position * self.entry_price +
                                     position_change * current_price)
                        self.entry_price = total_value / self.position
                    else:
                        self.entry_price = current_price

                    self.trades.append({
                        'step': self.current_step,
                        'action': 'buy',
                        'price': current_price,
                        'shares': position_change,
                        'commission': commission_cost
                    })
            else:  # Selling
                proceeds = trade_value - commission_cost
                self.cash += proceeds
                self.position += position_change

                if abs(self.position) < 1e-6:
                    self.position = 0.0
                    self.entry_price = 0.0

                self.trades.append({
                    'step': self.current_step,
                    'action': 'sell',
                    'price': current_price,
                    'shares': abs(position_change),
                    'commission': commission_cost
                })

    def step(self, action: int | float) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one time step.

        Parameters
        ----------
        action : int | float
            Action to take

        Returns
        -------
        observation : np.ndarray
            Next observation
        reward : float
            Reward for the action
        terminated : bool
            Whether episode is terminated
        truncated : bool
            Whether episode is truncated
        info : dict
            Additional information
        """
        # Calculate portfolio value before action
        current_price = self.prices[self.current_step]
        prev_portfolio_value = self.cash + self.position * current_price

        # Execute action
        self._execute_action(action)

        # Move to next step
        self.current_step += 1
        terminated = self.current_step >= len(self.prices) - 1
        truncated = False

        # Calculate portfolio value after action
        if not terminated:
            new_price = self.prices[self.current_step]
            current_portfolio_value = self.cash + self.position * new_price
        else:
            current_portfolio_value = prev_portfolio_value

        self.portfolio_values.append(current_portfolio_value)

        # Calculate reward
        reward = self._calculate_reward(prev_portfolio_value, current_portfolio_value)

        # Get observation
        observation = self._get_observation()

        # Info dict
        cumulative_return = (current_portfolio_value - self.initial_cash) / self.initial_cash
        info = {
            'portfolio_value': current_portfolio_value,
            'position': self.position,
            'cash': self.cash,
            'trades': len(self.trades),
            'cumulative_return': cumulative_return,
            'step': self.current_step
        }

        return observation, reward, terminated, truncated, info

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment.

        Parameters
        ----------
        seed : Optional[int], default=None
            Random seed
        options : Optional[Dict[str, Any]], default=None
            Additional options

        Returns
        -------
        observation : np.ndarray
            Initial observation
        info : dict
            Additional information
        """
        super().reset(seed=seed)

        self.current_step = 0
        self.cash = self.initial_cash
        self.position = 0.0
        self.entry_price = 0.0
        self.trades = []
        self.portfolio_values = [self.initial_cash]
        self.returns_history = []

        observation = self._get_observation()
        info = {
            'portfolio_value': self.initial_cash,
            'position': 0.0,
            'cash': self.initial_cash,
            'trades': 0,
            'cumulative_return': 0.0,
            'step': 0
        }

        return observation, info

    def render(self) -> None:
        """Print current state."""
        current_price = self.prices[self.current_step]
        portfolio_value = self.cash + self.position * current_price
        cumulative_return = (portfolio_value - self.initial_cash) / self.initial_cash

        print(f"Step: {self.current_step}/{len(self.prices)-1}")
        print(f"Price: ${current_price:.2f}")
        print(f"Position: {self.position:.2f} shares")
        print(f"Cash: ${self.cash:.2f}")
        print(f"Portfolio Value: ${portfolio_value:.2f}")
        print(f"Cumulative Return: {cumulative_return:.2%}")
        print(f"Trades: {len(self.trades)}")
        print("-" * 50)
