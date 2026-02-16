"""Gymnasium-compatible trading environment for reinforcement learning."""

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYM = True
except ImportError:
    HAS_GYM = False


if HAS_GYM:
    class TradingEnv(gym.Env):
        """Trading environment for RL agents.

        Actions: 0=hold, 1=buy, 2=sell
        Observation: Feature vector (configurable)
        Reward: P&L or Sharpe-based
        """

        metadata = {"render_modes": []}

        def __init__(
            self,
            data: pd.DataFrame,
            features: pd.DataFrame | None = None,
            initial_capital: float = 100_000.0,
            max_position: int = 100,
            reward_type: str = "pnl",
        ):
            super().__init__()

            self.data = data.reset_index(drop=True)
            self.features = features.reset_index(drop=True) if features is not None else None
            self.initial_capital = initial_capital
            self.max_position = max_position
            self.reward_type = reward_type

            # Action space: hold, buy, sell
            self.action_space = spaces.Discrete(3)

            # Observation space
            n_features = features.shape[1] if features is not None else 5  # OHLCV
            # Features + position + cash_ratio
            obs_dim = n_features + 2
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )

            self._reset_state()

        def _reset_state(self):
            self.current_step = 0
            self.cash = self.initial_capital
            self.position = 0
            self.portfolio_values = [self.initial_capital]

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self._reset_state()
            return self._get_observation(), {}

        def _get_observation(self):
            if self.features is not None:
                row = self.features.iloc[self.current_step].values
            else:
                row = self.data.iloc[self.current_step][
                    ["Open", "High", "Low", "Close", "Volume"]
                ].values

            # Normalize features
            obs = np.nan_to_num(row, nan=0.0).astype(np.float32)

            # Add position and cash ratio
            portfolio_value = self._portfolio_value()
            position_ratio = np.float32(self.position * self._current_price() / portfolio_value if portfolio_value > 0 else 0)
            cash_ratio = np.float32(self.cash / portfolio_value if portfolio_value > 0 else 0)

            return np.append(obs, [position_ratio, cash_ratio]).astype(np.float32)

        def _current_price(self) -> float:
            return float(self.data.iloc[self.current_step]["Close"])

        def _portfolio_value(self) -> float:
            return self.cash + self.position * self._current_price()

        def step(self, action: int):
            prev_value = self._portfolio_value()
            price = self._current_price()

            # Execute action
            if action == 1 and self.position < self.max_position:  # Buy
                qty = min(
                    self.max_position - self.position,
                    int(self.cash * 0.1 / price) if price > 0 else 0,
                )
                if qty > 0:
                    self.cash -= qty * price
                    self.position += qty

            elif action == 2 and self.position > 0:  # Sell
                self.cash += self.position * price
                self.position = 0

            self.current_step += 1
            done = self.current_step >= len(self.data) - 1
            truncated = False

            current_value = self._portfolio_value()
            self.portfolio_values.append(current_value)

            # Reward
            if self.reward_type == "pnl":
                reward = (current_value - prev_value) / prev_value
            elif self.reward_type == "sharpe":
                returns = np.diff(self.portfolio_values) / np.array(self.portfolio_values[:-1])
                if len(returns) > 1 and np.std(returns) > 0:
                    reward = np.mean(returns) / np.std(returns)
                else:
                    reward = 0.0
            else:
                reward = current_value - prev_value

            obs = self._get_observation() if not done else np.zeros(
                self.observation_space.shape, dtype=np.float32
            )

            info = {
                "portfolio_value": current_value,
                "position": self.position,
                "cash": self.cash,
            }

            return obs, float(reward), done, truncated, info
