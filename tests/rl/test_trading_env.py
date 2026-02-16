"""
Tests for custom trading environment.
"""

import pytest
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from puffin.rl.trading_env import TradingEnvironment


class TestTradingEnvironment:
    """Test TradingEnvironment class."""

    @pytest.fixture
    def simple_prices(self):
        """Simple price series for testing."""
        return np.array([100.0, 101.0, 102.0, 101.5, 103.0])

    @pytest.fixture
    def simple_features(self):
        """Simple features for testing."""
        return np.array([
            [0.1, 0.2],
            [0.2, 0.3],
            [0.3, 0.4],
            [0.2, 0.3],
            [0.4, 0.5]
        ])

    def test_initialization_prices_only(self, simple_prices):
        """Test initialization with prices only."""
        env = TradingEnvironment(prices=simple_prices)
        assert isinstance(env, gym.Env)
        assert len(env.prices) == 5
        assert env.initial_cash == 100000.0
        assert env.discrete_actions

    def test_initialization_with_features(self, simple_prices, simple_features):
        """Test initialization with features."""
        env = TradingEnvironment(prices=simple_prices, features=simple_features)
        assert env.features is not None
        assert env.features.shape == (5, 2)

    def test_initialization_invalid_prices(self):
        """Test error on invalid prices."""
        with pytest.raises(ValueError):
            TradingEnvironment(prices=np.array([]))

    def test_initialization_mismatched_lengths(self, simple_prices):
        """Test error on mismatched lengths."""
        features = np.array([[0.1, 0.2], [0.2, 0.3]])  # Wrong length
        with pytest.raises(ValueError, match="same length"):
            TradingEnvironment(prices=simple_prices, features=features)

    def test_action_space_discrete(self, simple_prices):
        """Test discrete action space."""
        env = TradingEnvironment(prices=simple_prices, discrete_actions=True)
        assert isinstance(env.action_space, spaces.Discrete)
        assert env.action_space.n == 3

    def test_action_space_continuous(self, simple_prices):
        """Test continuous action space."""
        env = TradingEnvironment(prices=simple_prices, discrete_actions=False)
        assert isinstance(env.action_space, spaces.Box)
        assert env.action_space.shape == (1,)

    def test_observation_space(self, simple_prices):
        """Test observation space dimensions."""
        env = TradingEnvironment(prices=simple_prices, lookback=1)
        assert isinstance(env.observation_space, spaces.Box)
        # price (1) + position + cash + unrealized_pnl
        assert env.observation_space.shape == (4,)

    def test_observation_space_with_features(self, simple_prices, simple_features):
        """Test observation space with features."""
        env = TradingEnvironment(
            prices=simple_prices,
            features=simple_features,
            lookback=1
        )
        # features (2) + price (1) + position + cash + unrealized_pnl
        assert env.observation_space.shape == (6,)

    def test_reset(self, simple_prices):
        """Test environment reset."""
        env = TradingEnvironment(prices=simple_prices)
        obs, info = env.reset()

        assert env.current_step == 0
        assert env.cash == env.initial_cash
        assert env.position == 0.0
        assert len(env.trades) == 0
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)
        assert info['portfolio_value'] == env.initial_cash

    def test_reset_with_seed(self, simple_prices):
        """Test reset with seed."""
        env = TradingEnvironment(prices=simple_prices)
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        assert np.allclose(obs1, obs2)

    def test_step_hold(self, simple_prices):
        """Test holding (no trade)."""
        env = TradingEnvironment(prices=simple_prices)
        env.reset()

        obs, reward, terminated, truncated, info = env.step(action=1)  # Hold

        assert env.current_step == 1
        assert env.position == 0.0
        assert len(env.trades) == 0
        assert not terminated

    def test_step_buy(self, simple_prices):
        """Test buying."""
        env = TradingEnvironment(prices=simple_prices, initial_cash=10000)
        env.reset()

        obs, reward, terminated, truncated, info = env.step(action=2)  # Buy

        assert env.position > 0
        assert env.cash < 10000
        assert len(env.trades) == 1
        assert env.trades[0]['action'] == 'buy'

    def test_step_sell(self, simple_prices):
        """Test selling."""
        env = TradingEnvironment(prices=simple_prices)
        env.reset()

        # First buy
        env.step(action=2)
        initial_cash = env.cash
        initial_position = env.position

        # Then sell
        env.step(action=0)

        assert env.position < initial_position
        assert env.cash > initial_cash
        assert len(env.trades) == 2
        assert env.trades[1]['action'] == 'sell'

    def test_step_termination(self, simple_prices):
        """Test episode termination."""
        env = TradingEnvironment(prices=simple_prices)
        env.reset()

        # Step through all prices
        for i in range(len(simple_prices) - 1):
            obs, reward, terminated, truncated, info = env.step(action=1)
            if i < len(simple_prices) - 2:
                assert not terminated
            else:
                assert terminated

    def test_commission_cost(self, simple_prices):
        """Test that commission is deducted."""
        env = TradingEnvironment(
            prices=simple_prices,
            initial_cash=10000,
            commission=0.01  # 1% commission
        )
        env.reset()

        initial_cash = env.cash
        env.step(action=2)  # Buy

        # Calculate expected cost
        price = simple_prices[0]
        max_shares = env.max_position
        trade_value = max_shares * price
        commission = trade_value * 0.01
        expected_cost = trade_value + commission

        assert np.isclose(env.cash, initial_cash - expected_cost, rtol=1e-5)

    def test_reward_pnl(self, simple_prices):
        """Test P&L reward calculation."""
        env = TradingEnvironment(prices=simple_prices, reward_type='pnl')
        env.reset()

        # Buy at first price
        _, reward1, _, _, _ = env.step(action=2)

        # Price goes up, should get positive reward
        _, reward2, _, _, _ = env.step(action=1)  # Hold

        # Reward should reflect price change
        assert isinstance(reward2, float)

    def test_reward_sharpe(self, simple_prices):
        """Test Sharpe-like reward calculation."""
        env = TradingEnvironment(prices=simple_prices, reward_type='sharpe')
        env.reset()

        rewards = []
        for _ in range(len(simple_prices) - 1):
            _, reward, terminated, truncated, _ = env.step(action=1)
            rewards.append(reward)
            if terminated:
                break

        # Should return rewards
        assert len(rewards) > 0

    def test_reward_risk_adjusted(self, simple_prices):
        """Test risk-adjusted reward calculation."""
        env = TradingEnvironment(prices=simple_prices, reward_type='risk_adjusted')
        env.reset()

        rewards = []
        for _ in range(len(simple_prices) - 1):
            _, reward, terminated, truncated, _ = env.step(action=1)
            rewards.append(reward)
            if terminated:
                break

        assert len(rewards) > 0

    def test_info_dict(self, simple_prices):
        """Test info dictionary contents."""
        env = TradingEnvironment(prices=simple_prices)
        env.reset()

        _, _, _, _, info = env.step(action=1)

        assert 'portfolio_value' in info
        assert 'position' in info
        assert 'cash' in info
        assert 'trades' in info
        assert 'cumulative_return' in info
        assert 'step' in info

    def test_continuous_actions(self, simple_prices):
        """Test continuous action space."""
        env = TradingEnvironment(prices=simple_prices, discrete_actions=False)
        env.reset()

        # Test different continuous actions
        env.step(action=0.5)  # Partial buy
        env.step(action=-0.3)  # Partial sell
        env.step(action=0.0)  # Hold

        # Should handle continuous actions
        assert True

    def test_lookback(self, simple_prices):
        """Test lookback window."""
        env = TradingEnvironment(prices=simple_prices, lookback=3)
        obs, _ = env.reset()

        # Observation should include lookback history
        # 3 prices + position + cash + unrealized_pnl = 6
        assert len(obs) == 6

    def test_max_position_limit(self, simple_prices):
        """Test max position limit."""
        env = TradingEnvironment(
            prices=simple_prices,
            max_position=0.5,
            discrete_actions=False
        )
        env.reset()

        # Try to buy more than max
        env.step(action=1.0)  # Should be clipped to max_position

        assert env.position <= 0.5

    def test_render(self, simple_prices, capsys):
        """Test render method."""
        env = TradingEnvironment(prices=simple_prices)
        env.reset()

        env.render()
        captured = capsys.readouterr()
        assert "Step:" in captured.out
        assert "Price:" in captured.out
        assert "Position:" in captured.out

    def test_full_episode(self, simple_prices):
        """Test a full episode with mixed actions."""
        env = TradingEnvironment(prices=simple_prices)
        obs, info = env.reset()

        actions = [2, 1, 0, 1]  # Buy, Hold, Sell, Hold
        total_reward = 0

        for action in actions:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        # Should complete without errors
        assert isinstance(total_reward, float)
        assert 'portfolio_value' in info

    def test_portfolio_value_tracking(self, simple_prices):
        """Test that portfolio values are tracked."""
        env = TradingEnvironment(prices=simple_prices)
        env.reset()

        # Execute some trades
        for _ in range(3):
            env.step(action=1)

        assert len(env.portfolio_values) > 0
        assert all(isinstance(pv, (int, float)) for pv in env.portfolio_values)
