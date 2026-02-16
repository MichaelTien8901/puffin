"""
Deep Reinforcement Learning trading agent.

This example demonstrates:
1. RL environment setup
2. Deep Q-Network training
3. Agent evaluation
4. Risk-aware reward function
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import deque
import random

from puffin.data import YFinanceProvider
from puffin.risk import PortfolioRiskManager


class TradingEnvironment:
    """Simplified trading environment for RL."""

    def __init__(self, data: pd.DataFrame, initial_capital=100000):
        """Initialize trading environment."""
        self.data = data.reset_index(drop=True)
        self.initial_capital = initial_capital
        self.reset()

    def reset(self):
        """Reset environment to initial state."""
        self.current_step = 0
        self.cash = self.initial_capital
        self.position = 0
        self.equity = self.initial_capital
        return self._get_state()

    def _get_state(self):
        """Get current state observation."""
        if self.current_step >= len(self.data):
            return None

        # Simple state: last 5 returns + position
        lookback = 5
        start = max(0, self.current_step - lookback)

        returns = self.data['close'].pct_change().iloc[start:self.current_step+1].fillna(0)
        state = list(returns.tail(lookback))

        # Pad if necessary
        while len(state) < lookback:
            state.insert(0, 0)

        # Add position
        state.append(self.position / 100)  # Normalize

        return np.array(state)

    def step(self, action):
        """
        Execute action and return (next_state, reward, done).

        Actions: 0=hold, 1=buy, 2=sell
        """
        if self.current_step >= len(self.data) - 1:
            return None, 0, True

        current_price = self.data['close'].iloc[self.current_step]
        next_price = self.data['close'].iloc[self.current_step + 1]

        # Execute action
        if action == 1 and self.position == 0:  # Buy
            shares = int(self.cash / current_price)
            self.position = shares
            self.cash -= shares * current_price

        elif action == 2 and self.position > 0:  # Sell
            self.cash += self.position * current_price
            self.position = 0

        # Update state
        self.current_step += 1

        # Calculate reward
        new_equity = self.cash + self.position * next_price
        reward = (new_equity - self.equity) / self.equity  # % return
        self.equity = new_equity

        # Penalize excessive drawdown
        if self.equity < self.initial_capital * 0.9:
            reward -= 0.1

        done = self.current_step >= len(self.data) - 1

        return self._get_state(), reward, done


class SimpleDQNAgent:
    """Simplified DQN agent for demonstration."""

    def __init__(self, state_size, action_size):
        """Initialize DQN agent."""
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        # Simple Q-table (in production, use neural network)
        self.q_table = {}

    def act(self, state):
        """Choose action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        # Get Q-values for state
        state_key = tuple(np.round(state, 2))
        q_values = self.q_table.get(state_key, [0] * self.action_size)

        return np.argmax(q_values)

    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=32):
        """Train on batch of experiences."""
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in batch:
            state_key = tuple(np.round(state, 2))

            # Initialize Q-values if needed
            if state_key not in self.q_table:
                self.q_table[state_key] = [0] * self.action_size

            # Q-learning update
            target = reward
            if not done and next_state is not None:
                next_key = tuple(np.round(next_state, 2))
                next_q = self.q_table.get(next_key, [0] * self.action_size)
                target += 0.95 * max(next_q)

            self.q_table[state_key][action] = target

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def main():
    """Run deep RL training."""
    print("=" * 60)
    print("Deep Reinforcement Learning Trading Agent")
    print("=" * 60)

    # 1. Load data
    print("\n1. Loading training data...")
    provider = YFinanceProvider()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    data = provider.get_historical(
        ticker='AAPL',
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        interval='1d'
    )

    print(f"   Data shape: {data.shape}")

    # 2. Initialize environment and agent
    print("\n2. Initializing RL components...")

    env = TradingEnvironment(data)
    state_size = 6  # 5 returns + position
    action_size = 3  # hold, buy, sell

    agent = SimpleDQNAgent(state_size, action_size)

    # 3. Train agent
    print("\n3. Training agent...")

    episodes = 50
    batch_size = 32

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)

            if next_state is None:
                break

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                break

        # Train on batch
        agent.replay(batch_size)

        if (episode + 1) % 10 == 0:
            print(f"   Episode {episode+1}/{episodes}, "
                  f"Total Reward: {total_reward:.4f}, "
                  f"Epsilon: {agent.epsilon:.3f}, "
                  f"Final Equity: ${env.equity:,.2f}")

    # 4. Evaluate agent
    print("\n4. Evaluating trained agent...")

    agent.epsilon = 0  # Disable exploration

    state = env.reset()
    actions_taken = {'hold': 0, 'buy': 0, 'sell': 0}

    while True:
        action = agent.act(state)

        action_names = ['hold', 'buy', 'sell']
        actions_taken[action_names[action]] += 1

        next_state, reward, done = env.step(action)

        if next_state is None or done:
            break

        state = next_state

    final_return = (env.equity - env.initial_capital) / env.initial_capital

    print(f"\n   Initial capital: ${env.initial_capital:,.2f}")
    print(f"   Final equity: ${env.equity:,.2f}")
    print(f"   Total return: {final_return:.2%}")
    print(f"\n   Actions taken:")
    print(f"   - Hold: {actions_taken['hold']}")
    print(f"   - Buy: {actions_taken['buy']}")
    print(f"   - Sell: {actions_taken['sell']}")

    # 5. Risk analysis
    print("\n5. Risk analysis...")

    # Buy and hold comparison
    buy_hold_return = (data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0]

    print(f"   Buy-and-hold return: {buy_hold_return:.2%}")
    print(f"   RL agent return: {final_return:.2%}")
    print(f"   Outperformance: {(final_return - buy_hold_return):.2%}")

    print("\n" + "=" * 60)
    print("RL training complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
