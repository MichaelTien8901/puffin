"""
Tabular Q-learning implementation for reinforcement learning trading.
"""

import numpy as np
from typing import Optional, List, Tuple


def discretize_state(observation: np.ndarray, bins: List[np.ndarray]) -> int:
    """
    Convert continuous observation to discrete state index.

    Parameters
    ----------
    observation : np.ndarray
        Continuous observation vector
    bins : List[np.ndarray]
        List of bin edges for each dimension

    Returns
    -------
    int
        Discrete state index

    Examples
    --------
    >>> obs = np.array([0.5, 1.2])
    >>> bins = [np.linspace(0, 1, 5), np.linspace(0, 2, 5)]
    >>> state = discretize_state(obs, bins)
    """
    if len(observation) != len(bins):
        raise ValueError(f"observation dim ({len(observation)}) != bins dim ({len(bins)})")

    # Digitize each dimension
    indices = []
    for i, (obs_val, bin_edges) in enumerate(zip(observation, bins)):
        # digitize returns 1-indexed, subtract 1 for 0-indexing
        # clip to valid range
        idx = np.digitize(obs_val, bin_edges) - 1
        idx = np.clip(idx, 0, len(bin_edges) - 2)
        indices.append(idx)

    # Convert multi-dimensional index to single state index
    # Use row-major ordering
    state = 0
    multiplier = 1
    for i in reversed(range(len(indices))):
        state += indices[i] * multiplier
        multiplier *= (len(bins[i]) - 1)

    return state


class QLearningAgent:
    """
    Tabular Q-learning agent for discrete state/action spaces.

    Parameters
    ----------
    n_states : int
        Number of discrete states
    n_actions : int, default=3
        Number of discrete actions (e.g., buy/hold/sell)
    lr : float, default=0.1
        Learning rate (alpha)
    gamma : float, default=0.99
        Discount factor
    epsilon : float, default=1.0
        Initial exploration rate
    epsilon_decay : float, default=0.995
        Epsilon decay factor per episode
    epsilon_min : float, default=0.01
        Minimum epsilon value

    Attributes
    ----------
    q_table : np.ndarray
        Q-value table of shape (n_states, n_actions)
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int = 3,
        lr: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Initialize Q-table with zeros
        self._q_table = np.zeros((n_states, n_actions))

    @property
    def q_table(self) -> np.ndarray:
        """Get the Q-table."""
        return self._q_table

    def choose_action(self, state: int) -> int:
        """
        Choose action using epsilon-greedy policy.

        Parameters
        ----------
        state : int
            Current state index

        Returns
        -------
        int
            Selected action index
        """
        if state < 0 or state >= self.n_states:
            raise ValueError(f"Invalid state {state}, must be in [0, {self.n_states})")

        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(0, self.n_actions)
        else:
            # Exploit: best known action
            return int(np.argmax(self._q_table[state]))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool = False
    ) -> None:
        """
        Update Q-table using Q-learning update rule.

        Q(s,a) <- Q(s,a) + lr * [reward + gamma * max_a' Q(s',a') - Q(s,a)]

        Parameters
        ----------
        state : int
            Current state
        action : int
            Action taken
        reward : float
            Reward received
        next_state : int
            Next state
        done : bool, default=False
            Whether episode is terminated
        """
        if state < 0 or state >= self.n_states:
            raise ValueError(f"Invalid state {state}")
        if action < 0 or action >= self.n_actions:
            raise ValueError(f"Invalid action {action}")
        if next_state < 0 or next_state >= self.n_states:
            raise ValueError(f"Invalid next_state {next_state}")

        # Current Q-value
        current_q = self._q_table[state, action]

        # Max Q-value for next state (0 if terminal)
        if done:
            max_next_q = 0.0
        else:
            max_next_q = np.max(self._q_table[next_state])

        # TD target
        target = reward + self.gamma * max_next_q

        # Q-learning update
        self._q_table[state, action] = current_q + self.lr * (target - current_q)

    def train(
        self,
        env,
        episodes: int = 1000,
        max_steps: Optional[int] = None,
        verbose: bool = False
    ) -> List[float]:
        """
        Train the agent on an environment.

        Parameters
        ----------
        env : gymnasium.Env
            Environment with discrete observation (or pre-discretized)
        episodes : int, default=1000
            Number of training episodes
        max_steps : Optional[int], default=None
            Maximum steps per episode
        verbose : bool, default=False
            Whether to print progress

        Returns
        -------
        List[float]
            List of total rewards per episode

        Examples
        --------
        >>> import gymnasium as gym
        >>> env = gym.make('FrozenLake-v1')
        >>> agent = QLearningAgent(n_states=16, n_actions=4)
        >>> rewards = agent.train(env, episodes=1000)
        """
        episode_rewards = []

        for episode in range(episodes):
            state, info = env.reset()

            # Handle continuous states (assume state is already discretized)
            if isinstance(state, np.ndarray):
                if state.size == 1:
                    state = int(state[0])
                elif state.size > 1:
                    raise ValueError(
                        "Multi-dimensional states must be discretized before training. "
                        "Use discretize_state() function."
                    )

            total_reward = 0.0
            done = False
            step = 0

            while not done:
                # Choose and take action
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Handle state conversion
                if isinstance(next_state, np.ndarray):
                    if next_state.size == 1:
                        next_state = int(next_state[0])

                # Update Q-table
                self.update(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward
                step += 1

                if max_steps is not None and step >= max_steps:
                    break

            episode_rewards.append(total_reward)

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # Verbose output
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode + 1}/{episodes}, "
                      f"Avg Reward (last 100): {avg_reward:.2f}, "
                      f"Epsilon: {self.epsilon:.3f}")

        return episode_rewards

    def get_policy(self) -> np.ndarray:
        """
        Get the deterministic policy derived from Q-table.

        Returns
        -------
        np.ndarray
            Array of shape (n_states,) with best action for each state
        """
        return np.argmax(self._q_table, axis=1)

    def save(self, filepath: str) -> None:
        """Save Q-table to file."""
        np.save(filepath, self._q_table)

    def load(self, filepath: str) -> None:
        """Load Q-table from file."""
        self._q_table = np.load(filepath)
        if self._q_table.shape != (self.n_states, self.n_actions):
            raise ValueError(
                f"Loaded Q-table shape {self._q_table.shape} "
                f"doesn't match expected ({self.n_states}, {self.n_actions})"
            )
