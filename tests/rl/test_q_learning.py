"""
Tests for Q-learning agent.
"""

import pytest
import numpy as np
import gymnasium as gym
from puffin.rl.q_learning import QLearningAgent, discretize_state


class TestDiscretizeState:
    """Test discretize_state function."""

    def test_discretize_state_1d(self):
        """Test 1D state discretization."""
        obs = np.array([0.5])
        bins = [np.linspace(0, 1, 5)]
        state = discretize_state(obs, bins)
        assert isinstance(state, int)
        assert 0 <= state < 4

    def test_discretize_state_2d(self):
        """Test 2D state discretization."""
        obs = np.array([0.5, 1.5])
        bins = [np.linspace(0, 1, 5), np.linspace(0, 2, 5)]
        state = discretize_state(obs, bins)
        assert isinstance(state, int)
        assert 0 <= state < 16  # 4 * 4

    def test_discretize_state_invalid_dims(self):
        """Test error on dimension mismatch."""
        obs = np.array([0.5])
        bins = [np.linspace(0, 1, 5), np.linspace(0, 2, 5)]
        with pytest.raises(ValueError, match="observation dim"):
            discretize_state(obs, bins)

    def test_discretize_state_edge_values(self):
        """Test edge value handling."""
        obs = np.array([0.0])
        bins = [np.linspace(0, 1, 5)]
        state = discretize_state(obs, bins)
        assert state == 0

        obs = np.array([1.0])
        state = discretize_state(obs, bins)
        assert 0 <= state < 4


class TestQLearningAgent:
    """Test QLearningAgent class."""

    def test_initialization(self):
        """Test agent initialization."""
        agent = QLearningAgent(n_states=10, n_actions=3)
        assert agent.n_states == 10
        assert agent.n_actions == 3
        assert agent.q_table.shape == (10, 3)
        assert np.all(agent.q_table == 0)

    def test_choose_action_exploration(self):
        """Test action selection during exploration."""
        agent = QLearningAgent(n_states=5, n_actions=3, epsilon=1.0)
        actions = [agent.choose_action(0) for _ in range(100)]
        assert all(0 <= a < 3 for a in actions)
        # With epsilon=1.0, should see variety in actions
        assert len(set(actions)) > 1

    def test_choose_action_exploitation(self):
        """Test action selection during exploitation."""
        agent = QLearningAgent(n_states=5, n_actions=3, epsilon=0.0)
        # Set best action for state 0
        agent._q_table[0, 2] = 10.0
        actions = [agent.choose_action(0) for _ in range(10)]
        # With epsilon=0.0, should always choose action 2
        assert all(a == 2 for a in actions)

    def test_choose_action_invalid_state(self):
        """Test error on invalid state."""
        agent = QLearningAgent(n_states=5, n_actions=3)
        with pytest.raises(ValueError, match="Invalid state"):
            agent.choose_action(10)
        with pytest.raises(ValueError, match="Invalid state"):
            agent.choose_action(-1)

    def test_update_q_table(self):
        """Test Q-table update."""
        agent = QLearningAgent(n_states=5, n_actions=3, lr=0.1, gamma=0.9)
        initial_q = agent.q_table[0, 1]

        # Update with positive reward
        agent.update(state=0, action=1, reward=10.0, next_state=1, done=False)
        updated_q = agent.q_table[0, 1]

        # Q-value should increase
        assert updated_q > initial_q

    def test_update_terminal_state(self):
        """Test update for terminal state."""
        agent = QLearningAgent(n_states=5, n_actions=3, lr=1.0, gamma=0.9)
        agent.update(state=0, action=1, reward=10.0, next_state=1, done=True)

        # For terminal state, target should be just the reward
        assert agent.q_table[0, 1] == 10.0

    def test_update_invalid_inputs(self):
        """Test error on invalid update inputs."""
        agent = QLearningAgent(n_states=5, n_actions=3)

        with pytest.raises(ValueError, match="Invalid state"):
            agent.update(state=10, action=0, reward=1.0, next_state=0)

        with pytest.raises(ValueError, match="Invalid action"):
            agent.update(state=0, action=5, reward=1.0, next_state=0)

        with pytest.raises(ValueError, match="Invalid next_state"):
            agent.update(state=0, action=0, reward=1.0, next_state=10)

    def test_get_policy(self):
        """Test policy extraction."""
        agent = QLearningAgent(n_states=5, n_actions=3)
        # Set different best actions for each state
        for state in range(5):
            agent._q_table[state, state % 3] = 10.0

        policy = agent.get_policy()
        assert len(policy) == 5
        for state in range(5):
            assert policy[state] == state % 3

    def test_train_frozen_lake(self):
        """Test training on FrozenLake environment."""
        env = gym.make('FrozenLake-v1', is_slippery=False)
        agent = QLearningAgent(n_states=16, n_actions=4, epsilon=1.0, epsilon_decay=0.99)

        rewards = agent.train(env, episodes=100, verbose=False)

        assert len(rewards) == 100
        # Should see some improvement over time
        assert np.mean(rewards[-10:]) >= np.mean(rewards[:10])
        # Epsilon should decay
        assert agent.epsilon < 1.0

    def test_save_load(self, tmp_path):
        """Test saving and loading Q-table."""
        agent = QLearningAgent(n_states=5, n_actions=3)
        agent._q_table[0, 0] = 42.0

        # Save
        filepath = tmp_path / "q_table.npy"
        agent.save(str(filepath))

        # Load into new agent
        new_agent = QLearningAgent(n_states=5, n_actions=3)
        new_agent.load(str(filepath))

        assert np.allclose(agent.q_table, new_agent.q_table)

    def test_load_wrong_shape(self, tmp_path):
        """Test error when loading wrong shape Q-table."""
        agent = QLearningAgent(n_states=5, n_actions=3)
        filepath = tmp_path / "q_table.npy"
        agent.save(str(filepath))

        # Try to load into different shape agent
        wrong_agent = QLearningAgent(n_states=10, n_actions=3)
        with pytest.raises(ValueError, match="shape"):
            wrong_agent.load(str(filepath))
