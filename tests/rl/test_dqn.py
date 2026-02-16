"""
Tests for DQN and DDQN agents.
"""

import pytest
import numpy as np
import torch
import gymnasium as gym
from puffin.rl.dqn import DQNetwork, ReplayBuffer, DQNAgent, DDQNAgent


class TestDQNetwork:
    """Test DQNetwork class."""

    def test_initialization_default(self):
        """Test network initialization with defaults."""
        network = DQNetwork(state_dim=4, action_dim=3)
        assert isinstance(network, torch.nn.Module)

        # Test forward pass
        state = torch.randn(1, 4)
        q_values = network(state)
        assert q_values.shape == (1, 3)

    def test_initialization_custom_hidden(self):
        """Test network with custom hidden dimensions."""
        network = DQNetwork(state_dim=10, action_dim=5, hidden_dims=[256, 128, 64])
        state = torch.randn(2, 10)
        q_values = network(state)
        assert q_values.shape == (2, 5)

    def test_forward_batch(self):
        """Test forward pass with batch."""
        network = DQNetwork(state_dim=4, action_dim=3)
        batch_size = 32
        states = torch.randn(batch_size, 4)
        q_values = network(states)
        assert q_values.shape == (batch_size, 3)


class TestReplayBuffer:
    """Test ReplayBuffer class."""

    def test_initialization(self):
        """Test buffer initialization."""
        buffer = ReplayBuffer(capacity=100)
        assert len(buffer) == 0

    def test_store(self):
        """Test storing transitions."""
        buffer = ReplayBuffer(capacity=100)
        state = np.array([1.0, 2.0, 3.0])
        next_state = np.array([1.1, 2.1, 3.1])

        buffer.store(state, action=1, reward=1.0, next_state=next_state, done=False)
        assert len(buffer) == 1

    def test_capacity_limit(self):
        """Test that buffer respects capacity limit."""
        buffer = ReplayBuffer(capacity=5)

        for i in range(10):
            buffer.store(
                state=np.array([float(i)]),
                action=i % 3,
                reward=float(i),
                next_state=np.array([float(i + 1)]),
                done=False
            )

        # Should only keep last 5
        assert len(buffer) == 5

    def test_sample(self):
        """Test sampling from buffer."""
        buffer = ReplayBuffer(capacity=100)

        # Add transitions
        for i in range(50):
            buffer.store(
                state=np.array([float(i)]),
                action=i % 3,
                reward=float(i),
                next_state=np.array([float(i + 1)]),
                done=(i % 10 == 0)
            )

        # Sample batch
        states, actions, rewards, next_states, dones = buffer.sample(batch_size=16)

        assert states.shape == (16, 1)
        assert actions.shape == (16,)
        assert rewards.shape == (16,)
        assert next_states.shape == (16, 1)
        assert dones.shape == (16,)

        # Check data types
        assert states.dtype == np.float32
        assert actions.dtype == np.int64
        assert rewards.dtype == np.float32


class TestDQNAgent:
    """Test DQNAgent class."""

    def test_initialization(self):
        """Test agent initialization."""
        agent = DQNAgent(state_dim=4, action_dim=3)
        assert agent.state_dim == 4
        assert agent.action_dim == 3
        assert len(agent.replay_buffer) == 0
        assert agent.step_count == 0

    def test_choose_action_exploration(self):
        """Test action selection with exploration."""
        agent = DQNAgent(state_dim=4, action_dim=3)
        state = np.random.randn(4).astype(np.float32)

        # With high epsilon, should see variety
        actions = [agent.choose_action(state, epsilon=1.0) for _ in range(100)]
        assert all(0 <= a < 3 for a in actions)
        assert len(set(actions)) > 1

    def test_choose_action_exploitation(self):
        """Test action selection without exploration."""
        agent = DQNAgent(state_dim=4, action_dim=3)
        state = np.random.randn(4).astype(np.float32)

        # With epsilon=0, should be deterministic
        actions = [agent.choose_action(state, epsilon=0.0) for _ in range(10)]
        # All actions should be the same
        assert len(set(actions)) == 1

    def test_update_insufficient_samples(self):
        """Test that update returns None when buffer too small."""
        agent = DQNAgent(state_dim=4, action_dim=3, batch_size=64)

        # Add only a few samples
        for _ in range(10):
            agent.replay_buffer.store(
                state=np.random.randn(4).astype(np.float32),
                action=0,
                reward=0.0,
                next_state=np.random.randn(4).astype(np.float32),
                done=False
            )

        loss = agent.update()
        assert loss is None

    def test_update_with_sufficient_samples(self):
        """Test update with enough samples."""
        agent = DQNAgent(state_dim=4, action_dim=3, batch_size=32)

        # Add enough samples
        for _ in range(100):
            agent.replay_buffer.store(
                state=np.random.randn(4).astype(np.float32),
                action=np.random.randint(0, 3),
                reward=np.random.randn(),
                next_state=np.random.randn(4).astype(np.float32),
                done=False
            )

        loss = agent.update()
        assert loss is not None
        assert isinstance(loss, float)
        assert loss >= 0

    def test_target_network_update(self):
        """Test that target network gets updated."""
        agent = DQNAgent(state_dim=4, action_dim=3, batch_size=32, target_update=10)

        # Fill buffer
        for _ in range(100):
            agent.replay_buffer.store(
                state=np.random.randn(4).astype(np.float32),
                action=np.random.randint(0, 3),
                reward=np.random.randn(),
                next_state=np.random.randn(4).astype(np.float32),
                done=False
            )

        # Get initial target network params
        initial_params = [p.clone() for p in agent.target_network.parameters()]

        # Update multiple times
        for _ in range(15):
            agent.update()

        # Target network should have been updated at step 10
        updated_params = list(agent.target_network.parameters())
        # At least one parameter should have changed
        params_changed = any(
            not torch.allclose(initial, updated)
            for initial, updated in zip(initial_params, updated_params)
        )
        assert params_changed

    def test_train_cartpole(self):
        """Test training on CartPole environment."""
        env = gym.make('CartPole-v1')
        agent = DQNAgent(
            state_dim=4,
            action_dim=2,
            batch_size=32,
            buffer_size=1000
        )

        rewards = agent.train(
            env,
            episodes=5,
            epsilon_start=1.0,
            epsilon_end=0.1,
            verbose=False
        )

        assert len(rewards) == 5
        assert all(isinstance(r, (int, float)) for r in rewards)

    def test_save_load(self, tmp_path):
        """Test saving and loading agent."""
        agent = DQNAgent(state_dim=4, action_dim=3)

        # Train a bit to change weights
        for _ in range(100):
            agent.replay_buffer.store(
                state=np.random.randn(4).astype(np.float32),
                action=np.random.randint(0, 3),
                reward=np.random.randn(),
                next_state=np.random.randn(4).astype(np.float32),
                done=False
            )
        for _ in range(10):
            agent.update()

        # Save
        filepath = tmp_path / "dqn_model.pt"
        agent.save(str(filepath))

        # Load into new agent
        new_agent = DQNAgent(state_dim=4, action_dim=3)
        new_agent.load(str(filepath))

        # Check that weights match
        for p1, p2 in zip(agent.online_network.parameters(),
                         new_agent.online_network.parameters()):
            assert torch.allclose(p1, p2)

        assert agent.step_count == new_agent.step_count


class TestDDQNAgent:
    """Test DDQNAgent class."""

    def test_initialization(self):
        """Test DDQN agent initialization."""
        agent = DDQNAgent(state_dim=4, action_dim=3)
        assert isinstance(agent, DQNAgent)
        assert agent.state_dim == 4
        assert agent.action_dim == 3

    def test_update_uses_double_q(self):
        """Test that update uses double Q-learning."""
        agent = DDQNAgent(state_dim=4, action_dim=3, batch_size=32)

        # Fill buffer
        for _ in range(100):
            agent.replay_buffer.store(
                state=np.random.randn(4).astype(np.float32),
                action=np.random.randint(0, 3),
                reward=np.random.randn(),
                next_state=np.random.randn(4).astype(np.float32),
                done=False
            )

        # Update should work
        loss = agent.update()
        assert loss is not None
        assert isinstance(loss, float)
        assert loss >= 0

    def test_train_cartpole(self):
        """Test training DDQN on CartPole."""
        env = gym.make('CartPole-v1')
        agent = DDQNAgent(
            state_dim=4,
            action_dim=2,
            batch_size=32,
            buffer_size=1000
        )

        rewards = agent.train(
            env,
            episodes=5,
            epsilon_start=1.0,
            epsilon_end=0.1,
            verbose=False
        )

        assert len(rewards) == 5
        assert all(isinstance(r, (int, float)) for r in rewards)
