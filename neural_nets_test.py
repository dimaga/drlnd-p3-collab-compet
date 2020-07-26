#!python
"""Test actor and critic neural networks"""

import unittest
import torch
from replay_buffer import DEVICE
from neural_nets import Actor, Critic


class TestNeuralNets(unittest.TestCase):
    """Test cases to verify Actor and Critic"""

    def test_actor_single_input(self):
        """Test actor forward() against a single state vector"""
        actor = Actor(state_size=5, action_size=3, seed=0).to(DEVICE)

        state = torch.Tensor([[0.1, 0.5, 1.0, 0.1, 0.5]]).to(DEVICE)

        actor.eval()
        action = actor.forward(state).to(DEVICE)
        self.assertEqual((1, 3), action.size())


    def test_actor_multiple_input(self):
        """Test actor forward() against multiple state vectors"""
        actor = Actor(state_size=3, action_size=2, seed=0).to(DEVICE)

        states = torch.Tensor([
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0]
        ]).to(DEVICE)

        actor.eval()
        actions = actor.forward(states).to(DEVICE)
        self.assertEqual((5, 2), actions.size())


    def test_critic_single_input(self):
        """Test critic forward() against a single state and action vectors"""
        critic = Critic(state_size=3, action_size=2, seed=0).to(DEVICE)

        state = torch.Tensor([[1.0, 0.1, 0.5]]).to(DEVICE)
        action = torch.Tensor([[0.1, 2.1]]).to(DEVICE)

        critic.eval()
        value = critic.forward(state, action).to(DEVICE)
        self.assertEqual((1, 1), value.size())


    def test_critic_multiple_input(self):
        """Test cricit forward() against multiple state and action vectors"""
        critic = Critic(state_size=2, action_size=3, seed=0).to(DEVICE)

        states = torch.Tensor([
            [-1.0, 2.0],
            [0.5, 0.1]
        ]).to(DEVICE)

        actions = torch.Tensor([
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0]
        ]).to(DEVICE)

        critic.eval()
        values = critic.forward(states, actions).to(DEVICE)
        self.assertEqual((2, 1), values.size())
