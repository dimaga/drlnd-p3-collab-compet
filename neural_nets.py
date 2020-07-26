#!python
"""Actor and critic neural network implementations"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Create an instance of Actor neural network, which takes state
        vector as the input and returns action vector as the output
        :param state_size: dimensionality of the state vector
        :param action_size: dimensionality of the action vector
        :param seed: random seen to reproduce results
        """
        super(Actor, self).__init__()

        fc_units = 256

        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc_units)
        self.batch_norm1 = nn.BatchNorm1d(fc_units)
        self.fc2 = nn.Linear(fc_units, action_size)
        self.__reset_parameters()


    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""

        # pylint: disable=arguments-differ

        state = F.relu(self.batch_norm1(self.fc1(state)))
        return F.tanh(self.fc2(state))


    def __reset_parameters(self):
        self.fc1.weight.data.uniform_(*_hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed):
        """Create an instance of Cricit neural network, which takes state
        vector and action vector as the input and returns the value of the
        (state, action pair)
        :param state_size: dimensionality of the state vector
        :param action_size: dimensionality of the action vector
        :param seed: random seen to reproduce results
        """
        super(Critic, self).__init__()

        fcs1_units = 256
        fc2_units = 256
        fc3_units = 128

        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units + action_size, fc2_units)
        self.batch_norm2 = nn.BatchNorm1d(fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, 1)
        self.__reset_parameters()


    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs ->
        Q-values."""

        # pylint: disable=no-member, arguments-differ

        state = F.leaky_relu(self.fcs1(state))
        data = torch.cat((state, action), dim=1)
        data = F.leaky_relu(self.batch_norm2(self.fc2(data)))
        data = F.leaky_relu(self.fc3(data))
        return self.fc4(data)


    def __reset_parameters(self):
        self.fcs1.weight.data.uniform_(*_hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*_hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*_hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)
