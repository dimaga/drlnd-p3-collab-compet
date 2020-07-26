#!python
"""Contains ReplayBuffer that collects agent experience for batch learning"""

import random
from collections import namedtuple, deque
import numpy as np
import torch

# pylint: disable=no-member
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


Experience = namedtuple(
    "Experience",
    field_names=["state", "action", "reward", "next_state", "done"])


def _experience_to_torch_device(experiences):
    """Returns torch.Tensor-s filled with components of Experience list

    :param experiences: agent experience stored in Experience tuples
    :return: (states, actions, rewards, next_states, dones) as torch tensors"""

    states = torch.from_numpy(np.vstack(
        [e.state for e in experiences if e is not None])).float().to(DEVICE)

    actions = torch.from_numpy(np.vstack(
        [e.action for e in experiences if e is not None])).float().to(DEVICE)

    rewards = torch.from_numpy(np.vstack(
        [e.reward for e in experiences if e is not None])).float().to(DEVICE)

    next_states = torch.from_numpy(np.vstack(
        [e.next_state for e in experiences if e is not None])).float().to(
            DEVICE)

    dones = torch.from_numpy(np.vstack(
        [e.done for e in experiences if e is not None]).astype(
            np.uint8)).float().to(DEVICE)

    return (states, actions, rewards, next_states, dones)


class ReplayBuffer:
    """Collects user experience from multiple agents in order to sample random
    batches and train neural network more efficiently"""

    def __init__(self, buffer_size, batch_size, seed):
        """Creates ReplayBuffer instance
        :param buffer_size: size of replay buffer, old experience will be
        forgotten
        :param batch_size: size of sample, must be less than buffer_size
        :param seed: random seed value to reproduce training results
        """

        self.__memory = deque(maxlen=buffer_size)
        self.__batch_size = batch_size
        random.seed(seed)


    def add(self, states, actions, info):
        """ Add experiences from parallel environments
        :param states: state vectors, corresponding to different environments
        :param actions: actions, conducted in different environments
        :param info: information about next states, rewards and completion in
        different environments
        """

        for state, action, reward, next_state, done in zip(
                states,
                actions,
                info.rewards,
                info.vector_observations,
                info.local_done):

            experience = Experience(state, action, reward, next_state, done)
            self.__memory.append(experience)


    def sample(self):
        """
        Randomly sample a batch of experiences from memory
        :return: (states, actions, rewards, next_states, dones) as torch tensors
        """
        experiences = random.sample(self.__memory, k=self.__batch_size)
        return _experience_to_torch_device(experiences)


    def __len__(self):
        """:return:current replay buffer size"""
        return len(self.__memory)
