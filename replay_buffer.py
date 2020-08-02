#!python
"""Contains ReplayBuffer that collects agent experience for batch learning"""

import random
from collections import deque
import numpy as np
import torch

# pylint: disable=no-member
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Experience:
    """Experience of the local agent"""

    # pylint: disable=too-few-public-methods

    __slots__ = [
        "state",
        "action",
        "all_state",
        "reward",
        "next_state",
        "all_next_state",
        "done"
    ]

    def __init__(self):
        self.state = None
        self.action = None
        self.all_state = None
        self.reward = None
        self.next_state = None
        self.all_next_state = None
        self.done = None


def _experience_to_torch_device(experiences):
    """Returns torch.Tensor-s filled with components of Experience list

    :param experiences: agent experience stored in Experience tuples
    :return: Experience of torch tensors"""

    torch_experiences = Experience()

    torch_experiences.state = torch.from_numpy(np.vstack(
        [e.state for e in experiences if e is not None])).float().to(DEVICE)

    torch_experiences.action = torch.from_numpy(np.vstack(
        [e.action for e in experiences if e is not None])).float().to(DEVICE)

    torch_experiences.all_state = torch.from_numpy(np.vstack(
        [e.all_state for e in experiences if e is not None])).float().to(DEVICE)

    torch_experiences.reward = torch.from_numpy(np.vstack(
        [e.reward for e in experiences if e is not None])).float().to(DEVICE)

    torch_experiences.next_state = torch.from_numpy(np.vstack(
        [e.next_state for e in experiences if e is not None])).float().to(
            DEVICE)

    torch_experiences.all_next_state = torch.from_numpy(np.vstack(
        [e.all_next_state for e in experiences if e is not None])).float().to(
            DEVICE)

    torch_experiences.done = torch.from_numpy(np.vstack(
        [e.done for e in experiences if e is not None]).astype(
            np.uint8)).float().to(DEVICE)

    return torch_experiences


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
        """ Add experiences from two agents
        :param states: state vectors, corresponding to two different agents
        :param actions: actions, conducted in two different agents
        :param info: information about next states, rewards and completion in
        two different agents
        """

        assert 2 == len(states)
        assert 2 == len(actions)
        assert 2 == len(info.rewards)
        assert 2 == len(info.vector_observations)
        assert 2 == len(info.local_done)

        for i in range(2):
            experience = Experience()

            experience.state = states[i]
            experience.action = actions[i]

            experience.all_state = np.concatenate((
                states[i],
                states[(i + 1) % 2]))

            experience.reward = info.rewards[i]
            experience.next_state = info.vector_observations[i]

            experience.all_next_state = np.concatenate((
                info.vector_observations[i],
                info.vector_observations[(i + 1) % 2]))

            experience.done = info.local_done[i]

            self.__memory.append(experience)


    def sample(self):
        """
        Randomly sample a batch of experiences from memory
        :return: torch tensors packed into Experience class
        """
        experiences = random.sample(self.__memory, k=self.__batch_size)
        return _experience_to_torch_device(experiences)


    def __len__(self):
        """:return:current replay buffer size"""
        return len(self.__memory)
