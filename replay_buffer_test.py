#!python
"""ReplayBuffer unit tests"""

import unittest
import numpy as np
from replay_buffer import ReplayBuffer
from environment import InfoStub

class TestReplayBuffer(unittest.TestCase):
    """Test cases to verify ReplayBuffer class"""

    def test_empty_replay_buffer(self):
        """Empty replay buffer must have zero length"""
        replay = ReplayBuffer(buffer_size=10, batch_size=3, seed=0)
        self.assertEqual(0, len(replay))


    def test_replay_buffer(self):
        """Test adding something to the replay buffer"""

        replay = ReplayBuffer(buffer_size=10, batch_size=2, seed=0)

        states = np.array([ # agents, state vector
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ])

        actions = np.array([ # agents, action vector
            [1.0, 2.0],
            [0.5, 0.2]
        ])

        info = InfoStub()

        info.vector_observations = np.array([ # agents, state vector
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

        info.rewards = np.array([1.0, 1.0])
        info.local_done = np.array([False, False], np.bool_)

        replay.add(states, actions, info)

        self.assertEqual(2, len(replay))

        torch_experiences = replay.sample()

        self.assertEqual((2, 3), torch_experiences.state.size())
        self.assertEqual((2, 2), torch_experiences.action.size())
        self.assertEqual((2, 6), torch_experiences.all_state.size())
        self.assertEqual((2, 1), torch_experiences.reward.size())
        self.assertEqual((2, 3), torch_experiences.next_state.size())
        self.assertEqual((2, 6), torch_experiences.all_next_state.size())
        self.assertEqual((2, 1), torch_experiences.done.size())



if __name__ == '__main__':
    unittest.main()
