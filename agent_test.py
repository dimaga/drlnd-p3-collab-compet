#!python
"""agent.py unit tests"""

import unittest
import numpy as np
from agent import Agent, OUNoise


class TestAgent(unittest.TestCase):
    """Test cases to verify agent module"""


    def test_agent_act(self):
        """Test how an agent can act"""

        agent = Agent(3, 5, 1)
        states = np.array([[1.0, 2.0, 3.0], [0.3, 2.0, 1.0]])

        actions1 = agent.act(states, False)
        self.assertEqual((2, 5), actions1.shape)

        actions2 = agent.act(states, False)
        self.assertTrue(np.allclose(actions2, actions1))

        actions3 = agent.act(states, True)
        self.assertFalse(np.allclose(actions2, actions3))


class TestQUNoise(unittest.TestCase):
    """Test QUNoise"""

    def test_sample(self):
        """Test how an agent can act"""

        noise = OUNoise(2, 0)

        sample = noise.sample()
        self.assertEqual((2,), sample.shape)
        self.assertNotAlmostEqual(sample[0], sample[1])



if __name__ == '__main__':
    unittest.main()
