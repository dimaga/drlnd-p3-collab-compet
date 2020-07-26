#!python
"""environment.py unit tests"""

import unittest
import numpy as np
from neural_nets import Actor, Critic
from environment import EnvBase, InfoStub


class AgentOnPlane:
    """Agent steps in the direction of the goal at maximum velocity possible.
    This is an Agent stub to unit test environment.py"""

    def __init__(self, max_velocity, state_size, action_size):
        """Creates an agent to train and test its multiple copies
        :param max_velocity: maximum velocity of the agent
        :param state_size: dimensionality of the state vector
        :param action_size: dimensionality of the action vector"""

        self.__max_velocity = max_velocity
        self.actor_local = Actor(state_size, action_size, 0)
        self.critic_local = Critic(state_size, action_size, 0)
        self.steps = []
        self.reset_calls = 0


    def step(self, states, actions, env_info):
        """Performs a training step of the agent

        :param states: current states of environments:
        states[agent_idx][0] - x position of an agent
        states[agent_idx][1] - y position of an agent
        states[agent_idx][2] - x position of a goal
        states[agent_idx][3] - y position of a goal

        :param actions: actions which were taken by the agent upon states.
        Format:
        actions[agent_idx][0] - x component of an agent velocity vector
        actions[agent_idx][1] - y component of an agent velocity vector

        :param env_info: InfoStub of agent states after applying actions"""
        self.steps.append((states, actions, env_info))


    def reset(self):
        """The method is called in the beginning of each episode"""
        self.reset_calls += 1


    def act(self, states, _):
        """Returns actions to respond to states. This agent stub
        approaches the goal by the nearest trajectory.

        :param states: current states of environments:
        states[agent_idx][0] - x position of an agent
        states[agent_idx][1] - y position of an agent
        states[agent_idx][2] - x position of a goal
        states[agent_idx][3] - y position of a goal

        :return: actions taken by the agent upon states.
        Format:
        actions[agent_idx][0] - x component of an agent velocity vector
        actions[agent_idx][1] - y component of an agent velocity vector"""

        diffs = states[:, 2:] - states[:, :2]
        dists = np.linalg.norm(diffs, axis=1).reshape(-1, 1)
        mask = (dists < self.__max_velocity).reshape(-1, 1)

        step = mask.astype(np.float)
        step += (~mask).astype(np.float) / (dists + 1e-10) * self.__max_velocity
        return step * diffs


class EnvPlane(EnvBase):
    """World, consisting of an agent 2D position and a goal position."""

    def __init__(self, n_agents, goal_x, goal_y, on_reset_event):
        """This is an EnvBase stub implementation to unit test environment.py

        :param n_agents: number of agents, which are trained and tested in
        parallel

        :param goal_x: x-coordinate of a goal
        :param goal_y: y-coordinate of a goal

        :param on_reset_event: an event which is called upon every _reset() call
        to adjust state vector for more diverse unit testing scenarios"""
        super(EnvPlane, self).__init__(n_agents)

        self.__goal = np.array([goal_x, goal_y])
        self.__states = np.zeros((self.num_agents, 4))
        self.__on_reset_event = on_reset_event
        self.train_mode = None
        self.episode = 0


    @property
    def action_size(self):
        """An agent action vector dimension"""
        return 2


    @property
    def state_size(self):
        """A state dimension of the environment"""
        return self.__states.shape[1]


    def _step(self, actions):
        """Apply agent actions to environments to produce new actions, rewards;
        and estimate which environments are done.

        :param actions: actions taken by an agent upon states.
        Format:
        actions[agent_idx][0] - x component of an agent velocity vector
        actions[agent_idx][1] - y component of an agent velocity vector

        :return: InfoStub of agent states after applying actions"""

        self.__states[:, :2] += actions

        info = InfoStub()
        info.vector_observations = self.__states.copy()

        diffs = (info.vector_observations[:, :2]
                 - info.vector_observations[:, 2:])

        dists = np.linalg.norm(diffs, axis=1)

        info.rewards = (dists < 0.1).reshape(-1) * 2.0 - 1.0
        info.local_done = info.rewards > 0

        return info


    def _reset(self, train_mode):
        """Resets environment to restart an episode

        :param train_mode: If true, this is a training mode

        :return: initial states of environments:
        states[agent_idx][0] - x position of an agent
        states[agent_idx][1] - y position of an agent
        states[agent_idx][2] - x position of a goal
        states[agent_idx][3] - y position of a goal"""

        self.train_mode = train_mode

        self.__states[:, :2].fill(0.0)
        self.__states[:, 2:] = self.__goal
        self.__on_reset_event(self, self.__states)

        return self.__states.copy()


class TestEnvironment(unittest.TestCase):
    """Test cases to verify environment module"""

    def setUp(self):
        """The method is called before any test case"""
        self.__test_episode = 0
        self.__train_episode = 0


    def test_environment_test(self):
        """Unit test EnvBase.test()"""

        env = EnvPlane(3, 5.0, 0.0, self.__on_reset_test)
        agent = AgentOnPlane(1.0, env.action_size, env.state_size)
        env.test(agent, 2)

        self.assertFalse(env.train_mode)
        self.assertIsNotNone(env.train_mode)
        self.assertAlmostEqual(-5.5 / 3.0, env.avg_score)
        self.assertAlmostEqual(-4.0 / 3.0, env.last_score)


    def test_environment_train(self):
        """Unit test EnvBase.train()"""

        env = EnvPlane(2, 0.0, 5.0, self.__on_reset_train)
        agent = AgentOnPlane(0.5, env.action_size, env.state_size)
        env.train(agent, 3)

        self.assertTrue(env.train_mode)
        self.assertEqual(10 * 3, len(agent.steps))
        self.assertEqual(3, agent.reset_calls)

        self.assertAlmostEqual(0.0, agent.steps[0][0][0][0])
        self.assertAlmostEqual(0.0, agent.steps[0][0][0][1])
        self.assertAlmostEqual(0.0, agent.steps[0][0][0][2])
        self.assertAlmostEqual(5.0, agent.steps[0][0][0][3])

        self.assertAlmostEqual(0.0, agent.steps[0][1][0][0])
        self.assertAlmostEqual(0.5, agent.steps[0][1][0][1])

        self.assertAlmostEqual(-1.0, agent.steps[0][2].rewards[0])
        self.assertAlmostEqual(1.0, agent.steps[-1][2].rewards[0])


    def __on_reset_test(self, env, state):
        """Event handler of EnvPlane._reset() calls to generate diverse states
        for different environments and episodes in test() call

        :param env: EnvPlane instance

        :param state: states to adjust before returning from EnvPlane:
        states[agent_idx][0] - x position of an agent
        states[agent_idx][1] - y position of an agent
        states[agent_idx][2] - x position of a goal
        states[agent_idx][3] - y position of a goal"""

        for i in range(env.num_agents):
            state[i, 0] = self.__test_episode + i
            state[i, 1] = 0.0

        self.__test_episode += 1


    def __on_reset_train(self, *_):
        """Event handler of EnvPlane._reset() calls to generate diverse states
        for different environments and episodes in train() call"""

        self.__train_episode += 1


if __name__ == '__main__':
    unittest.main()
