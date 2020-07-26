#!python
"""Create a unity environment instance and an agent for it to avoid code
repetition"""

from environment import UnityEnv
from agent import Agent

def create_env_agent():
    """:return: unity_environment, agent"""

    env = UnityEnv()
    agent = Agent(env.state_size, env.action_size, env.num_agents)
    return env, agent
