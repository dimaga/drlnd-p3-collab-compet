#!python
"""Main module to test the agent training results"""

import torch
from env_agent_factory import create_env_agent

def main():
    """Shows training agent results"""

    env, agent = create_env_agent()

    agent.actor_local.load_state_dict(torch.load("actor.pth"))
    agent.critic_local.load_state_dict(torch.load("critic.pth"))

    env.test(agent, 100)

    print("Average Scores:", env.avg_score)
    print("Last Scores:", env.last_score)


if __name__ == "__main__":
    main()
