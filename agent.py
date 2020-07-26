#!python
"""Reinforcement Learning Agent that will train and act in the environment"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from replay_buffer import ReplayBuffer, DEVICE
from neural_nets import Actor, Critic


BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
EPS_START = 1.0         # Initial noise scalar multiplier
EPS_END = 0.01          # Final noise scalar multiplier
EPS_DECAY = 0.99995     # Noise exponential rate


class Agent:
    """Policy gradient agent to train and act in a distributed environment"""

    # pylint: disable=no-member, too-many-instance-attributes

    def __init__(self, state_size, action_size, num_agents):
        """Create an instance of Agent
        :param state_size: state vector dimension
        :param action_size: action vector dimension"""

        random_seed = 5

        self.__step_counter = 0
        self.__eps = EPS_START

        self.actor_local = Actor(
            state_size, action_size, random_seed).to(DEVICE)

        self.__actor_target = Actor(
            state_size, action_size, random_seed+1).to(DEVICE)

        self.__actor_optimizer = optim.Adam(
            self.actor_local.parameters())


        self.critic_local = Critic(
            state_size, action_size, random_seed+2).to(DEVICE)

        self.__critic_target = Critic(
            state_size, action_size, random_seed+3).to(DEVICE)

        self.__critic_optimizer = optim.Adam(
            self.critic_local.parameters())

        self.__memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed+4)

        # Noise process
        self.__noises = [
            OUNoise(action_size, random_seed+i) for i in range(num_agents)
        ]


    def reset(self):
        """The method is called in the beginning of each episode"""
        for noise in self.__noises:
            noise.reset()


    def step(self, states, actions, env_info):
        """Performs a training step
        :param states: current states of environments
        :param actions: actions which were taken by the agent upon states.
        :param env_info: Info of agent states after applying actions
        """

        # Save experiences / rewards
        self.__memory.add(states, actions, env_info)

        self.__step_counter += 1

        # Learn, if enough samples are available in memory
        if len(self.__memory) > BATCH_SIZE and 0 == self.__step_counter % 2:
            experiences = self.__memory.sample()
            self.__learn(experiences, GAMMA)


    def act(self, states, add_noise):
        """Calculates action vectors from state vectors for multiple
        environments
        :param states: state vectors from multiple environments
        :param add_noise: if True, adds noise vector
        :return: action vectors for multiple environments"""

        torch_states = torch.from_numpy(states).float().to(DEVICE)

        self.actor_local.eval()

        with torch.no_grad():
            actions = self.actor_local(torch_states).cpu().data.numpy()

        self.actor_local.train()

        if add_noise:
            for action, noise in zip(actions, self.__noises):
                action += self.__eps * noise.sample()

            self.__eps = max(EPS_END, EPS_DECAY * self.__eps)

        return np.clip(actions, -1.0, 1.0)


    def __learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience
        tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done)
            tuples gamma (float): discount factor"""

        states, actions, rewards, next_states, dones = experiences

        # Update critic
        # Get predicted next-state actions and Q values from target models
        actions_next = self.__actor_target(next_states)
        q_targets_next = self.__critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))
        # Compute critic loss
        q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(q_expected, q_targets)
        # Minimize the loss
        self.__critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.__critic_optimizer.step()

        # Update actor
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.__actor_optimizer.zero_grad()
        actor_loss.backward()
        self.__actor_optimizer.step()

        # Update target networks
        _soft_update(self.critic_local, self.__critic_target, TAU)
        _soft_update(self.actor_local, self.__actor_target, TAU)


def _soft_update(local_model, target_model, tau):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target

    Params
    ======
        local_model: PyTorch model (weights will be copied from)
        target_model: PyTorch model (weights will be copied to)
        tau (float): interpolation parameter
    """
    for target_param, local_param in zip(target_model.parameters(),
                                         local_model.parameters()):
        target_param.data.copy_(
            tau * local_param.data + (1.0 - tau) * target_param.data)



class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""

        self.state = np.zeros(size)
        self.theta = theta
        self.sigma = sigma

        np.random.seed(seed)


    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state.fill(0)


    def sample(self):
        """Update internal state and return it as a noise sample."""

        state = self.state

        dstate = self.theta * (-state) + self.sigma * np.random.randn(
            *state.shape)

        self.state = state + dstate

        return self.state
