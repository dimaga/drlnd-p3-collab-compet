# Project 3. Collaboration and Competition
### Learning Algorithm

![Tennis](tennis.png)

The algorithm is based on a paper _Multi-Agent Actor-Critic for Mixed
Cooperative-Competitive Environments by OpenAI_ available at
<https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf>.
It generalizes Actor-Critic DDPG algorithm to multiple competing or cooperating
agents. The main idea of the approach is to extend a Critic so that it uses
observations of all agents. This restores stationarity condition of the
environment. The Critic is used only at a training stage. The Actor takes only
local observations of its agent as the input. Only Actor is used on the testing
stage. Therefore, an agent does not need full access to the state environment.

#### Actor

An Actor takes a state vector of 8 elements, corresponding to the position and
velocity of the ball and racket, and produces 2 action values, corresponding to
movement toward (or away from) the net, and jumping.

The structure of the Actor network is shown in the picture below. It is borrowed
from my Project 2: Continuous Control solution.

![Actor](actor.png)

#### Critic

A critic accepts a 16-dimensional __full__ state vector, consisting of states of
an agent and its opponent. It passes the full state vector through a fully 
connected layer and a leaky relu unit. Then it combines results with 2-element
action vector, passes through another fully connected layer and through the
batch normalization.

The structure of the Actor network is shown in the picture below. It is also
borrowed from my Project 2: Continuous Control solution.

![Critic](critic.png)

#### Training the Agent

__train.py__ activates the training of the neural network. It creates an
agent, defined in __agent.py__, in a Unity environment through
__environment.py__.

The environment consists of 2 similar agents, learning to play tennis. 

As the training progresses, the agents maximize their cumulative rewards and
improve their scores:

```shell
python train.py

...
Episode 3100	Average Score Over All Agents: 0.09 
Episode 3200	Average Score Over All Agents: 0.12 
Episode 3300	Average Score Over All Agents: 0.12 
Episode 3400	Average Score Over All Agents: 0.31 
Episode 3500	Average Score Over All Agents: 1.23 
Saving actor.pth with score 1.2316000183857978
```

In the end of training, agents almost never miss a ball.

Actor neural network parameters are saved in __actor.pth__. Critic neural
network parameters are not saved, because it is not used in testing. You can
replay the training results later with __test.py__ script:

```shell
python test.py
```

In the beginning of __agent.py__, the following hyper-parameters are defined for
the training process:

```python
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
EPS_START = 1.0         # Initial noise scalar multiplier
EPS_END = 0.01          # Final noise scalar multiplier
EPS_DECAY = 0.999995    # Noise exponential rate
```

The values and most of the code were borrowed from my Project 2: Continuous
Control solution. I only increased the value of ```EPS_DECAY``` to keep the
noise high during during an increased number of training period: the agent
achieves +1.23 score after 3500 episodes.

### Plot of Rewards

The plot of an average maximum score of 2 agents over 100-episode window is
shown in the picture below. The agent achieves a score above +0.5 in
~3440 episodes, which meets the passing criteria for this project.

![TrainingGraph](training_graph.png)

The agent was trained on a CPU device of PyTorch library. The CUDA device will
likely to show different results, as it applies random seeds differently.

### Ideas for Future Work

* Include an opponent action into a full state vector of a Critic to train
agents faster

* Train policy ensembles: multiple critics. At each iteration, randomly pick
a critic for stochastic gradient descent step

* Try a different metric to evaluate performance of competing agents. Instead
of score, use ELO (see <https://en.wikipedia.org/wiki/Elo_rating_system>) or
trueskill (see <https://en.wikipedia.org/wiki/TrueSkill>)

* Try to apply the multiple agent DDPG algorithm to different domains:
    * Extend the tennis environment to 3D table to include lateral motion of
      the racket   
    * Try to build a real robot, playing tennis    
    * Try to apply multi-agent reinforcement learning to cryptography or
hierarchical planning

*  Try hidden memory, LSTM, intrinsic motivation, attention networks,
self-supervised learning and other buzzwords to improve performance of the
tennis player

* Look through recent papers for further ideas:
    * <https://openai.com/blog/emergent-tool-use/>
    * <https://deepmind.com/research/publications/A-Generalized-Training-Approach-for-Multiagent-Learning>
    * <https://deepmind.com/research/publications/Learning-to-cooperate-Emergent-communication-in-multi-agent-navigation>
    * <https://deepmind.com/research/publications/learning-communicate-deep-multi-agent-reinforcement-learning>
    * <https://deepmind.com/research/publications/unified-game-theoretic-approach-multiagent-reinforcement-learning>
    * <https://deepmind.com/research/publications/relational-forward-models-multi-agent-learning>
