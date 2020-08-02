# Project 3. Collaboration and Competition 
### Project Details

![Tennis](tennis.png)

This is my solution to Collaboration and Competitionl Project of Udacity Deep Reinforcement
Learning course. Original project template is available at
https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet

In this environment, two agents control rackets to bounce a ball over a net. If
an agent hits the ball over the net, it receives a reward of +0.1. If an agent
lets a ball hit the ground or hits the ball out of bounds, it receives a reward
of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and
velocity of the ball and racket. Each agent receives its own, local observation.
Two continuous actions are available, corresponding to movement toward (or away
from) the net, and jumping.

The task is episodic, and in order to solve the environment, the agents must
get an average score of +0.5 (over 100 consecutive episodes, after taking the
maximum over both agents). Specifically,

* After each episode, add up the rewards that each agent received (without
discounting), to get a score for each agent. This yields 2 (potentially
different) scores. Then take the maximum of these 2 scores.

* This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of
those scores is at least +0.5.

Besides README.md, this repository holds of the following files:

* __Report.md__ provides a description of the implementation
* __test.py__ is the main file for testing
* __train.py__ is the main file for training
* __actor.pth__ is the Actor neural network trained parameters
* __agent.py__ implements an agent for training and testing
* __env\_agent\_factory.py__ creates an environment and its agent
* __neural\_nets.py__ creates neural networks for an Actor and a Critic.
* __replay\_buffer.py__ implements a Replay Buffer 
* __*\_test.py__ unit tests of corresponding modules

All the Python code is pylint-compliant.

### Getting Started

Follow the steps, described in
https://github.com/udacity/deep-reinforcement-learning/tree/dc65050c8f47b365560a30a112fb84f762005c6b README.md,
Dependencies section, to deploy your development environment for this project.

Basically, you will need:

* Python 3.6
* PyTorch 0.4.0
* Numpy and Matplotlib, compatible with PyTorch
* Unity ML Agents. Udacity Navigation Project requires its own version of this
environment, available
https://github.com/udacity/deep-reinforcement-learning/tree/dc65050c8f47b365560a30a112fb84f762005c6b/python
with references to other libraries

The project has been developed and tested on Mac OS Catalina with a CPU version
of PyTorch 0.4.0.

### Instructions

1. Download the project to your PC
1. Open __environment.py__ in your text editor and set a correct path to Tennis
simulator in ```ENV_PATH``` variable
1. Open your terminal, cd to the project folder
1. Run __test.py__ to test previously trained agent over 100 episodes
1. Run __train.py__ to retrain the agent 
1. Look through __Report.md__ of this repository to learn further details about
my solution