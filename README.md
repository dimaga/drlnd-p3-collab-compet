# Project 3. Collaboration and Competition 
### Project Details

This is my solution to Collaboration and Competitionl Project of Udacity Deep Reinforcement
Learning course. Original project template is available at
https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet

TODO: The README describes the the project environment details (i.e., the state
and action spaces, and when the environment is considered solved).

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

Follow the steps, described in https://github.com/udacity/deep-reinforcement-learning/tree/dc65050c8f47b365560a30a112fb84f762005c6b README.md,
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
of PyTorch 0.4.0, and in Udacity Workspace with a CUDA version of PyTorch.

### Instructions

1. Download the project to your PC
1. Open __environment.py__ in your text editor and set a correct path to Reacher
simulator with 20 agents in ```ENV_PATH``` variable
1. Open your terminal, cd to the project folder
1. Run __test.py__ to test previously trained agent over 100 episodes
1. Run __train.py__ to retrain the agent 
1. Look through __Report.md__ of this repository to learn further details about
my solution