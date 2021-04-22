# RL Lightning
A Research-Oriented RL Framework Using Pytorch-Lightning

#Basic idea
Using pytorch + lightning for an easy to use RL framework.

##Algorithm should include:
- DQN
- Multi-step DQN
- Prioritized DQN
- Double DQN
- Dueling DQN
- Rainbow?
- TD3
- SAC
- DDPG
- PPO
- A2C?
- EMDQN?
- Rainbow?



##Functionals should include:

- auto logging and plotting
  - yes！
- easy hyper-parameter saving and loading
- hyper parameter tuning
  - use Optuna
- expandability
  - Well inherited and disentangled code 
  - No boilerplate code 


## What should each part do?

###agent
- should define how to interact with env (given a state and exploration, return an action)
- should define the learning loss

[//]: <> (should know which state it is in?Yes.)

###learner
- should define the optimizer
- should define the training step, control how each part is used, e.g. exploration schedule, how long to interact with env and how long to train
- should couple with agent one-to-one?
  - No.

###buffer/data module
- should define how to restore(off policy)/generate(on policy) and sample data

### env
- should define the MDP agent is in
- should do all the wrapping and normalzation?
  - No.Normalization should be done in training for efficiency.

#Others

Implement NStep, Noisy, etc. from here:
https://github.com/djbyrne/core_rl
