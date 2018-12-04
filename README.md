# Resnet-type Architecture for Actor-Critic Reinforcement learning
### Objective 
To use a deeper neural network for feature extraction in an actor-critic algorithm for reinforcement learning.

### Background
Actor-Critic models are a popular form of Policy Gradient model. Customized actor-critic methods use a smaller version of the convolution neural network. Actor-critic uses two neural network- (a) Critic- that measures how good the action taken is (value-based) and (b) an Actor that controls how our agent behaves (policy-based).

<img src="/imgs/figure1 alt="(a)">

Customized A3C architecture passes the input state st through a sequence of four convolution layers with 32 filters each, a kernel size of 3x3, a stride of 2 and padding of 1. An exponential
linear unit (ELU) is used after each convolution layer. The output of the last convolution layer is fed into an LSTM with 256 units. Two separate fully connected layers are used to predict the value function and the action from the LSTM feature representation.

In this study, we hypothesize that using deeper convolution networks would extract higher-level features resulting in better performance and an increase in the average reward per episode.

