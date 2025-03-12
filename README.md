# Reinforcement Learning for Blackjack
## Overview
This project explores Reinforcement Learning (RL) techniques for optimizing strategy in the game of Blackjack. We implement and compare multiple RL algorithms, including:

#### Monte Carlo methods (On-Policy & Off-Policy)
#### Q-Learning & Deep Q-Learning (DQN)
#### SARSA (State-Action-Reward-State-Action)
#### Advantage Actor-Critic (A2C)
#### Proximal Policy Optimization (PPO)

Our goal is to analyze these methods' learning efficiency and performance in decision-making under uncertainty, evaluating factors like convergence speed, stability, and final policy effectiveness.

## Motivation

Blackjack is a well-defined environment with stochastic rewards and hidden information, making it an excellent testbed for reinforcement learning algorithms. The game presents challenges such as:

#### Large and heterogeneous state space
#### Sparse rewards (only at the end of each round)
#### Partial observability (dealer's full hand is unknown)
#### Exploration vs. exploitation trade-offs

## Environment

We use the OpenAI Gymnasium Blackjack-v1 environment, which simulates a simplified version of Blackjack. The game state consists of:

#### The player's current sum (12–21)
#### The dealer's visible card (Ace–10)
#### Whether the player has a usable Ace (counts as 11 without busting)
#### Actions:
##### Hit: Take another card.
##### Stick: End the turn.
#### Rewards:
##### +1 for winning
##### 0 for drawing
##### -1 for losing

## Methods

### Baseline Strategy
A simple rule-based strategy where the agent:
Hits if the sum is below 16
Sticks otherwise
### Monte Carlo Methods
On-Policy Control: Uses ε-greedy strategy to update Q-values based on sampled returns.
Off-Policy Control: Uses importance sampling to separate exploration and exploitation.
### Q-Learning & Deep Q-Network (DQN)
Q-Learning: Off-policy temporal difference learning, updating Q-values using the Bellman equation.
DQN: Uses a neural network instead of a Q-table, with experience replay and target networks for stability.
### SARSA (On-Policy TD Learning)
Updates Q-values based on actual next action taken, making it more conservative than Q-Learning.
### Advantage Actor-Critic (A2C) & Proximal Policy Optimization (PPO)
A2C: Uses a policy gradient approach with an actor-critic framework.
PPO: An improved policy gradient method with a clipped objective function for stable updates.

## Insights
PPO achieved the highest win rate and most stable policy learning.
Monte Carlo methods performed well due to Blackjack’s episodic nature.
Q-Learning and SARSA showed higher variance and slower convergence.
DQN improved upon Q-Learning but required significant hyperparameter tuning.
Entropy loss analysis suggests PPO maintains better exploration balance.

## Installation & Setup

### Prerequisites
Ensure you have Python 3.8+ and install the necessary dependencies:
```
pip install numpy gymnasium torch stable-baselines3 matplotlib tensorboard
```
### Run the Training
You can train different RL agents using:
```
python train_agent.py --method ppo
```
Replace ppo with qlearning, sarsa, montecarlo, or other methods.

## Future Improvements

Longer training time to observe the full convergence behavior of SARSA and A2C.
Hyperparameter tuning to optimize learning rate and exploration strategies.
Card-counting integration for more advanced Blackjack strategies.
Comparison with human expert play for real-world benchmarking.
Authors & Contributions

## References

#### Sutton & Barto, Reinforcement Learning: An Introduction, MIT Press, 2020.
#### OpenAI Gym: https://gymnasium.farama.org
