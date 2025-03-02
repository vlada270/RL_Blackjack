import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, action_dim)

    def forward(self, state):
        n = torch.tanh(self.l1(state))
        n = torch.tanh(self.l2(n))
        return n

    def pi(self, state, softmax_dim = 0):
        n = self.forward(state)
        prob = F.softmax(self.l3(n), dim=softmax_dim)
        return prob

class Critic(nn.Module):
    def __init__(self, state_dim,net_width):
        super(Critic, self).__init__()

        self.C1 = nn.Linear(state_dim, net_width)
        self.C2 = nn.Linear(net_width, net_width)
        self.C3 = nn.Linear(net_width, 1)

    def forward(self, state):
        v = torch.relu(self.C1(state))
        v = torch.relu(self.C2(v))
        v = self.C3(v)
        return v

def evaluate_policy(env, agent, turns = 3):
    total_scores = 0
    wins = 0
    ties = 0
    for j in range(turns):
        s, info = env.reset()
        s = tuple_to_one_hot(s, env.observation_space) 
        done = False
        while not done:
            # Take deterministic actions at test time
            a, logprob_a = agent.select_action(s, deterministic=True)
            s_next, r, dw, tr, info = env.step(a)
            s_next = tuple_to_one_hot(s_next, env.observation_space) 
            done = (dw or tr)

            total_scores += r
            s = s_next

        if r == 1:
            wins += 1
        elif r == 0:
            ties += 1

    win_rate = float(wins) / float(turns)

    return total_scores / turns, win_rate


def tuple_to_one_hot(state, space):
    """
    transfer Tuple(Discrete) to one-hot
    """
    one_hot_vectors = []
    for value, discrete_space in zip(state, space):
        one_hot = np.zeros(discrete_space.n)  
        one_hot[value] = 1  
        one_hot_vectors.append(one_hot)
    return np.concatenate(one_hot_vectors)  
