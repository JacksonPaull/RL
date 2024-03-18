from typing import Iterable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from tqdm import tqdm

class PiApproximationWithNN(nn.Module):
    def __init__(self, state_dims, num_actions, alpha):
        """
        state_dims: the number of dimensions of state space
        action_dims: the number of possible actions
        alpha: learning rate
        """
        super(PiApproximationWithNN, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(state_dims, 32),
            nn.Linear(32, 32),
            nn.Linear(32, num_actions)
        ])

        self.optimizer = optim.Adam(self.parameters(), lr = alpha, betas=(0.9, 0.999))

    def forward(self, states, return_prob=False):

        # Note: You will want to return either probabilities or an action
        # Depending on the return_prob parameter
        # This is to make this function compatible with both the
        # update function below (which needs probabilities)
        # and because in test cases we will call pi(state) and 
        # expect an action as output.
        x = torch.from_numpy(states.astype('float32'))
        x = self.layers[0](x)
        x = F.relu(x)
        x = self.layers[1](x)
        x = F.relu(x)
        x = self.layers[2](x)
        x = F.softmax(x, dim=0)

        if return_prob:
            return x
        
        return np.random.choice(a = np.arange(len(x)), p=x.detach().numpy())
    
    def __call__(self, states):
        self.eval()
        return self.forward(states)

    def update(self, states, actions_taken, gamma_t, delta):
        """
        states: states
        actions_taken: actions_taken
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        # TODO: probably a bug here
        self.train()
        action_probs = self.forward(states, return_prob=True)
        a = np.zeros(action_probs.shape)
        a[actions_taken] = 1.0

        loss = gamma_t * delta * F.cross_entropy(action_probs, 
                                                 torch.from_numpy(a))
        F.binary_cross_entropy()
        self.zero_grad()
        loss.backward()
        self.optimizer.step()


class Baseline(object):
    """
    The dumbest baseline; a constant for every state
    There is no need to change this class.
    """
    def __init__(self,b):
        self.b = b
        
    def __call__(self, states):
        return self.forward(states)
        
    def forward(self, states) -> float:
        return self.b

    def update(self, states, G):
        pass

class VApproximationWithNN(nn.Module):
    def __init__(self, state_dims, alpha):
        """
        state_dims: the number of dimensions of state space
        alpha: learning rate
        """
        super(VApproximationWithNN, self).__init__()
        
        self.layers = nn.ModuleList([
            nn.Linear(state_dims, 32),
            nn.Linear(32, 32),
            nn.Linear(32, 1)
        ])
        self.optimizer = optim.Adam(self.parameters(), lr = alpha, betas=(0.9, 0.999))

    def forward(self, states) -> float:
        x = torch.from_numpy(states.astype('float32'))
        x = self.layers[0](x)
        x = F.relu(x)
        x = self.layers[1](x)
        x = F.relu(x)
        x = self.layers[2](x).flatten()
        return x

    def __call__(self, states):
        self.eval()
        return self.forward(states).detach().numpy()[0]

    def update(self, states, G):
        self.train()
        v = self.forward(states)
        loss = 1/2 * (torch.tensor([G]) - v) ** 2
        self.zero_grad()
        loss.backward()
        self.optimizer.step()


def REINFORCE(
    env, #open-ai environment
    gamma:float,
    num_episodes:int,
    pi:PiApproximationWithNN,
    V:VApproximationWithNN) -> Iterable[float]:
    """
    implement REINFORCE algorithm with and without baseline.

    input:
        env: target environment; openai gym
        gamma: discount factor
        num_episode: #episodes to iterate
        pi: policy
        V: baseline
    output:
        a list that includes the G_0 for every episodes.
    """
    Gs = []
    for episode in tqdm(range(num_episodes)):
        # Generate the entire episode
        s = env.reset()
        a = pi(s)
        S = [s]
        R = [0]
        A = []

        while True:
            s_prime, reward, done, _ = env.step(a)
            A.append(a)
            S.append(s_prime)
            R.append(reward)
            if done:
                break
            
            s = s_prime
            a = pi(s)

        T = len(R)
        for t in range(T-1): # A has length T - 1, all others have length T
            G = 0
            for k in range(t+1, T):
                G += gamma ** (k - t - 1) * R[k]
            delta = G - V(S[t])
            
            if t == 0:
                Gs.append(G) # Save the first G from each iteration

            pi.update(S[t], A[t], gamma ** t, delta)
            V.update(S[t], G)

    return Gs
