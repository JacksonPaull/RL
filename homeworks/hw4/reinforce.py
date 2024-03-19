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
        x = self.layers[0](states)
        x = F.relu(x)
        x = self.layers[1](x)
        x = F.relu(x)
        x = self.layers[2](x)
        x = F.softmax(x, dim=-1)

        if return_prob:
            return x
        
        return torch.argmax(x, dim=-1)
    
    def __call__(self, state):
        assert(len(state.shape) == 1), 'Call only works for a single state'
        self.eval()
        return self.forward(torch.tensor(state, dtype=torch.float32)).item()

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

        # Binary Cross entropy scaled by delta * gamma_t
        loss =  ((delta * gamma_t).unsqueeze(dim=1) * torch.log(action_probs)).T @ actions_taken
        loss = torch.sum(loss)
        
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
        x = self.layers[0](states)
        x = F.relu(x)
        x = self.layers[1](x)
        x = F.relu(x)
        x = self.layers[2](x).flatten()
        return x

    def __call__(self, states):
        self.eval()
        return self.forward(torch.tensor(states, dtype=torch.float32))

    def update(self, states, G):
        self.train()
        v = self.forward(states)
        loss = F.mse_loss(v, G)
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
        A = [a]
        G_episode = [0]

        while True:
            s_prime, reward, done, _ = env.step(a)
            
            R.append(reward)
            if done:
                break
            
            # Don't append action and state' which take us to the terminal state
            S.append(s_prime)
            s = s_prime

            a = pi(s)
            A.append(a)

        T = len(A)
        for t in range(T-1): # A has length T - 1, all others have length T
            G = 0
            for k in range(t+1, T):
                G += gamma ** (k - t - 1) * R[k]
            delta = G - V(S[t])
            
            if t == 0:
                Gs.append(G) # Save the first G from each iteration

            G_episode.append(G)

        A = np.array(A)


        actions_taken = np.zeros((len(A), env.action_space.n))
        actions_taken[np.arange(A.size), A] = 1.0
        actions_taken = torch.from_numpy(actions_taken.astype('float32'))

        G_episode = torch.tensor(G_episode, dtype=torch.float32)
        states = torch.tensor(np.array(S), dtype=torch.float32)
        gamma_t = torch.tensor([gamma ** t for t in range(T)])

        pi.update(states, actions_taken, gamma_t, delta)
        V.update(states, G_episode)

    return Gs
