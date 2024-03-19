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
        
        hidden_size = 32
        self.layers = nn.ModuleList([
            nn.Linear(state_dims, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, num_actions)
        ])

        self.optimizer = optim.Adam(self.parameters(), lr = alpha, betas=(0.9, 0.999))

    def forward(self, states, return_prob=False):

        # Note: You will want to return either probabilities or an action
        # Depending on the return_prob parameter
        # This is to make this function compatible with both the
        # update function below (which needs probabilities)
        # and because in test cases we will call pi(state) and 
        # expect an action as output.
        x = states
        for i, l in enumerate(self.layers):
            x = l(x)
            if i == len(self.layers) - 1:
                x = F.softmax(x, dim=-1)
            else:
                x = F.relu(x)

        if return_prob:
            return x
        
        x = x.detach().numpy()
        return np.random.choice(np.arange(len(x)), p=x)
    
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
        loss =  (delta * gamma_t).unsqueeze(dim=1) * (torch.log(action_probs) @ actions_taken.T)
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
        
        hidden_size = 32
        self.layers = nn.ModuleList([
            nn.Linear(state_dims, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, 1)
        ])
        self.optimizer = optim.Adam(self.parameters(), lr = alpha, betas=(0.9, 0.999))

    def forward(self, states) -> float:
        x = states
        for i, l in enumerate(self.layers):
            x = l(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x.flatten()

    def __call__(self, states):
        self.eval()
        return self.forward(torch.tensor(states, dtype=torch.float32))

    def update(self, states, G):
        self.train()
        v = self.forward(states)
        loss = torch.sum((v - G) ** 2)
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
        deltas = [-V(S[0])]

        # Generate episode
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
        for t in range(T-1):
            G = np.array([gamma ** (k - t - 1) for k in range(t+1, T+1)]) @ np.array(R[t+1:])
            delta = G - V(S[t])
            deltas.append(delta)

            G_episode.append(G)

        # Cache G_0
        Gs.append(G_episode[1])
        A = np.array(A)

        actions_taken = np.zeros((len(A), env.action_space.n))
        actions_taken[np.arange(A.size), A] = 1.0
        actions_taken = torch.from_numpy(actions_taken.astype('float32'))

        G_episode = torch.tensor(G_episode, dtype=torch.float32)
        states = torch.tensor(np.array(S), dtype=torch.float32)
        gamma_t = torch.tensor([gamma ** t for t in range(T)])

        pi.update(states, actions_taken, gamma_t, torch.tensor(deltas, dtype=torch.float32))
        V.update(states, G_episode)

    return Gs
