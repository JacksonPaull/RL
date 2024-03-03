import numpy as np
import torch
from algo import ValueFunctionWithApproximation

class NN(torch.nn.Module):
    def __init__(self, 
                 in_feats = 2,
                 num_hidden_neurons = 32,
                 activation_fn = torch.nn.functional.relu):
        super().__init__()
        self.input = torch.nn.Linear(in_feats, num_hidden_neurons)
        self.hidden_1 = torch.nn.Linear(num_hidden_neurons, num_hidden_neurons)
        self.hidden_2 = torch.nn.Linear(num_hidden_neurons, 1)
        self.activation_fn = activation_fn

    def forward(self, x):
        x = torch.from_numpy(x.astype('float32'))
        x = self.input(x)
        x = self.activation_fn(x)
        x = self.hidden_1(x)
        x = self.activation_fn(x)
        x = self.hidden_2(x)
        return x

class ValueFunctionWithNN(ValueFunctionWithApproximation):
    def __init__(self, state_dims):
        """
        state_dims: the number of dimensions of state space
        """
        self.nn = NN(in_feats = state_dims)
        self.optimizer = torch.optim.Adam(self.nn.parameters(), betas=[0.9, 0.999])
        
    def __call__(self,s):
        self.nn.eval()
        return self.nn.forward(s).detach().numpy()[0]

    def update(self, alpha, G, s_tau):
        v = self.nn.forward(s_tau)
        self.nn.train()
        loss = 1/2 * (torch.tensor([G]) - v) ** 2
        self.nn.zero_grad()
        loss.backward()
        self.optimizer.step()

        return None

