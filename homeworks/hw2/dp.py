from typing import Tuple

import numpy as np
from env import EnvWithModel
from policy import Policy

import logging

logger = logging.getLogger('dp.py')

class GreedyPolicy(Policy):    
    def __init__(self, Q=None, V=None, nA=None):
        if Q is not None:
            self._Q = Q
            self._V = np.max(Q, axis=-1)
        elif V is not None and nA is not None:
            self._V = V
            self._Q = np.column_stack([self._V] * nA)
        else:
            raise ValueError('Incorrect init for Greedy Policy')

    @property
    def Q(self):
        return self._Q
    
    # Setting the action values inherently sets the Q value as well
    @Q.setter
    def Q(self, Q):
        self._Q = Q
        self._V = np.max(Q, axis=-1)

    @property
    def V(self):
        return self._V

    
    def action_prob(self,state:int,action:int) -> float:
        """
        input:
            state, action
        return:
            \pi(a|s)
        """
        # pi(a|s) is 1 for the greedy action and 0 for the non-greedy actions
        return np.argmax(self.V[state]) == action

    def action(self,state:int) -> int:
        """
        input:
            state
        return:
            action
        """
        return np.argmax(self.V[state])
    
def value_prediction(env:EnvWithModel, pi:Policy, initV:np.array, theta:float) -> Tuple[np.array,np.array]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        pi: policy
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        V: $v_\pi$ function; numpy array shape of [nS]
        Q: $q_\pi$ function; numpy array shape of [nS,nA]
    """


    V = initV
    Q = np.column_stack([initV] * env.spec.nA)

    while True:
        v = np.copy(V)
        for s in range(env.spec.nS):
            s_prime_probs = env.TD[s]
            r = env.R[s]
            action_probs = [pi.action_prob(s, a) for a in range(env.spec.nA)]

            Q[s] = np.sum(s_prime_probs * (r + env.spec.gamma * V), axis=-1)
            V[s] = action_probs @ Q[s]


        delta = np.max(np.abs(v - V))
        if delta < theta:
            break

    return V, Q

def value_iteration(env:EnvWithModel, initV:np.array, theta:float) -> Tuple[np.array,Policy]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        value: optimal value function; numpy array shape of [nS]
        policy: optimal deterministic policy; instance of Policy class
    """

    pi = GreedyPolicy(V=initV, nA=env.spec.nA)
    V = pi.V

    while True:
        v = np.copy(V)
        for s in range(env.spec.nS):
            s_prime_probs = env.TD[s]
            r = env.R[s]

            V[s] = np.max(np.sum(s_prime_probs * (r + env.spec.gamma * V),axis=-1), axis=0)

        delta = np.max(np.abs(v - V))
        if delta < theta:
            break
    
    return V, pi
