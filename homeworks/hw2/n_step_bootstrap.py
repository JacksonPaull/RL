from typing import Iterable, Tuple

import numpy as np
from env import EnvSpec
from policy import Policy

# This is the same as a greedy policy normally?
# Original note/hint:
# "QPolicy" here refers to a policy that takes 
#    greedy actions w.r.t. Q values
class QPolicy(Policy):
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
        return np.argmax(self.Q[state]) == action

    def action(self,state:int) -> int:
        """
        input:
            state
        return:
            action
        """
        return np.argmax(self.Q[state])

def on_policy_n_step_td(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    n:int,
    alpha:float,
    initV:np.array
) -> Tuple[np.array]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        n: how many steps?
        alpha: learning rate
        initV: initial V values; np array shape of [nS]
    ret:
        V: $v_pi$ function; numpy array shape of [nS]
    """

    V = np.copy(initV)

    for traj in trajs:
        T = len(traj)

        for t in range(n-1, T+n-1):
            tao = t-n+1

            G = 0
            for i in range(tao, min(tao+n, T)):
                G += env_spec.gamma ** (i-tao-1) * traj[i][2]
            
            if tao + n < T:
                G = G + env_spec.gamma ** n * V[traj[tao+n][0]]
            V[traj[tao][0]] = V[traj[tao][0]] + alpha * (G - V[traj[tao][0]])

    return V

def off_policy_n_step_sarsa(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    n:int,
    alpha:float,
    initQ:np.array
) -> Tuple[np.array,Policy]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        n: how many steps?
        alpha: learning rate
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_star$ function; numpy array shape of [nS,nA]
        policy: $pi_star$; instance of policy class
    """

    pi = QPolicy(Q=initQ)
    Q = pi.Q

    for traj in trajs:
        T = len(traj)
        for t in np.arange(n-1, T+n-1):
            tao = t - n + 1

            # Calculate rho
            rho = 1
            for i in range(tao+1, min(tao+n+1, T)):
                s, a, _, _ = traj[i]
                rho *= pi.action_prob(s, a) / bpi.action_prob(s, a)

            G = 0
            for i in range(tao+1, min(tao + n + 1, T)):
                G += env_spec.gamma ** (i-tao-1) * traj[i-1][2]
            
            if tao + n < T:
                s_t1, a_t1, _, _ = traj[tao + n]
                G += env_spec.gamma ** n * Q[s_t1, a_t1]

            s_t, a_t, _, _ = traj[tao]
            Q[s_t, a_t] = Q[s_t, a_t] + alpha * rho * (G - Q[s_t, a_t])
            pi.Q = Q

    return Q, pi
