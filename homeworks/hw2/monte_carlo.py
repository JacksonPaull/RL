from typing import Iterable, Tuple

import numpy as np
from env import EnvSpec
from policy import Policy

def off_policy_mc_prediction_ordinary_importance_sampling(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    pi:Policy,
    initQ:np.array
) -> np.array:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        pi: evaluation target policy
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_pi$ function; numpy array shape of [nS,nA]
    """

    Q = np.copy(initQ)
    tao = np.zeros(Q.shape)

    for traj in trajs:
        G = 0
        W = 1

        # Loop from t = T -> 0
        for t in np.arange(len(traj))[::-1]:
            s_t, a_t, r_t1, s_t1 = traj[t]
            tao[s_t, a_t] += 1
            G = env_spec.gamma * G + r_t1
            Q[s_t, a_t] = Q[s_t, a_t] + W / tao[s_t, a_t] * (G - Q[s_t, a_t]) 
            W = W * (pi.action_prob(s_t, a_t)) / (bpi.action_prob(s_t, a_t))

            if W == 0:
                break
    return Q

def off_policy_mc_prediction_weighted_importance_sampling(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    pi:Policy,
    initQ:np.array
) -> np.array:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using behavior policy bpi
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        pi: evaluation target policy
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_pi$ function; numpy array shape of [nS,nA]
    """

    Q = np.copy(initQ)
    C = np.zeros(Q.shape)

    for traj in trajs:
        G = 0
        W = 1

        # Loop from t = T -> 0
        for t in np.arange(len(traj))[::-1]:
            s_t, a_t, r_t1, s_t1 = traj[t]
            G = env_spec.gamma * G + r_t1
            C[s_t, a_t] = C[s_t, a_t] + W
            Q[s_t, a_t] = Q[s_t, a_t] + W / C[s_t, a_t] * (G - Q[s_t, a_t])
            W = W * (pi.action_prob(s_t, a_t)) / (bpi.action_prob(s_t, a_t))

            if W == 0:
                break
    return Q
