import numpy as np
from policy import Policy
from tqdm import tqdm #TODO Remove

class ValueFunctionWithApproximation(object):
    def __call__(self,s) -> float:
        """
        return the value of given state; \hat{v}(s)

        input:
            state
        output:
            value of the given state
        """
        raise NotImplementedError()

    def update(self,alpha,G,s_tau):
        """
        Implement the update rule;
        w <- w + \alpha[G- \hat{v}(s_tau;w)] \nabla\hat{v}(s_tau;w)

        input:
            alpha: learning rate
            G: TD-target
            s_tau: target state for updating (yet, update will affect the other states)
        ouptut:
            None
        """
        raise NotImplementedError()

def semi_gradient_n_step_td(
    env, #open-ai environment
    gamma:float,
    pi:Policy,
    n:int,
    alpha:float,
    V:ValueFunctionWithApproximation,
    num_episode:int,
):
    """
    implement n-step semi gradient TD for estimating v

    input:
        env: target environment
        gamma: discounting factor
        pi: target evaluation policy
        n: n-step
        alpha: learning rate
        V: value function
        num_episode: #episodes to iterate
    output:
        None
    """
    
    for _ in tqdm(range(num_episode)):
        s = env.reset()
        t = 0
        T = np.inf
        S = np.array([s])
        R = np.array([0])
        while True:
            if t < T:
                # take action
                action = pi.action(S[t])
                s, reward, terminal, __ = env.step(action)
                R = np.append(R, reward)
                S = np.row_stack([S, s])
                if terminal:
                    T = t+1

            tau = t - n + 1
            if tau >= 0:
                G = 0
                for i in range(tau + 1, min(tau + n, T)+1):
                    G += R[i] * (gamma ** (i - tau - 1))
                if tau + n < T:
                    G += (gamma ** n) * V(S[tau + n])
                
                V.update(alpha, G, S[tau])

            if tau == T - 1:
                break   

            t += 1