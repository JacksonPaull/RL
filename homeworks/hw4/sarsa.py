import numpy as np
#from tqdm import tqdm

class StateActionFeatureVectorWithTile():
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_actions:int,
                 num_tilings:int,
                 tile_width:np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maimum value for each dimension in state
        num_actions: the number of possible actions
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        self.num_tilings = num_tilings
        self.num_actions = num_actions
        self.tile_width = tile_width

        self.tiling_dims = (np.ceil((state_high - state_low) / tile_width) + 1).astype('int')
        self.num_tiles = np.prod(self.tiling_dims)
        self.state_low = state_low
        self.state_high = state_high

    def feature_vector_len(self) -> int:
        """
        return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
        """
        return self.num_actions * self.num_tiles * self.num_tilings

    def __call__(self, s, done, a) -> np.array:
        """
        implement function x: S+ x A -> [0,1]^d
        if done is True, then return 0^d
        """
        X = np.zeros(np.prod(self.feature_vector_len()))

        # Early exit condition
        if done:
            return X
        
        for t in range(self.num_tilings):
            offset = self.tile_width * t/self.num_tilings
            activated_tiles = np.floor((s - self.state_low + offset ) / self.tile_width).astype('int')
            ids = np.ravel_multi_index(activated_tiles, self.tiling_dims) + \
                  t * self.num_tiles + a * self.num_tiles * self.num_tilings
            X[ids] = 1
        
        return X

def SarsaLambda(
    env, # openai gym environment
    gamma:float, # discount factor
    lam:float, # decay rate
    alpha:float, # step size
    X:StateActionFeatureVectorWithTile,
    num_episode:int,
) -> np.array:
    """
    Implement True online Sarsa(\lambda)
    """

    def epsilon_greedy_policy(s,done,w,epsilon=.0):
        nA = env.action_space.n
        Q = [np.dot(w, X(s,done,a)) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)

    w = np.zeros(X.feature_vector_len())
    for _ in range(num_episode):
        done = False
        s = env.reset()
        a = epsilon_greedy_policy(s, done, w, 0.1)
        x = X(s, done, a)
        z = np.zeros(x.shape)
        Q_old = 0

        while not done:
            s_prime, reward, done, __ = env.step(a)
            a_prime = epsilon_greedy_policy(s_prime, done, w)
            x_prime = X(s_prime, done, a_prime)

            Q = w @ x
            Q_prime = w @ x_prime

            delta = reward + gamma * Q_prime - Q
            z = gamma*lam*z + (1 - alpha * gamma * lam * z @ x) * x

            w = w + alpha * (delta + Q - Q_old) * z - alpha * (Q - Q_old) * x
            Q_old = Q_prime
            x = x_prime
            a = a_prime

    return w