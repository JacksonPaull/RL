import numpy as np

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

        self.num_tiles = (np.ceil((state_high - state_low) / tile_width) + 1).astytpe('int')

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
        X = np.zeros(self.feature_vector_len())

        # Early exit condition
        if done:
            return X
        
        for t in range(self.num_tilings):
            offset = self.tile_width * t/self.num_tilings
            activated_tiles = np.floor((s - self.state_low + offset ) / self.tile_width).astype('int')
            ids = np.ravel_multi_index(activated_tiles, self.num_tiles) + (a + 1) * t * np.prod(self.num_tiles)
            X[ids] = 1

        assert(X.sum() == self.num_tilings), 'Number of activated tiles is different than the number of tilings'
        
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

    w = np.zeros((X.feature_vector_len()))

    #TODO: implement this function
    raise NotImplementedError()