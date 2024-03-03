import numpy as np
from algo import ValueFunctionWithApproximation

class ValueFunctionWithTile(ValueFunctionWithApproximation):
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_tilings:int,
                 tile_width:np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maximum value for each dimension in state
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        self.state_low = state_low
        self.state_high = state_high
        self.num_tilings = num_tilings
        self.tile_width = tile_width
        self.tiling_dims = (np.ceil((state_high - state_low) / tile_width) + 1).astype('int')
        self.W = np.zeros(num_tilings * np.prod(self.tiling_dims))

    def construct_x(self, s):
        X = np.zeros(self.W.shape)
        for t in range(self.num_tilings):
            offset = self.tile_width * t/self.num_tilings
            activated_tiles = np.floor((s - self.state_low + offset ) / self.tile_width).astype('int')
            ids = np.ravel_multi_index(activated_tiles, self.tiling_dims) + t * np.prod(self.tiling_dims)
            X[ids] = 1
        assert(X.sum() == self.num_tilings), 'Number of activated tiles is different than the number of tilings'
        return X

    def __call__(self,s):
        X = self.construct_x(s)
        return np.dot(self.W, X)

    def update(self,alpha,G,s_tau):
        grad = self.construct_x(s_tau)
        self.W = self.W + alpha * (G - self(s_tau)) * grad
        return None