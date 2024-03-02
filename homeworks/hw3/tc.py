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
        self.tiling_dims = np.ceil((state_high - state_low) / tile_width) + 1
        total_feats = num_tilings * np.prod(self.tiling_dims)
        self.W = np.array([0.0 for _ in range(total_feats)])

    def construct_x(self, s):
        # TODO test this
        X = np.zeros(self.W.shape)
        for t in range(self.num_tilings):
            activated_tiles = np.floor((s - self.state_low + self.tile_width * t/self.num_tilings ) / self.tile_width)
            ids = np.ravel_multi_index(activated_tiles, self.tiling_dims) + t * np.prod(self.tiling_dims)
            X[ids] = 1
        
        return X

    def __call__(self,s):
        X = self.construct_x(s)
        return np.dot(self.W, X)

    def update(self,alpha,G,s_tau):
        grad = self.construct_x(s_tau)
        self.W = self.W + alpha * (G - self(s_tau)) * grad
        return None


"""
Notes:

Tile coding is a stacked one-hot vector
number of tiles could be defined as num_tilings * (state_high - state_low) / tile_width <-- should be length of dimension vector
gradient is trivial as simply equal to the weight vector since this is linear
"""