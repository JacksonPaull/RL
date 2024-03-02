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
        # TODO: implement this method

    def __call__(self,s):
        # TODO: implement this method
        return 0.

    def update(self,alpha,G,s_tau):
        # TODO: implement this method
        return None


"""
Notes:

Tile coding is a stacked one-hot vector
number of tiles could be defined as num_tilings * (state_high - state_low) / tile_width <-- should be length of dimension vector
gradient is trivial as simply equal to the weight vector since this is linear
"""