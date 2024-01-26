import copy
import numpy as np

def clamp(x, _min, _max):
    return max(min(x, _max), _min)

class GridWorld():

    def __init__(self):
        self.grid_values = [[0 for j in range(4)] for i in range(5)]
        self.grid_values = np.asarray(self.grid_values).astype(float)

    def update_values(self):
        g = np.copy(self.grid_values)
        for x in range(1, 15):
            i = x // 4
            j = x % 4

            v = 0

            if x == 13:
                # update with special case of transition to 15
                for di, dj in [(0, -1), (1, 0), (0, 1)]:
                    ni, nj = clamp(i + di, 0, 3), clamp(j + dj, 0, 3)
                    v += (-1 + self.grid_values[ni][nj]) / 4

                v += (-1 + self.grid_values[4][1]) / 4 # Add value for space 15

            else:
                # update like normal
                for di, dj in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                    ni, nj = clamp(i + di, 0, 3), clamp(j + dj, 0, 3)
                    v += (-1 + self.grid_values[ni][nj]) / 4

            g[i][j] = v

        #update space 15
        g[4][1] = (g[3][0] + g[3][1] + g[3][2] + g[4][1])/4 - 1

        self.grid_values = g

    def value_iteration(self):
        k = 0
        while True:
            g = copy.deepcopy(self.grid_values)
            print('Running update, k=', k)
            self.update_values()
            print('Updated Array:\n', self.grid_values)
            if (self.grid_values - g).sum() == 0:
                return g
            k += 1
            

if __name__ == '__main__':
    gw = GridWorld()

    g = gw.value_iteration()
    #print(g)


