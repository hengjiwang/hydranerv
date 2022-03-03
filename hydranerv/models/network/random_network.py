import numpy as np
from hydranerv.models.bursting.pressure.network_base import NetworkBase

class RandomNetwork(NetworkBase):

    def __init__(self, num=10, gc=100, dt=0.01):
        """constructor"""
        super().__init__(num, gc, dt)

    def make_connections(self, density=.2):
        """make connections between neurons"""
        for i in range(self.num):
            for j in range(i + 1, self.num):
                if np.random.rand() < density:
                    self.neighbors[i].append(j)
                    self.neighbors[j].append(i)

if __name__ == '__main__':
    randnet = RandomNetwork(gc=.075)
    randnet.run(100)
    randnet.disp(spike=False)