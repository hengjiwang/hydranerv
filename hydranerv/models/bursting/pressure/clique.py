import numpy as np
from hydranerv.models.bursting.pressure.network_base import NetworkBase

class Clique(NetworkBase):

    def __init__(self, num=10, gc=100, dt=0.01):
        """constructor"""
        super().__init__(num, gc, dt)

    def make_connections(self):
        """make connections between neurons"""
        for i in range(self.num):
            for j in range(i + 1, self.num):
                self.neighbors[i].append(j)
                self.neighbors[j].append(i)

if __name__ == '__main__':
    clique = Clique(gc=.075)
    clique.run(200)
    clique.disp(spike=False)