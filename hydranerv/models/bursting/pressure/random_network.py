import numpy as np
from hydranerv.models.bursting.pressure.network_base import NetworkBase

class Network(NetworkBase):

    def __init__(self, num=10, gc=100, dt=0.01):
        """constructor"""
        super().__init__(num, gc, dt)

    def make_connections(self, density=.1):
        """make connections between neurons"""
        pass

if __name__ == '__main__':
    network = Network(gc=.2)
    network.run(100)
    network.disp(spike=False)