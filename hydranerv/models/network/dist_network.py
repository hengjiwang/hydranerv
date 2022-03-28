import numpy as np
import matplotlib.pyplot as plt
from hydranerv.models.network.neuron import Neuron
from hydranerv.models.network.network import Network
from hydranerv.utils import utils

class DistNetwork(Network):
    """a model for neuronal networks where the connectivity is exponentially decay with distance"""
    def __init__(self, num=10, gc=1, dt=.01, tmax=1000, pacemakers=[0], t_ref=.1,
                 conn_type='gap_junction', t_syn=.01, wnoise=0, is_semi_pm=False, lambda_d=.1, seed=0):
        """constructor"""
        self.locations = []
        self.lambda_d = lambda_d
        super().__init__(num, [], gc, dt, tmax, pacemakers, t_ref, conn_type, t_syn, wnoise, is_semi_pm, seed)

    def set_locations(self):
        """set locations for neurons"""
        for _ in self.neurons:
            x = np.random.rand()
            y = np.random.rand()
            self.locations.append((x, y))

    def set_connections(self):
        """set connections for neurons"""
        self.set_locations()
        self.neighbors = [[] for _ in range(self.num)]
        for i in range(self.num):
            for j in range(i+1, self.num):
                dist = utils.euclid_dist(self.locations[i], self.locations[j])
                if np.random.rand() < self.conn_prob(dist):
                    self.add_edge((i, j))

    def conn_prob(self, d):
        """connectivity probability"""
        return np.exp(- d ** 2 / 2 / self.lambda_d ** 2)

    def disp_network(self, show_pm=True):
        """display the network connectivity"""
        plt.figure(figsize=(8, 8))
        for loc in self.locations:
            plt.scatter(loc[0], loc[1], color='lightskyblue', s=100)
        if show_pm:
            for pacemaker in self.pacemakers:
                x = self.locations[pacemaker][0]
                y = self.locations[pacemaker][1]
                plt.scatter(x, y, color='lightskyblue', s=50, edgecolors=['r'], )
        for edge in self.edges:
            x1 = self.locations[edge[0]][0]
            y1 = self.locations[edge[0]][1]
            x2 = self.locations[edge[1]][0]
            y2 = self.locations[edge[1]][1]
            plt.plot([x1, x2], [y1, y2], 'gray', lw=.5)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    def find_nearst_neighbors(self, i, k):
        """find k nearst neighbors of neuron i (inclusive)"""
        distances = [utils.euclid_dist(self.locations[x], self.locations[i]) for x in range(self.num)]
        indices = np.argsort(distances)
        return indices[:k]




