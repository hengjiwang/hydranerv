import numpy as np
import matplotlib.pyplot as plt
from hydranerv.models.network.rpnet.rp_neuron import RPNeuron
from hydranerv.models.network.rpnet.rp_network import RPNetwork
from hydranerv.utils import utils

class CylRPNetwork(RPNetwork):
    """a model of cylindrical network with rp neurons"""
    def __init__(self, num=100, gc=100, dt=.01, tmax=1000, t_ref=.1, conn_type='gap_junction',
                 t_syn=.01, k=3, rho=.5, lambda_d=.15, seed=0):
        """constructor"""
        self.locations = [] # (phi, z)
        self.k = k
        self.num = num
        self.rho = rho
        self.lambda_d = lambda_d
        super().__init__(num, [], gc, dt, tmax, t_ref, conn_type, t_syn, seed)

    def set_locations(self):
        """set locations for rp neurons"""
        for _ in self.neurons:
            phi = np.random.uniform(0, 2 * np.pi)
            z = np.random.uniform(0, 1)
            self.locations.append((phi, z))

    def set_connections(self):
        """set connections for neurons"""
        self.set_locations()
        self.neighbors = [[] for _ in range(self.num)]

        dists = np.zeros((self.num, self.num))

        # measure distances
        for i in range(self.num):
            for j in range(i+1, self.num):
                dist = utils.cyl_dist((self.rho, self.locations[i][0], self.locations[i][1]),
                                      (self.rho, self.locations[j][0], self.locations[j][1]))

                # if np.random.rand() < self.conn_prob(dist):
                #     self.add_edge((i, j))
                dists[i, j] = dist
                dists[j, i] = dist

        # # connect top k shortest
        # for i in range(self.num):
        #     for j in dists[i].argsort()[1:self.k+1]:
        #         self.add_edge((i, j))
        for i in range(self.num):
            for j in range(i+1, self.num):
                dist = utils.cyl_dist((self.rho, self.locations[i][0], self.locations[i][1]),
                                      (self.rho, self.locations[j][0], self.locations[j][1]))
                if np.random.rand() < self.conn_prob(dist):
                    self.add_edge((i, j))

    def conn_prob(self, d):
        """connectivity probability"""
        return np.exp(- d ** 2 / 2 / self.lambda_d ** 2)


    def disp_network(self, figsize=(8,8)):
        """display the network connectivity"""
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect((2*self.rho, 2*self.rho, 1))

        # Plot the surface
        u = np.linspace(0, 2 * np.pi, 50)
        h = np.linspace(0, 1, 20)
        x = np.outer(self.rho * np.sin(u), np.ones(len(h)))
        y = np.outer(self.rho * np.cos(u), np.ones(len(h)))
        z = np.outer(np.ones(len(u)), h)
        ax.plot_surface(x, y, z, color='lightgreen', alpha=.2)

        # Plot neurons
        for i, loc in enumerate(self.locations):
            phi = loc[0]
            z = loc[1]
            ax.scatter(self.rho * np.cos(phi),
                    self.rho * np.sin(phi),
                    z,
                    color='C1', alpha=1)

        # plot edges
        for edge in self.edges:
            phi1 = self.locations[edge[0]][0]
            z1 = self.locations[edge[0]][1]
            phi2 = self.locations[edge[1]][0]
            z2 = self.locations[edge[1]][1]
            ax.plot([self.rho * np.cos(phi1), self.rho * np.cos(phi2)],
                    [self.rho * np.sin(phi1), self.rho * np.sin(phi2)],
                    [z1, z2],
                    color='gray', lw=.5)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        # make the panes transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # make the grid lines transparent
        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

        plt.show()

if __name__ == '__main__':
    ntwk = CylRPNetwork()
    ntwk.disp_network()
