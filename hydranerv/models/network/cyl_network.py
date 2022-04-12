import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from hydranerv.models.network.neuron import Neuron
from hydranerv.models.network.network import Network
from hydranerv.utils import utils

class CylNetwork(Network):
    """a model of cylindrical neuronal networks"""
    def __init__(self, num=150, num_hyp=40, num_ped=40, gc=1, dt=.01, tmax=1000, pacemakers=[],
                 t_ref=.1, conn_type='gap_junction', t_syn=.01, wnoise=0, is_semi_pm=False,
                 lambda_d=.15, rho=.5, seed=0):
        """constructor"""
        self.locations = [] # (phi, z)
        self.lambda_d = lambda_d
        self.num_hyp = num_hyp
        self.num_ped = num_ped
        self.num_body = num - num_hyp - num_ped
        self.rho = rho
        super().__init__(num, [], gc, dt, tmax, pacemakers, t_ref, conn_type,
                         t_syn, wnoise, is_semi_pm, seed)

    def set_locations(self):
        """set locations for neurons"""
        # hyperstomal neurons
        for _ in self.neurons[ : self.num_hyp]:
            phi = np.random.uniform(0, 2 * np.pi)
            z = np.random.uniform(.95, 1)
            self.locations.append((phi, z))
        # body column neurons
        for _ in self.neurons[self.num_hyp : self.num - self.num_ped]:
            phi = np.random.uniform(0, 2 * np.pi)
            z = np.random.uniform(.05, .95)
            self.locations.append((phi, z))
        # peduncle neurons
        for _ in self.neurons[self.num - self.num_ped : ]:
            phi = np.random.uniform(0, 2 * np.pi)
            z = np.random.uniform(0, .05)
            self.locations.append((phi, z))

    def set_connections(self):
        """set connections for neurons"""
        self.set_locations()
        self.neighbors = [[] for _ in range(self.num)]
        for i in range(self.num):
            for j in range(i+1, self.num):
                dist = utils.cyl_dist((self.rho, self.locations[i][0], self.locations[i][1]),
                                      (self.rho, self.locations[j][0], self.locations[j][1]))
                if np.random.rand() < self.conn_prob(dist):
                    self.add_edge((i, j))

    def conn_prob(self, d):
        """connectivity probability"""
        return np.exp(- d ** 2 / 2 / self.lambda_d ** 2)

    def disp_network(self, figsize=(8,8), show_pm=True):
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
            if show_pm and i in self.pacemakers:
                ax.scatter(self.rho * np.cos(phi),
                        self.rho * np.sin(phi),
                        z,
                        color='#1f77b4', alpha=1, edgecolors=['r'])
            else:
                ax.scatter(self.rho * np.cos(phi),
                        self.rho * np.sin(phi),
                        z,
                        color='#1f77b4', alpha=1)

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

        plt.show()



if __name__ == '__main__':
    ntwk = CylNetwork()
    ntwk.disp_network()
