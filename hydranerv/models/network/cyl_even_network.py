import numpy as np
import matplotlib.pyplot as plt
from hydranerv.models.network.neuron import Neuron
from hydranerv.models.network.network import Network
from hydranerv.utils import utils

class CylEvenNetwork(Network):
    """a model of cylindrical neuronal networks where neurons are evenly distributed"""
    def __init__(self, num_lon=15, num_cir=15, gc=1, dt=.01, tmax=1000, pacemakers=[],
                t_ref=.1, conn_type='gap_junction', t_syn=.01, wnoise=0, is_semi_pm=False,
                prob_lon=1, prob_cir=1, rho=.5, seed=0):
        """constructor"""
        self.locations = [] # 2d matrix (phi, z)
        self.num_lon = num_lon
        self.num_cir = num_cir
        self.num = self.num_cir * self.num_lon
        self.prob_lon = prob_lon
        self.prob_cir = prob_cir
        self.rho = rho
        super().__init__(num_lon * num_cir, [], gc, dt, tmax, pacemakers, t_ref, conn_type,
                        t_syn, wnoise, is_semi_pm, seed)
        self.regions_cut = set()

    def set_locations(self):
        """set locations for neurons"""
        dphi = 2 * np.pi / self.num_cir
        dz = 1 / (self.num_lon - 1)
        for i, z in enumerate(np.arange(1, -dz/2, -dz)):
            for j, phi in enumerate(np.arange(0, 2 * np.pi, dphi)):
                self.locations.append((phi, z))

    def set_connections(self):
        """set connections for neurons"""
        self.set_locations()
        self.neighbors = [[] for _ in range(self.num)]
        for i in range(self.num_lon):
            for j in range(self.num_cir):
                curr = i * self.num_cir + j

                # set longitudinal edge
                if i < self.num_lon - 1:
                    below = (i + 1) * self.num_cir + j
                    if np.random.rand() < self.prob_lon:
                        self.add_edge((curr, below))

                # set circular edge
                nex = i * self.num_cir + j + 1
                if j == self.num_cir - 1:
                    nex = i * self.num_cir
                if np.random.rand() < self.prob_cir or i <= 0 or i >= self.num_lon - 1:
                    self.add_edge((curr, nex))

    def cut(self, phi_start, phi_end, z_start, z_end):
        """cut a specified region from the shell"""

        # remove neurons
        for i, nrn in enumerate(self.neurons):
            if nrn is not None:
                phi, z = self.locations[i]
                if phi_start <= np.round(phi, 3) <= phi_end and z_start <= np.round(z, 3) <= z_end:
                    self.neurons[i] = None
                    self.locations[i] = None

        # remove edges
        for k, edge in enumerate(self.edges):
            if edge is not None:
                i, j = edge
                if self.neurons[i] is None or self.neurons[j] is None:
                    self.edges[k] = None

        # remove neighbors
        for i in range(self.num):
            if self.neurons[i] is None:
                self.neighbors[i] = []
            else:
                li = self.neighbors[i][:]
                for j in self.neighbors[i]:
                    if self.neurons[j] is None:
                        li.remove(j)
                self.neighbors[i] = li

        self.regions_cut.add(((phi_start, phi_end), (z_start, z_end)))


    def disp_network(self, figsize=(8,8), show_pm=True):
        """display the network connectivity"""
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect((2*self.rho, 2*self.rho, 1))

        # Plot the surface
        u = np.linspace(0, 2 * np.pi, 50)
        h = np.linspace(0, 1, 20)
        x = np.outer(self.rho * np.cos(u), np.ones(len(h)))
        y = np.outer(self.rho * np.sin(u), np.ones(len(h)))
        z = np.outer(np.ones(len(u)), h)
        ax.plot_surface(x, y, z, color='lightgreen', alpha=.2)

        # Plot cut regions
        for region in self.regions_cut:
            u = np.linspace(region[0][0], region[0][1], 50)
            h = np.linspace(region[1][0], region[1][1], 20)
            x = np.outer(self.rho * np.cos(u), np.ones(len(h)))
            y = np.outer(self.rho * np.sin(u), np.ones(len(h)))
            z = np.outer(np.ones(len(u)), h)
            ax.plot_surface(x, y, z, color='k', alpha=.2)

        # Plot neurons
        for i, loc in enumerate(self.locations):
            if loc is not None:
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
            if edge is not None:
                phi1 = self.locations[edge[0]][0]
                z1 = self.locations[edge[0]][1]
                phi2 = self.locations[edge[1]][0]
                z2 = self.locations[edge[1]][1]
                ax.plot([self.rho * np.cos(phi1), self.rho * np.cos(phi2)],
                        [self.rho * np.sin(phi1), self.rho * np.sin(phi2)],
                        [z1, z2],
                        color='k', lw=.5)

        # make the panes transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # make the grid lines transparent
        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)


        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        plt.show()
