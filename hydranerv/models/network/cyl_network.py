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
        self.cutlines = set()

    def set_locations(self):
        """set locations for neurons"""
        # hyperstomal neurons
        for _ in self.neurons[ : self.num_hyp]:
            phi = np.random.uniform(0, 2 * np.pi)
            z = np.random.uniform(.9, 1)
            self.locations.append((phi, z))
        # body column neurons
        for _ in self.neurons[self.num_hyp : self.num - self.num_ped]:
            phi = np.random.uniform(0, 2 * np.pi)
            z = np.random.uniform(.1, .9)
            self.locations.append((phi, z))
        # peduncle neurons
        for _ in self.neurons[self.num - self.num_ped : ]:
            phi = np.random.uniform(0, 2 * np.pi)
            z = np.random.uniform(0, .1)
            self.locations.append((phi, z))

    def set_connections(self):
        """set connections for neurons"""
        self.set_locations()
        self.neighbors = [[] for _ in range(self.num)]

        # generally
        for i in range(self.num):
            for j in range(i+1, self.num):
                dist = utils.cyl_dist((self.rho, self.locations[i][0], self.locations[i][1]),
                                      (self.rho, self.locations[j][0], self.locations[j][1]),
                                      z_scale=.3)
                if np.random.rand() < self.conn_prob(dist):
                    self.add_edge((i, j))

        # hypostomal ring
        for i in range(self.num_hyp):
            for j in range(i+1, self.num_hyp):
                dist = utils.cyl_dist((self.rho, self.locations[i][0], self.locations[i][1]),
                                      (self.rho, self.locations[j][0], self.locations[j][1]))
                if np.random.rand() < self.conn_prob(dist, lambda_d=.3):
                    self.add_edge((i, j))

        # hypostomal ring
        for i in range(self.num - self.num_ped, self.num):
            for j in range(i+1, self.num):
                dist = utils.cyl_dist((self.rho, self.locations[i][0], self.locations[i][1]),
                                      (self.rho, self.locations[j][0], self.locations[j][1]))
                if np.random.rand() < self.conn_prob(dist, lambda_d=.3):
                    self.add_edge((i, j))


    def conn_prob(self, d, lambda_d=None):
        """connectivity probability"""
        lambda_d = self.lambda_d if lambda_d is None else lambda_d
        return np.exp(- d ** 2 / 2 / lambda_d ** 2)

    def cut(self, direction, start, end, otherval):
        """cut a specified region from the shell"""

        # self.regions_cut.add(((phi_start, phi_end), (z_start, z_end)))
        if direction == 'phi':
            self.cutlines.add(((start, otherval), (end, otherval)))
        elif direction == 'z':
            self.cutlines.add(((otherval, start), (otherval, end)))

        for k, edge in enumerate(self.edges):

            # cut edge
            if edge is not None:
                i, j = edge
                phi1, z1 = self.locations[i]
                phi2, z2 = self.locations[j]

                # counterclockwise
                if phi1 > phi2:
                    phi1, phi2 = phi2, phi1
                    z1, z2 = z2, z1

                # choose shorter edge
                if phi2 - phi1 > np.pi:
                    phi2 -= 2 * np.pi

                if direction == 'phi':
                    if z1 <= otherval <= z2 or z2 <= otherval <= z1:
                        phi_intp = (phi1 + (phi2 - phi1) / (z2 - z1) * (otherval - z1)) % (2 * np.pi)
                        phi_intp = round(phi_intp, 3)
                        # if i == 17 and j == 71:
                        #     print(phi1, phi2, phi_intp, z1, z2)
                        #     print(utils.angle_in_range(start, end, phi_intp))
                        if utils.angle_in_range(start, end, phi_intp):
                            self.edges[k] = None
                            if j in self.neighbors[i]:
                                self.neighbors[i].remove(j)
                                self.neighbors[j].remove(i)

                elif direction == 'z':
                    lo, hi = (phi1, phi2) if phi2 > 0 else (phi2, phi1) # reorder
                    if utils.angle_in_range(lo, hi, otherval):
                        z_intp = (z2 - z1) / (phi2 - phi1) * (otherval - phi1) + z1
                        z_intp = round(z_intp, 3)
                        # if i == 9 and j == 93:
                        #     print(z1, z2, phi1, phi2, z_intp)
                        if start <= z_intp <= end or start <= z_intp <= end:
                            self.edges[k] = None
                            if j in self.neighbors[i]:
                                self.neighbors[i].remove(j)
                                self.neighbors[j].remove(i)

    def disp_network(self, figsize=(8,8), show_pm=True, plot_nid=False, savepath=None):
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

        for line in self.cutlines:
            u = np.linspace(line[0][0], line[1][0] + np.pi/50, 50)
            h = np.linspace(line[0][1], line[1][1] + .02, 20)
            x = np.outer(self.rho * np.cos(u), np.ones(len(h)))
            y = np.outer(self.rho * np.sin(u), np.ones(len(h)))
            z = np.outer(np.ones(len(u)), h)
            ax.plot_surface(x, y, z, color='k', alpha=1)

        # Plot neurons
        for i, loc in enumerate(self.locations):
            if loc is not None:
                phi = loc[0]
                z = loc[1]
                if show_pm and i in self.pacemakers:
                    ax.scatter(self.rho * np.cos(phi),
                            self.rho * np.sin(phi),
                            z,
                            color='C0', alpha=1, edgecolors=['r'])
                else:
                    ax.scatter(self.rho * np.cos(phi),
                            self.rho * np.sin(phi),
                            z,
                            color='C0', alpha=1)
                if plot_nid:
                    ax.text(self.rho * np.cos(phi),
                            self.rho * np.sin(phi),
                            z,
                            str(i))

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

        if savepath:
            plt.savefig(savepath, dpi=300, bbox_inches='tight')

        plt.show()



if __name__ == '__main__':
    ntwk = CylNetwork()
    ntwk.disp_network()
