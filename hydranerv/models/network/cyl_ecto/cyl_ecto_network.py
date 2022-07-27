import numpy as np
import matplotlib.pyplot as plt
from hydranerv.models.network.cyl_network import CylNetwork
from hydranerv.models.network.rpnet.cyl_rp_network import CylRPNetwork
from collections import defaultdict
from hydranerv.utils import utils
from tqdm import tqdm

class CylEctoNetwork:
    """a cylindrical network model including both cb and rp neurons"""
    def __init__(self,
                 dt=.01,
                 tmax=1000,
                 num_cb=100,
                 num_rp=100,
                 num_hyp=30,
                 num_ped=30,
                 gc_cb=500,
                 gc_rp=500,
                 pm_cb=None,
                 t_ref=.1,
                 edges=None,
                 tau_inh_cb=5,
                 tau_inh_rp=5,
                 a_inh_cb=50,
                 a_inh_rp=100,
                 lambda_d=(.1, .05),
                 wnoise=10000,
                 seed_cb=123,
                 seed_rp=5):
        """constructor"""
        self.dt = dt
        self.tmax = tmax
        self.cbnet = CylNetwork(num=num_cb,
                                num_hyp=num_hyp,
                                num_ped=num_ped,
                                gc=gc_cb,
                                dt=dt,
                                tmax=tmax,
                                pacemakers=pm_cb,
                                t_ref=t_ref,
                                conn_type="gap_junction",
                                t_syn=.01,
                                wnoise=wnoise,
                                is_semi_pm=False,
                                lambda_d=.15,
                                rho=.5,
                                seed=seed_cb)
        self.rpnet = CylRPNetwork(num=num_rp,
                                  num_hyp=num_hyp,
                                  num_ped=num_ped,
                                  gc=gc_rp,
                                  dt=dt,
                                  tmax=tmax,
                                  t_ref=t_ref,
                                  conn_type="gap_junction",
                                  t_syn=.01,
                                  rho=.5,
                                  lambda_d=.25,
                                  seed=seed_rp)
        self.tau_inh_cb = tau_inh_cb
        self.tau_inh_rp = tau_inh_rp
        self.a_inh_cb = a_inh_cb
        self.a_inh_rp = a_inh_rp
        self.edges = edges
        self.rho = .5
        self.lambda_d = lambda_d
        self.i_inh = defaultdict(defaultdict)
        self.setup()
        self.reset()
        self.cutlines = set()

    def define_edges(self, method='dist_decay', k=1):
        """define cross edges"""

        self.edges = defaultdict(list)
        if method == 'dist_decay':
            for i in range(self.cbnet.num):
                rho_cb = self.cbnet.rho
                phi_cb, z_cb = self.cbnet.locations[i]
                for j in range(self.rpnet.num):
                    rho_rp = self.rpnet.rho
                    phi_rp, z_rp = self.rpnet.locations[j]
                    dist = utils.cyl_dist((rho_cb, phi_cb, z_cb),
                                          (rho_rp, phi_rp, z_rp))

                    randnum = np.random.rand()
                    if randnum < np.exp(- dist ** 2 / 2 / (self.lambda_d[0]) ** 2): # * .4: # np.exp( - 2 * z_rp):
                        self.edges['rp_to_cb'].append((i, j))
                    randnum = np.random.rand()
                    if randnum < np.exp(- dist ** 2 / 2 / (self.lambda_d[1]) ** 2): # * .4: # * np.exp(2 * (z_cb - 1)):
                        self.edges['cb_to_rp'].append((j, i))

        elif method == 'nearest_neighbor':
            # calculate distances
            distances = defaultdict(list)
            distances['rp_to_cb'] = np.zeros(self.cbnet.num, self.rpnet.num)
            distances['cb_to_rp'] = np.zeros(self.rpnet.num, self.cbnet.num)

            for i in range(self.cbnet.num):
                rho_cb = self.cbnet.rho
                phi_cb, z_cb = self.cbnet.locations[i]
                for j in range(self.rpnet.num):
                    rho_rp = self.rpnet.rho
                    phi_rp, z_rp = self.rpnet.locations[j]
                    dist = utils.cyl_dist((rho_cb, phi_cb, z_cb),
                                          (rho_rp, phi_rp, z_rp))
                    distances['rp_to_cb'][i, j] = dist
                    distances['cb_to_rp'][j, i] = dist

            # define rp -> cb edges
            for i in range(self.cbnet.num):
                for j in distances['rp_to_cb'][i].argsort()[:k]:
                    self.edges['rp_to_cb'].append((i, j))

            # define cb -> rp edges
            for i in range(self.rpnet.num):
                for j in distances['cb_to_rp'][i].argsort()[:k]:
                    self.edges['cb_to_rp'].append((i, j))

        else:
            raise ValueError("method is not defined.")

    def setup(self):
        """set up the structure"""
        self.define_edges()

        self.prev = defaultdict(list)
        # set rp -> cb edges
        self.prev['cb'] = [[] for _ in range(self.cbnet.num)]
        for edge in self.edges['rp_to_cb']:
            self.prev['cb'][edge[0]].append(edge[1])

        # set cb -> rp edges
        self.prev['rp'] = [[] for _ in range(self.rpnet.num)]
        for edge in self.edges['cb_to_rp']:
            self.prev['rp'][edge[0]].append(edge[1])

    def reset(self):
        """reset the state"""
        self.t = 0
        self.cbnet.reset()
        self.rpnet.reset()
        for edge in self.edges['rp_to_cb']:
            self.i_inh['rp_to_cb'][edge] = [0]
        for edge in self.edges['cb_to_rp']:
            self.i_inh['cb_to_rp'][edge] = [0]

    def step(self):
        """step function"""
        voltages = defaultdict(list)
        voltages['cb'] = [x.v() for x in self.cbnet.neurons]
        voltages['rp'] = [x.v() for x in self.rpnet.neurons]

        # update cb neurons
        for i, cb in enumerate(self.cbnet.neurons):
            i_inh_total = 0
            for j in self.prev['cb'][i]:
                rp = self.rpnet.neurons[j]
                i_inh = self.i_inh['rp_to_cb'][(i, j)][-1]
                if np.abs(self.t - self.dt - rp.t_last) < self.dt / 2:
                    i_inh += self.a_inh_rp
                i_inh -= self.dt * i_inh / self.tau_inh_rp
                self.i_inh['rp_to_cb'][(i, j)][-1] = i_inh
                i_inh_total += i_inh
            cb.step(self.cbnet.i_c(i, voltages['cb']) - i_inh_total)


        # update rp neurons
        for i, rp in enumerate(self.rpnet.neurons):
            i_inh_total = 0
            for j in self.prev['rp'][i]:
                cb = self.cbnet.neurons[j]
                i_inh = self.i_inh['cb_to_rp'][(i, j)][-1]
                if np.abs(self.t - self.dt - cb.t_last) < self.dt / 2:
                    i_inh += self.a_inh_cb
                i_inh -= self.dt * i_inh / self.tau_inh_cb
                self.i_inh['cb_to_rp'][(i, j)][-1] = i_inh
                i_inh_total += i_inh
            sigma_m = cb.sigma_m() # TODO: replace this cb to the closest cb to rp
            rp.step(sigma_m, self.rpnet.i_c(i, voltages['rp']) - i_inh_total)

        self.t += self.dt

    def run(self):
        """run simulation"""
        self.reset()
        time_axis = np.arange(self.dt, self.tmax, self.dt)
        for t in tqdm(time_axis):
            self.step()

    def cut(self, direction, start, end, otherval):
        """cut a specified region from the shell"""

        if direction == 'phi':
            self.cutlines.add(((start, otherval), (end, otherval)))
        elif direction == 'z':
            self.cutlines.add(((otherval, start), (otherval, end)))

        # cut each net
        self.cbnet.cut(direction, start, end, otherval)
        self.rpnet.cut(direction, start, end, otherval)


        # cut edges and prevs
        for k, edge in enumerate(self.edges['rp_to_cb']):

            # cut edge
            if edge is not None:
                i, j = edge
                phi1, z1 = self.cbnet.locations[i]
                phi2, z2 = self.rpnet.locations[j]

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
                        if utils.angle_in_range(start, end, phi_intp):
                            self.edges['rp_to_cb'][k] = None
                            li = self.prev['cb'][i][:]
                            if j in self.prev['cb'][i]:
                                li.remove(j)
                            self.prev['cb'][i] = li

                elif direction == 'z':
                    lo, hi = (phi1, phi2) if phi2 > 0 else (phi2, phi1) # reorder
                    if utils.angle_in_range(lo, hi, otherval):
                        z_intp = (z2 - z1) / (phi2 - phi1) * (otherval - phi1) + z1
                        z_intp = round(z_intp, 3)
                        if start <= z_intp <= end or start <= z_intp <= end:
                            self.edges['rp_to_cb'][k] = None
                            li = self.prev['cb'][i][:]
                            if j in self.prev['cb'][i]:
                                li.remove(j)
                            self.prev['cb'][i] = li

        for k, edge in enumerate(self.edges['cb_to_rp']):

            # cut edge
            if edge is not None:
                i, j = edge
                phi1, z1 = self.rpnet.locations[i]
                phi2, z2 = self.cbnet.locations[j]

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
                        if utils.angle_in_range(start, end, phi_intp):
                            self.edges['cb_to_rp'][k] = None
                            li = self.prev['rp'][i][:]
                            if j in self.prev['rp'][i]:
                                li.remove(j)
                            self.prev['rp'][i] = li

                elif direction == 'z':
                    lo, hi = (phi1, phi2) if phi2 > 0 else (phi2, phi1) # reorder
                    if utils.angle_in_range(lo, hi, otherval):
                        z_intp = (z2 - z1) / (phi2 - phi1) * (otherval - phi1) + z1
                        z_intp = round(z_intp, 3)
                        if start <= z_intp <= end or start <= z_intp <= end:
                            self.edges['cb_to_rp'][k] = None
                            li = self.prev['rp'][i][:]
                            if j in self.prev['rp'][i]:
                                li.remove(j)
                            self.prev['rp'][i] = li


    def disp_network(self, figsize=(8,8), edge_type='rp_to_cb', savepath=None, title=None, show_pm=False, upper_bound=1, plot_nid=False):
        """display the network connectivity"""


        if edge_type in ('rp_to_cb', 'cb_to_rp'):
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
            ax.set_box_aspect((2*self.rho, 2*self.rho, 1))

            # Plot the surface
            u = np.linspace(0, 2 * np.pi, 50)
            h = np.linspace(0, upper_bound, 20)
            x = np.outer(self.rho * np.cos(u), np.ones(len(h)))
            y = np.outer(self.rho * np.sin(u), np.ones(len(h)))
            z = np.outer(np.ones(len(u)), h)
            ax.plot_surface(x, y, z, color='lightgreen', alpha=.2)

            # Plot cut lines
            for line in self.cutlines:
                u = np.linspace(line[0][0], line[1][0] + np.pi/50, 50)
                h = np.linspace(line[0][1], line[1][1] + .02, 20)
                x = np.outer(self.rho * np.cos(u), np.ones(len(h)))
                y = np.outer(self.rho * np.sin(u), np.ones(len(h)))
                z = np.outer(np.ones(len(u)), h)
                ax.plot_surface(x, y, z, color='k', alpha=1)

            # Plot neurons
            for i, loc in enumerate(self.cbnet.locations):
                if loc is not None:
                    phi = loc[0]
                    z = loc[1]
                    if z <= upper_bound:
                        if not show_pm or i not in self.cbnet.pacemakers:
                            ax.scatter(self.rho * np.cos(phi),
                                    self.rho * np.sin(phi),
                                    z,
                                    color='C0', alpha=1)
                        else:
                            ax.scatter(self.rho * np.cos(phi),
                                    self.rho * np.sin(phi),
                                    z,
                                    color='C0', alpha=1, edgecolors=['r'])


            for i, loc in enumerate(self.rpnet.locations):
                if loc is not None:
                    phi = loc[0]
                    z = loc[1]
                    if z <= upper_bound:
                        ax.scatter(self.rho * np.cos(phi),
                                self.rho * np.sin(phi),
                                z,
                                color='C1', alpha=1)

            # Plot edges
            if edge_type == 'rp_to_cb':
                for edge in self.edges['rp_to_cb']:
                    if edge is not None:
                        phi1 = self.cbnet.locations[edge[0]][0]
                        z1 = self.cbnet.locations[edge[0]][1]
                        phi2 = self.rpnet.locations[edge[1]][0]
                        z2 = self.rpnet.locations[edge[1]][1]
                        if z1 <= upper_bound and z2 <= upper_bound:
                            ax.plot([self.rho * np.cos(phi1), self.rho * np.cos(phi2)],
                                    [self.rho * np.sin(phi1), self.rho * np.sin(phi2)],
                                    [z1, z2],
                                    color='grey', lw=1)
            else:
                for edge in self.edges['cb_to_rp']:
                    if edge is not None:
                        phi1 = self.rpnet.locations[edge[0]][0]
                        z1 = self.rpnet.locations[edge[0]][1]
                        phi2 = self.cbnet.locations[edge[1]][0]
                        z2 = self.cbnet.locations[edge[1]][1]
                        if z1 <= upper_bound and z2 <= upper_bound:
                            ax.plot([self.rho * np.cos(phi1), self.rho * np.cos(phi2)],
                                    [self.rho * np.sin(phi1), self.rho * np.sin(phi2)],
                                    [z1, z2],
                                    color='grey', lw=1)


            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_zlim(0, 1)

            # make the panes transparent
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            # make the grid lines transparent
            ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
            ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
            ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

            if title:
                plt.title(title)

            if savepath:
                plt.savefig(savepath, dpi=300, bbox_inches='tight')
                plt.show()

        elif edge_type == 'cb':
            self.cbnet.disp_network(show_pm=show_pm, savepath=savepath, upper_bound=upper_bound, plot_nid=plot_nid)
        elif edge_type == 'rp':
            self.rpnet.disp_network(savepath=savepath, upper_bound=upper_bound, plot_nid=plot_nid)
        else:
            raise ValueError('edge type not valid.')

if __name__ == '__main__':
    ntwk = CylEctoNetwork()
    ntwk.run()
    ntwk.cbnet.disp(style='trace')
    ntwk.rpnet.disp()