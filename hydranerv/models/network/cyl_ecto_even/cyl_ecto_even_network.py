import numpy as np
import matplotlib.pyplot as plt
from hydranerv.models.network.cyl_even_network import CylEvenNetwork
from hydranerv.models.network.rpnet.cyl_rp_even_network import CylRPEvenNetwork
from collections import defaultdict
from hydranerv.utils import utils
from tqdm import tqdm

class CylEctoEvenNetwork:
    """a cylindrical network model including both cb and rp neurons"""
    def __init__(self,
                 dt=.01,
                 tmax=1000,
                 num_lon=200,
                 num_cir=200,
                 gc_cb=100,
                 gc_rp=100,
                 pm_cb=[],
                 t_ref=.1,
                 edges=None,
                 tau_inh_cb=5,
                 tau_inh_rp=5,
                 a_inh_cb=50,
                 a_inh_rp=100,
                 prob_cb_to_rb=.5,
                 prob_rp_to_cb=.5,
                 prob_random=0,
                 method='two_ends',
                 seed=0):
        """constructor"""
        self.dt = dt
        self.tmax = tmax
        self.cbnet = CylEvenNetwork(num_lon=num_lon,
                                    num_cir=num_cir,
                                    gc=gc_cb,
                                    dt=dt,
                                    tmax=tmax,
                                    pacemakers=pm_cb,
                                    t_ref=t_ref,
                                    conn_type='gap_junction',
                                    t_syn=.01,
                                    wnoise=0,
                                    is_semi_pm=False,
                                    prob_lon=1,
                                    prob_cir=.2,
                                    rho=.5,
                                    seed=seed
                                    )
        self.rpnet = CylRPEvenNetwork(num_lon=num_lon,
                                      num_cir=num_cir,
                                      gc=gc_rp,
                                      dt=dt,
                                      tmax=tmax,
                                      t_ref=t_ref,
                                      conn_type="gap_junction",
                                      t_syn=.01,
                                      prob_lon=1,
                                      prob_cir=.2,
                                      phi_offset=np.pi/10,
                                      rho=.5,
                                      seed=seed)
        self.tau_inh_cb = tau_inh_cb
        self.tau_inh_rp = tau_inh_rp
        self.a_inh_cb = a_inh_cb
        self.a_inh_rp = a_inh_rp
        self.edges = edges
        self.rho = .5
        self.prob_cb_to_rp = prob_cb_to_rb
        self.prob_rp_to_cb = prob_rp_to_cb
        self.prob_random = prob_random
        self.i_inh = defaultdict(defaultdict)
        self.method = method
        self.setup()
        self.reset()
        self.regions_cut = set()

    def define_edges(self):
        """define cross edges"""

        self.edges = defaultdict(list)

        if self.method == 'two_ends':
        # Config 1 -- two ends mutual inhibition
            # add edges on hypostomal end (majorly cb -> rp)
            for i in range(2*self.cbnet.num_cir):
                for j in range(2*self.rpnet.num_cir):
                    if np.random.rand() < self.prob_cb_to_rp:
                        self.edges['cb_to_rp'].append((j, i))

            # add edges on peduncular end (majorly rp -> cb)
            for i in range(self.cbnet.num - 2*self.cbnet.num_cir, self.cbnet.num):
                for j in range(self.rpnet.num - 2*self.rpnet.num_cir, self.rpnet.num):
                    if np.random.rand() < self.prob_rp_to_cb:
                        self.edges['rp_to_cb'].append((i, j))

            # randomly add edges on body column
            for i in range(self.cbnet.num):
                if np.random.rand() < self.prob_random:
                    self.edges['rp_to_cb'].append((i, i))
                if np.random.rand() < self.prob_random:
                    self.edges['cb_to_rp'].append((i, i))

        elif self.method == 'gradient':
            # Config 2 -- gradient mutual inhibition (distance decay)
            lambda_d_0 = .3
            lambda_d_1 = .3
            # add cb -> rp
            for i in range(self.cbnet.num):
                rho_cb = self.cbnet.rho
                phi_cb, z_cb = self.cbnet.locations[i]
                for j in range(self.rpnet.num):
                    rho_rp = self.rpnet.rho
                    phi_rp, z_rp = self.rpnet.locations[j]
                    dist = utils.cyl_dist((rho_cb, phi_cb, z_cb),
                                            (rho_rp, phi_rp, z_rp))

                    randnum = np.random.rand()
                    if randnum < np.exp(- dist ** 2 / 2 / (lambda_d_0 * np.exp( - z_rp)) ** 2):
                        self.edges['rp_to_cb'].append((i, j))
                    if randnum < np.exp(- dist ** 2 / 2 / (lambda_d_1 * np.exp(z_cb - 1)) ** 2):
                        self.edges['cb_to_rp'].append((j, i))

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
        voltages['cb'] = [x.v() if x is not None else None for x in self.cbnet.neurons]
        voltages['rp'] = [x.v() if x is not None else None for x in self.rpnet.neurons]

        # update cb neurons
        for i, cb in enumerate(self.cbnet.neurons):
            if cb is not None:
                i_inh_total = 0
                # print([self.rpnet.neurons[j] for j in self.prev['cb'][i]])
                for j in self.prev['cb'][i]:
                    rp = self.rpnet.neurons[j]
                    i_inh = self.i_inh['rp_to_cb'][(i, j)][-1]
                    if np.abs(self.t - self.dt - rp.t_last) < self.dt / 2:
                        i_inh += self.a_inh_rp
                    i_inh -= self.dt * i_inh / self.tau_inh_rp
                    self.i_inh['rp_to_cb'][(i, j)].append(i_inh)
                    i_inh_total += i_inh
                cb.step(self.cbnet.i_c(i, voltages['cb']) - i_inh_total)


        # update rp neurons
        for i, rp in enumerate(self.rpnet.neurons):
            if rp is not None:
                i_inh_total = 0
                for j in self.prev['rp'][i]:
                    cb = self.cbnet.neurons[j]
                    i_inh = self.i_inh['cb_to_rp'][(i, j)][-1]
                    if np.abs(self.t - self.dt - cb.t_last) < self.dt / 2:
                        i_inh += self.a_inh_cb
                    i_inh -= self.dt * i_inh / self.tau_inh_cb
                    self.i_inh['cb_to_rp'][(i, j)].append(i_inh)
                    i_inh_total += i_inh
                sigma_m = cb.sigma_m() # TODO: replace this cb to the closest cb to rp
                # TODO: temporarily use a constant sigma_m here, to use constant current in rp
                # sigma_m = 50000
                rp.step(sigma_m, self.rpnet.i_c(i, voltages['rp']) - i_inh_total)

        self.t += self.dt

    def run(self):
        """run simulation"""
        self.reset()
        time_axis = np.arange(self.dt, self.tmax, self.dt)
        for t in tqdm(time_axis):
            self.step()

    def cut(self, phi_start, phi_end, z_start, z_end):
        """cut a specified region from the shell"""

        # cut each net
        self.cbnet.cut(phi_start, phi_end, z_start, z_end)
        self.rpnet.cut(phi_start, phi_end, z_start, z_end)

        # remove edges
        for k, edge in enumerate(self.edges['rp_to_cb']):
            if edge is not None:
                i, j = edge
                if self.cbnet.neurons[i] is None or self.rpnet.neurons[j] is None:
                    self.edges['rp_to_cb'][k] = None

        for k, edge in enumerate(self.edges['cb_to_rp']):
            if edge is not None:
                j, i = edge
                if self.cbnet.neurons[i] is None or self.rpnet.neurons[j] is None:
                    self.edges['cb_to_rp'][k] = None

        # remove prev
        for i in range(self.cbnet.num):
            if self.cbnet.neurons[i] is None:
                self.prev['cb'][i] = []
            else:
                # print(self.prev['cb'][i])
                li = self.prev['cb'][i][:]
                for j in self.prev['cb'][i]:
                    if self.rpnet.neurons[j] is None:
                        li.remove(j)
                self.prev['cb'][i] = li

        for i in range(self.rpnet.num):
            if self.rpnet.neurons[i] is None:
                self.prev['rp'][i] = []
            else:
                li = self.prev['rp'][i][:]
                for j in self.prev['rp'][i]:
                    if self.cbnet.neurons[j] is None:
                        li.remove(j)
                self.prev['rp'][i] = li

        self.regions_cut.add(((phi_start, phi_end), (z_start, z_end)))


    def disp_network(self, figsize=(8,8), edge_type='rp_to_cb'):
        """display the network connectivity"""


        if edge_type in ('rp_to_cb', 'cb_to_rp'):
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
            for i, loc in enumerate(self.cbnet.locations):
                if loc is not None:
                    phi = loc[0]
                    z = loc[1]
                    ax.scatter(self.rho * np.cos(phi),
                            self.rho * np.sin(phi),
                            z,
                            color='C0', alpha=1)

            for i, loc in enumerate(self.rpnet.locations):
                if loc is not None:
                    phi = loc[0]
                    z = loc[1]
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
                        ax.plot([self.rho * np.cos(phi1), self.rho * np.cos(phi2)],
                                [self.rho * np.sin(phi1), self.rho * np.sin(phi2)],
                                [z1, z2],
                                color='grey', lw=1)


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

        elif edge_type == 'cb':
            self.cbnet.disp_network(show_pm=False)
        elif edge_type == 'rp':
            self.rpnet.disp_network()
        else:
            raise ValueError('edge type not valid.')

if __name__ == '__main__':
    ntwk = CylEctoEvenNetwork()
    ntwk.run()
    ntwk.cbnet.disp(style='trace')
    ntwk.rpnet.disp()