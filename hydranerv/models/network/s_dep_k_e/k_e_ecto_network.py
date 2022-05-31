import numpy as np
import matplotlib.pyplot as plt
from hydranerv.models.network.s_dep_k_e.k_e_network import KENetwork
from hydranerv.models.network.rpnet.rp_network import RPNetwork
from collections import defaultdict
from tqdm import tqdm

class KEEctoNetwork:
    """a network model including both cb and rp neurons"""
    def __init__(self,
                 dt=.01,
                 tmax=1000,
                 num_cb=50,
                 num_rp=50,
                 edges_cb=[],
                 edges_rp=[],
                 gc=100,
                 pm_cb=[],
                 t_ref=.1,
                 edges_rp_cb=[],
                 edges_cb_rp=[],
                 tau_inh=5,
                 a_inh=100,
                 seed=0):
        """constructor"""
        self.dt = dt
        self.tmax = tmax
        self.cbnet = KENetwork(num_cb, edges_cb, gc, dt, tmax, pm_cb, t_ref, "gap_junction", .01, 0, False, seed)
        self.rpnet = RPNetwork(num_rp, edges_rp, gc, dt, tmax, t_ref, "gap_junction", .01, seed)
        self.tau_inh = tau_inh
        self.a_inh = a_inh
        self.edges_rp_cb = edges_rp_cb
        self.edges_cb_rp = edges_cb_rp
        self.i_inh_rp_cb_dic = defaultdict(list)
        self.i_inh_cb_rp_dic = defaultdict(list)
        self.setup()
        self.reset()

    def setup(self):
        """set up the structure"""
        # set rp -> cb edges
        self.prev_cb = [[] for _ in range(self.cbnet.num)]
        for edge in self.edges_rp_cb:
            self.prev_cb[edge[0]].append(edge[1])

        # set cb -> rp edges
        self.prev_rp = [[] for _ in range(self.rpnet.num)]
        for edge in self.edges_cb_rp:
            self.prev_rp[edge[0]].append(edge[1])

    def reset(self):
        """reset the state"""
        self.t = 0
        self.cbnet.reset()
        self.rpnet.reset()
        for edge in self.edges_rp_cb:
            self.i_inh_rp_cb_dic[edge] = [0]
        for edge in self.edges_cb_rp:
            self.i_inh_cb_rp_dic[edge] = [0]

    def step(self):
        """step function"""

        voltages_cb = [x.v() for x in self.cbnet.neurons]
        voltages_rp = [x.v() for x in self.rpnet.neurons]

        # update cb neurons
        for i, cb in enumerate(self.cbnet.neurons):
            i_inh_total = 0
            for j in self.prev_cb[i]:
                rp = self.rpnet.neurons[j]
                i_inh = self.i_inh_rp_cb_dic[(i, j)][-1]
                if np.abs(self.t - self.dt - rp.t_last) < self.dt / 2:
                    i_inh += self.a_inh
                i_inh -= self.dt * i_inh / self.tau_inh
                self.i_inh_rp_cb_dic[(i, j)].append(i_inh)
                i_inh_total += i_inh
            cb.step(self.cbnet.i_c(i, voltages_cb) - i_inh_total)


        # update rp neurons
        for i, rp in enumerate(self.rpnet.neurons):
            i_inh_total = 0
            for j in self.prev_rp[i]:
                cb = self.cbnet.neurons[j]
                i_inh = self.i_inh_cb_rp_dic[(i, j)][-1]
                if np.abs(self.t - self.dt - cb.t_last) < self.dt / 2:
                    i_inh += self.a_inh
                i_inh -= self.dt * i_inh / self.tau_inh
                self.i_inh_cb_rp_dic[(i, j)].append(i_inh)
                i_inh_total += i_inh
            sigma_m = self.cbnet.neurons[0].sigma_m()
            rp.step(sigma_m, self.rpnet.i_c(i, voltages_rp) - i_inh_total)

        self.t += self.dt

    def run(self):
        """run simulation"""
        self.reset()
        time_axis = np.arange(self.dt, self.tmax, self.dt)
        for t in tqdm(time_axis):
            self.step()

if __name__ == '__main__':
    ntwk = KEEctoNetwork()
    ntwk.run()
    ntwk.cbnet.disp(style='trace')
    ntwk.rpnet.disp()