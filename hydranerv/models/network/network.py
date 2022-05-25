from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from hydranerv.utils import utils
from tqdm import tqdm
from hydranerv.models.network.neuron import Neuron

class Network:
    """a model for neuronal networks"""
    def __init__(self, num=10, edges=[], gc=1, dt=.01, tmax=1000, pacemakers=[0], t_ref=.1,
                 conn_type='gap_junction', t_syn=.01, wnoise=0, is_semi_pm=False, seed=0):
        """constructor"""
        self.num = num
        self.edges = edges
        self.gc = gc
        self.dt = dt
        self.tmax = tmax
        self.seed = seed
        self.pacemakers = pacemakers
        self.t_ref = t_ref
        self.conn_type = conn_type
        self.t_syn = t_syn
        self.wnoise = wnoise
        self.is_semi_pm = is_semi_pm
        self.a_stim = 200000
        self.i_stim = 10000
        self.t_stim = 1
        self.setup()
        self.reset()

    def setup(self):
        """set up the structure"""
        self.neurons = []
        np.random.seed(self.seed)
        for i in range(self.num):
            if i in self.pacemakers:
                nrn = Neuron(self.dt, self.tmax, np.random.uniform(-self.wnoise, self.wnoise), True, self.t_ref)
                if self.is_semi_pm:
                    nrn.k_in = 2 * nrn.k_in
                self.neurons.append(nrn)
            elif self.is_semi_pm:
                nrn = Neuron(self.dt, self.tmax, np.random.uniform(-self.wnoise, self.wnoise), True, self.t_ref)
                self.neurons.append(nrn)
            else:
                self.neurons.append(Neuron(self.dt, self.tmax, np.random.uniform(-self.wnoise, self.wnoise), False, self.t_ref))

        # Construct connections
        self.set_connections()

    def reset(self):
        """reset the state"""
        # Construct neurons
        self.t = 0
        for neuron in self.neurons:
            neuron.reset()

    def set_connections(self, add_edges=[]):
        """set connections for neurons"""
        self.neighbors = [[] for _ in range(self.num)]
        for edge in add_edges:
            if edge not in self.edges:
                self.edges.append(edge)
        for edge in self.edges:
            self.neighbors[edge[0]].append(edge[1])
            self.neighbors[edge[1]].append(edge[0])

    def add_edge(self, edge):
        if edge not in self.edges:
            self.edges.append(edge)
            self.neighbors[edge[0]].append(edge[1])
            self.neighbors[edge[1]].append(edge[0])

    def i_c(self, i, voltages):
        """input currents"""

        ic = 0
        v = voltages[i]
        neighbors = self.neighbors[i]
        for neighbor in neighbors:

            if self.conn_type == 'gap_junction':
                vneighbor = voltages[neighbor] if voltages[neighbor] < self.neurons[neighbor].v_th else self.neurons[neighbor].v_spike
                ic += self.gc * (vneighbor - v)
            elif self.conn_type == 'synapse':
                if self.t < self.neurons[neighbor].t_last + self.t_syn:
                    ic += self.gc

        return ic

    def step(self, stim_nrns=set(), stim_type='mechanical'):
        """step function"""
        voltages = [x.v() for x in self.neurons]
        for i, neuron in enumerate(self.neurons):
            if i in stim_nrns:
                if stim_type == 'mechanical':
                    neuron.step(self.i_c(i, voltages), mech_stim=self.a_stim)
                elif stim_type == 'electrical':
                    neuron.step(self.i_c(i, voltages) + self.i_stim)
                # neuron.step(self.i_c(i, voltages), mech_stim=self.a_stim)
                # print('neuron #' + str(i) + ' is stimulated at ' + str(round(self.t + self.dt, 5)) + 's')
            else:
                neuron.step(self.i_c(i, voltages))
        self.t += self.dt

    def run(self, stim={}, stim_type='mechanical'):
        """run simulation"""
        self.reset()
        time_axis = np.arange(self.dt, self.tmax, self.dt)
        for t in tqdm(time_axis):
            stim_nrns = []
            for t_st in stim:
                if 0 <= t - t_st < self.t_stim:
                    stim_nrns = set(stim[t_st])
                    break
            self.step(stim_nrns, stim_type)

    def disp(self, figsize=(10, 6), xlim=None, style='spike', ineurons=None, skip=1, savefig=None, dpi=300):
        """display simulation results"""

        ineurons = range(self.num) if ineurons is None else ineurons

        if style == 'spike':
            fig, ax1 = plt.subplots(1, 1, figsize=figsize)
            for i in ineurons:
                neuron = self.neurons[i]
                ax1.vlines(neuron.spikes, i + .1, i + .9, lw=1, color='k')

        elif style == 'trace':
            time_axis = np.arange(0, self.tmax, self.dt)
            fig, ax1 = plt.subplots(1, 1, figsize=figsize)
            for i in ineurons:
                neuron = self.neurons[i]
                ax1.plot(time_axis[skip::skip], [(x + 75) / 110 + i for x in neuron.v_train[skip::skip]], lw=.75)
                # ax1.plot(time_axis[skip::skip], np.array(neuron.v_train[skip::skip]) )
        # ax1.plot(time_axis, utils.min_max_norm(self.pcontroller.p_train[1:], .9, self.num), 'k--')
        ax1.set_ylim(0, self.num + 1)
        if xlim:
            ax1.set_xlim(xlim[0], xlim[1])
        else:
            ax1.set_xlim(0, self.tmax)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel('time (s)', fontsize=20)
        plt.ylabel('neuron #', fontsize=20)
        if savefig:
            plt.savefig(savefig, dpi=dpi, bbox_inches='tight')
        plt.show()

    def disp_conn_mat(self):
        """display the connectivity matrix"""
        conn_mat = np.zeros((self.num, self.num))
        for i in range(self.num):
            for j in self.neighbors[i]:
                conn_mat[i, j] = 1

        plt.figure(figsize=(8, 8))
        plt.imshow(conn_mat, cmap='binary')
        plt.show()

if __name__ == '__main__':
    ntwk = Network(num=10, gc=500, pacemakers=[0], conn_type='gap_junction')
    ntwk.set_connections(add_edges=[(0, 1)])
    ntwk.run()
    ntwk.disp(style='trace')

