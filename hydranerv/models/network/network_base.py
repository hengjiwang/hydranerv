import numpy as np
import matplotlib.pyplot as plt
from hydranerv.utils import utils
from hydranerv.models.network.neuron import Neuron

class NetworkBase:
    """a base model for neuronal networks"""
    def __init__(self, num=20, gc=1, dt=.01, tmax=1000, pacemakers=[0], conn_type='gap_junction', t_syn=.01):
        """constructor"""
        self.num = num
        self.gc = gc
        self.dt = dt
        self.tmax = tmax
        self.pacemakers = pacemakers
        self.conn_type = conn_type
        self.t_syn = t_syn
        self.reset()

    def reset(self):
        """reset the state"""
        # Construct neurons
        self.t = 0
        self.neurons = []
        for i in range(self.num):
            if i in self.pacemakers:
                self.neurons.append(Neuron(self.dt, self.tmax, 0, True))
            else:
                self.neurons.append(Neuron(self.dt, self.tmax, 0, False))
        # Construct connections
        self.neighbors = [[] for _ in range(self.num)]
        self.set_connections()

    def set_connections(self):
        """set connections for neurons"""
        # pass
        self.neighbors[0].append(1)
        self.neighbors[1].append(0)


    def i_c(self, i, voltages):
        """input currents"""

        ic = 0
        v = voltages[i]
        neighbors = self.neighbors[i]
        for neighbor in neighbors:

            if self.conn_type == 'gap_junction':
                vneighbor = voltages[neighbor] if voltages[neighbor] < self.neurons[neighbor].v_th else 20
                ic += self.gc * (vneighbor - v)
            elif self.conn_type == 'synapse':
                if self.t < self.neurons[neighbor].t_last + self.t_syn:
                    ic += self.gc

        return ic

    def step(self):
        """step function"""
        voltages = [x.v() for x in self.neurons]
        for i, neuron in enumerate(self.neurons):
            neuron.step(self.i_c(i, voltages))
        self.t += self.dt

    def run(self):
        """run simulation"""
        self.reset()
        time_axis = np.arange(self.dt, self.tmax, self.dt)
        for t in time_axis:
            self.step()

    def disp(self, figsize=(10, 6), style='spike', ineurons=None, skip=1):
        """display simulation results"""

        ineurons = range(self.num) if ineurons is None else ineurons

        if style == 'spike':
            fig, ax1 = plt.subplots(1, 1, figsize=figsize)
            for i in ineurons:
                neuron = self.neurons[i]
                ax1.vlines(neuron.spikes, i + .1, i + .9)
            ax1.set_xlabel('time (s)')
            ax1.set_ylabel('neuron #')

        elif style == 'trace':
            time_axis = np.arange(0, self.tmax, self.dt)
            fig, ax1 = plt.subplots(1, 1, figsize=figsize)
            for i in ineurons:
                neuron = self.neurons[i]
                ax1.plot(time_axis[skip::skip], [(x + 75) / 110 + i for x in neuron.v_train[skip::skip]], lw=.75)
                # ax1.plot(time_axis[skip::skip], np.array(neuron.v_train[skip::skip]) )
            ax1.set_xlabel('time (s)')
            # ax1.set_ylabel('neuron #')

        # ax1.plot(time_axis, utils.min_max_norm(self.pcontroller.p_train[1:], .9, self.num), 'k--')
        # ax1.set_ylim(0, self.num + 1)
        ax1.set_xlim(600, 950)
        plt.show()

    def disp_conn_mat(self):
        """display the connectivity matrix"""
        conn_mat = np.zeros((self.num, self.num))
        for i in range(self.num):
            for j in self.neighbors[i]:
                conn_mat[i, j] = 1

        plt.figure()
        plt.imshow(conn_mat, cmap='binary')
        plt.show()

if __name__ == '__main__':
    network = NetworkBase(num=10, gc=200, pacemakers=[0], conn_type='gap_junction')
    network.run()
    network.disp(style='trace')

