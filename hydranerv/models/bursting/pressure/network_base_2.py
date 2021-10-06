import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from hydranerv.utils import utils
from hydranerv.models.bursting.pressure.mech_sense_neuron import MechSenseNeuron
from hydranerv.models.bursting.pressure.pressure_controller import PressureController
from hydranerv.models.bursting.pressure.network_base import NetworkBase

class NetworkBase2:
    """A chain (1D) of gap-junctional coupled neurons"""
    def __init__(self, num=10, gc=100, dt=0.01):
        """constructor"""
        self.num = num
        self.gc = gc # uS
        self.dt = dt

        self.reset()

    def reset(self):
        """reset the states"""
        self.neurons = [MechSenseNeuron(self.dt) for _ in range(self.num)]
        self.pcontrollers = [PressureController(self.neurons[i]) for i in range(self.num)]
        for i in range(self.num):
            self.pcontrollers[i].p_train = [.9 - .1 * i]
        self.neighbors = [[] for _ in range(self.num)]
        self.make_connections()

    def make_connections(self):
        """make neuronal connections"""
        pass

    def i_couple(self, i, voltages):
        """coupling current of potential"""
        ic = 0
        v = voltages[i]
        vneighbors = self.v_neighbors(i, voltages)
        for vneighbor in vneighbors:
            ic += self.gc * (vneighbor - v)
        return ic

    def v_neighbors(self, i, voltages):
        """return the potential of neuron i's neighbors"""
        vneighbors = []
        for neighbor in self.neighbors[i]:
            vneighbors.append(voltages[neighbor])
        return vneighbors

    def step(self):
        """step function"""
        voltages = [x.v() for x in self.neurons]
        for i in range(self.num):
            neuron = self.neurons[i]
            neuron.step(p=self.pcontrollers[i].pressure(),
                        i_input=self.i_couple(i, voltages))

            self.pcontrollers[i].update()

    def disp(self, spike=True, figsize=(10, 6)):
        """display simulation results"""
        time_axis = np.arange(self.dt, len(self.pcontrollers[0].p_train) * self.dt, self.dt)
        if spike:
            fig, ax1 = plt.subplots(1, 1, figsize=figsize)
            for i in range(self.num):
                neuron = self.neurons[i]
                ax1.vlines(neuron.spike_train, i + .1, i + .9)
            ax1.set_xlabel('time (s)')
            ax1.set_ylabel('neuron #')

        else:
            fig, ax1 = plt.subplots(1, 1, figsize=figsize)
            for i in range(self.num):
                neuron = self.neurons[i]
                ax1.plot(time_axis, utils.min_max_norm(neuron.v_train[1:], .9, i))
            ax1.set_xlabel('time (s)')
            ax1.set_ylabel('neuron #')

        for i in range(self.num):
            ax1.plot(time_axis, utils.min_max_norm(self.pcontrollers[i].p_train[1:], .9, self.num), 'k--')
        ax1.set_ylim(0, self.num + 1)
        plt.show()

    def run(self, t_total):
        """run the chain model"""
        self.reset()
        time_axis = np.arange(self.dt, t_total, self.dt)
        for t in time_axis:
            self.step()