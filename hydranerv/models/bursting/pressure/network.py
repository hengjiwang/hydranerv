import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from hydranerv.utils import utils
from hydranerv.models.bursting.pressure.mech_sense_neuron import MechSenseNeuron
from hydranerv.models.bursting.pressure.pressure_controller import PressureController
from hydranerv.models.bursting.pressure.chain import Chain

class Network(Chain):

    def __init__(self, num=10, gc=100, dt=0.01):
        """constructor"""
        super().__init__(num, gc, dt)
        self.neighbors = [[] for _ in range(self.num)]
        self.make_connections()

    def make_connections(self, density=.1):
        """make connections between neurons"""
        pass

    def v_neighbors(self, i, voltages):
        vneighbors = []
        for neighbor in self.neighbors[i]:
            vneighbors.append(voltages[neighbor])
        return vneighbors

if __name__ == '__main__':
    network = Network(gc=.2)
    network.run(100)
    network.disp(spike=False)