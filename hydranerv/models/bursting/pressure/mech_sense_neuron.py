import numpy as np
import matplotlib.pyplot as plt
from hydranerv.models.bursting.pressure.lif_neuron import LIFNeuron

class MechSenseNeuron(LIFNeuron):
    """A modified LIF neuron with mechanosensitive ion channel"""
    def __init__(self, dt=0.01):
        super().__init__(dt)
        self.p_th = 1 # Pa
        self.k_p = 1 # nA/Pa
        self.reset()

    def i_mem(self, p, i_input):
        """membrane channel current"""
        i_mech = self.k_p * (p - self.p_th) if p > self.p_th else 0
        return i_mech + i_input

    def step(self, p=1.6, i_input=0):
        """step function"""

        v = self.v_train[-1]
        i_mem = self.i_mem_train[-1]

        # Update potential
        v = self.update_v(v, i_mem)
        self.v_train.append(v)

        # Update membrane current
        self.i_mem_train.append(self.i_mem(p, i_input))

        # Update time
        self.t += self.dt

if __name__ == '__main__':
    neuron = MechSenseNeuron(0.02)
    neuron.run(200)
    neuron.disp()


