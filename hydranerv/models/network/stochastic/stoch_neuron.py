import numpy as np
import matplotlib.pyplot as plt
from hydranerv.models.network.neuron import Neuron

class StochNeuron(Neuron):
    """a stochastic version neuron model"""
    def __init__(self, dt=.01, tmax=1000, wnoise=0, ispacemaker=True, t_ref=.1):
        """constructor"""
        super().__init__(dt, tmax, wnoise, ispacemaker, t_ref)

    def i_s(self, mech_stim):
        """mechanosensitive current"""
        p_o = 1 / (1 + self.k_b * np.exp(- self.s * ((self.sigma_m() + mech_stim) / self.m) ** self.q))
        if self.ispacemaker and np.random.rand() < p_o:
            return self.g_s * (self.v() - self.e_s)
        else:
            return 0

if __name__ == '__main__':
    nrn = StochNeuron()
    nrn.run()
    nrn.disp(figsize=(10,3))