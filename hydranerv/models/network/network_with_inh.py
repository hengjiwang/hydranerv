from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from hydranerv.utils import utils
from tqdm import tqdm
from hydranerv.models.network.network import Network

class NetworkWithInh(Network):
    """a network model with periodical inhibition"""
    def __init__(self, num=10, edges=[], gc=1, dt=.01, tmax=1000, pacemakers=[0], t_ref=.1,
                 conn_type='gap_junction', t_syn=.01, wnoise=0, is_semi_pm=False, seed=0,
                 tau_inh=5, a_inh=100, interval=20):
        """constructor"""
        super().__init__(num, edges, gc, dt, tmax, pacemakers, t_ref, conn_type,
                        t_syn, wnoise, is_semi_pm, seed)
        self.tau_inh = tau_inh
        self.a_inh = a_inh
        self.interval = interval
        self.inh_moments = range(self.interval, self.tmax, self.interval)
        self.i_inh_train = [0]

    def step(self, stim_nrns=set(), stim_type=None):
        """step function"""
        voltages = [x.v() for x in self.neurons]

        # update inhibition current
        i_inh = self.i_inh_train[-1]
        for t_inh in self.inh_moments:
            if np.abs(self.t - t_inh) < self.dt / 2:
                i_inh += self.a_inh
        i_inh -= self.dt * i_inh / self.tau_inh

        for i, neuron in enumerate(self.neurons):
            if i in stim_nrns:
                neuron.step(i_ex = self.i_c(i, voltages) - i_inh)
            else:
                neuron.step(i_ex = self.i_c(i, voltages))

        self.t += self.dt
        self.i_inh_train.append(i_inh)
