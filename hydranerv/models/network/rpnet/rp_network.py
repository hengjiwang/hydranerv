from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from hydranerv.utils import utils
from tqdm import tqdm
from hydranerv.models.network.network import Network
from hydranerv.models.network.rpnet.rp_neuron import RPNeuron


class RPNetwork(Network):
    """a network model composed of RP neurons"""

    def __init__(self, num=10, edges=[], gc=1, dt=.01, tmax=1000, t_ref=.1,
                 conn_type='gap_junction', t_syn=.01, seed=0):
        super().__init__(num, edges, gc, dt, tmax, [],
                         t_ref, conn_type, t_syn, 0, False, seed)
        self.setup()
        self.reset()

    def setup(self):
        """set up the structure"""
        self.neurons = []
        self.edges = [] # TODO: this is because Network class constructor previously execuated setup
        np.random.seed(self.seed)
        for i in range(self.num):
            self.neurons.append(RPNeuron(self.dt, self.tmax, self.t_ref))

        # Construct connections
        self.set_connections()

    def step(self, sigma_m_li):
        """step function"""
        voltages = [x.v() for x in self.neurons]
        for i, neuron in enumerate(self.neurons):
            neuron.step(sigma_m_li[i], self.i_c(i, voltages))
        self.t += self.dt

    def run(self, sigma_m_li):
        """run simulation"""
        self.reset()
        time_axis = np.arange(self.dt, self.tmax, self.dt)
        for t in tqdm(time_axis):
            self.step(sigma_m_li)


if __name__ == '__main__':

    sigma_m_li = [35000] * 10

    ntwk = RPNetwork(num=10, gc=500, conn_type='gap_junction')
    ntwk.set_connections(add_edges=[(0, 1)])
    ntwk.run(sigma_m_li)
    ntwk.disp(style='trace')
