import numpy as np
import hydranerv.utils.utils as utils


class LIFNeuron:
    def __init__(self):
        """Configuration"""
        self.dt = 1
        self.v_r = 0
        self.v = self.v_r
        self.r = 1
        self.c = 100000
        self.v_train = []
        self.spike_train = []
        self.tau = self.r * self.c
        self.last_spike = -np.infty
        self.tau_ref = 4
        self.v_th = 1
        self.v_spike = 5

    def _theta_to_i_ext(self, theta):
        """Encode theta to i_ext"""
        return 0

    def _light_to_i_ext(self, light):
        """Encode light to i_ext"""
        return 0

    def rhs(self, theta, light, i_syn, i_stim):
        """dv/dt"""
        i_ext = self._theta_to_i_ext(theta) + self._light_to_i_ext(light) + i_stim
        return 1 / self.tau * (-self.v + self.r * (i_ext + i_syn))

    def step(self, t, theta, light, i_syn, i_stim):
        """Step function"""

        # If the neuron just fired, reset v and set last spike time
        if self.v == self.v_spike or t < self.last_spike + self.tau_ref:
            self.v = self.v_r
            self.spike_train.append(False)
        else:
            v = self.v + self.rhs(theta, light, i_syn, i_stim) * self.dt
            if v >= self.v_th:
                # If v reaches the threshold, it fires
                self.v = self.v_spike
                self.spike_train.append(True)
                self.last_spike = t
            else:
                self.v = v
                self.spike_train.append(False)

        self.v_train.append(self.v)


if __name__ == "__main__":
    T = 100000
    dt = 1
    i_ext_train = np.zeros(int(T/dt)+1)
    neuron = LIFNeuron()
    utils.run_neuron(T, dt, i_ext_train, neuron)










