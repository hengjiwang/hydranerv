import numpy as np
import matplotlib.pyplot as plt

class RPNeuron:
    """a lif-based model for hydra rp neuron"""
    def __init__(self, dt=.01, tmax=1000, t_ref=.1):
        """constructor"""

        # Similation parameters
        self.tmax = tmax
        self.dt = dt

        # LIF parameters
        self.c_m = 50 # nF
        self.v_th = -55 # mV
        self.v_r = -75 # mV
        self.g_l = 1 # nS
        self.e_l = self.v_r
        self.v_spike = 20 # mV
        self.t_pk = .0 # mV
        self.t_ref = t_ref # s

        # PIEZO channel parameters
        self.g_s = 5 # nS
        self.e_s = 10 # mV
        self.s = .00277 # 1/Pa
        self.k_b = 106
        self.m = 25
        self.q = 1

        # Reset state
        self.reset()

    def reset(self):
        """reset state"""
        # Memory trains
        self.v_train = []
        self.spikes = []

        # Reset values
        self.t = 0
        self.t_last = -np.inf
        self.v_train.append(self.v_r)

    def v(self):
        """get v"""
        return self.v_train[-1]

    def i_l(self):
        """leak current"""
        return self.g_l * (self.v() - self.e_l)

    def i_s(self, sigma_m):
        """mechanosensitive current"""
        return self.g_s / (1 + self.k_b * np.exp(- self.s * (sigma_m / self.m) ** self.q)) * (self.v() - self.e_s)
        # return -100

    def step(self, sigma_m, i_ext=0):
        """step function"""
        v = self.v()

        if self.t < self.t_last + self.t_pk:

            v = self.v_spike

        elif self.t < self.t_last + self.t_pk + self.t_ref or v == self.v_spike:

            v = self.v_r

        else:

            dv = 1 / self.c_m * (- self.i_l() - self.i_s(sigma_m) + i_ext)
            v += dv * self.dt

            if v > self.v_th:

                v = self.v_spike
                self.spikes.append(self.t)
                self.t_last = self.t

        self.v_train.append(v)
        self.t += self.dt

    def run(self, sigma_m):
        """run simulation"""
        self.reset()
        time_axis = np.arange(self.dt, self.tmax, self.dt)
        for _ in time_axis:
            self.step(sigma_m)

    def disp(self, figsize=None):
        """display simulation results"""
        time_axis = np.arange(self.dt, self.tmax, self.dt)
        if not figsize:
            fig = plt.figure()
        else:
            fig = plt.figure(figsize=figsize)

        plt.plot(time_axis, self.v_train[1:])
        # plt.xlim(550, 950)
        plt.xlabel('time (s)')
        plt.ylabel('membrane potential (mV)')
        plt.show()

if __name__ == '__main__':
    nrn = RPNeuron(tmax=200)
    nrn.run(sigma_m=35000)
    nrn.disp()