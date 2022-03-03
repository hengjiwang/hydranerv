import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import k_means
from sympy import divisor_sigma

class Neuron:
    """a lif-based model for hydra cb neuron"""
    def __init__(self, dt=.01, tmax=1000):
        """configurator"""

        # Simulation parameters
        self.tmax = tmax
        self.dt = dt

        # LIF parameters
        self.dt = dt # s
        self.c_m = 50 # nF
        self.v_th = -55 # mV
        self.v_r = -75 # mV
        self.g_l = 15 # nS
        self.e_l = self.v_r
        self.v_spike = 20 # mV

        # Stress parameters
        self.tau_p = 5 # s
        self.k_in = 50 # Pa/s
        self.k_a = 5150 # Pa
        self.k_e = 600 # Pa

        # PIEZO channel parameters
        self.g_s = 25 # nS
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
        self.sigma_a_train = []
        self.sigma_w_train = []
        self.spikes = []

        # Reset values
        self.t = 0 # s
        self.t_last = - np.inf # s
        self.v_train.append(self.v_r)
        self.sigma_a_train.append(0)
        self.sigma_w_train.append(25000)

    def v(self):
        """get v"""
        return self.v_train[-1]

    def sigma_a(self):
        """get active stress"""
        return self.sigma_a_train[-1]

    def sigma_w(self):
        """get water stress"""
        return self.sigma_w_train[-1]

    def sigma_m(self):
        """get stress"""
        return self.sigma_a_train[-1] + self.sigma_w_train[-1]

    def i_l(self):
        """leak current"""
        return self.g_l * (self.v() - self.e_l)

    def i_s(self):
        """mechanosensitive current"""
        return self.g_s / (1 + self.k_b * np.exp(- self.s * (self.sigma_m() / self.m) ** self.q)) * (self.v() - self.e_s)

    def step(self):
        """step function"""
        v = self.v()
        sigma_a = self.sigma_a()
        sigma_w = self.sigma_w()
        if v < self.v_th:

            # Derivatives
            dv = 1 / self.c_m * (- self.i_l() - self.i_s())
            da = - self.sigma_a() / self.tau_p
            dw = self.k_in

            # Update new values
            v += dv * self.dt
            sigma_a += da * self.dt
            sigma_w += dw * self.dt

        else:

            self.v_train[-1] = self.v_spike
            v = self.v_r
            sigma_a += self.k_a
            sigma_w += - self.k_e
            self.spikes.append(self.t * self.dt)
            self.t_last = self.t

        # Update neuron state
        self.v_train.append(v)
        self.sigma_a_train.append(sigma_a)
        self.sigma_w_train.append(sigma_w)
        self.t += self.dt

    def run(self):
        """run simulation"""
        self.reset()
        time_axis = np.arange(self.dt, self.tmax, self.dt)
        for _ in time_axis:
            self.step()

    def disp(self):
        """display simulation results"""
        time_axis = np.arange(self.dt, self.tmax, self.dt)
        plt.figure()
        plt.plot(time_axis, self.v_train[1:])
        plt.xlim(550, 950)
        plt.xlabel('time (s)')
        plt.ylabel('membrane potential (mV)')
        plt.show()

if __name__ == '__main__':
    neuron = Neuron()
    neuron.run()
    neuron.disp()