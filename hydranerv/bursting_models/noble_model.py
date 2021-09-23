import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from hydranerv.utils import utils

class NobleModel:
    """Reproduce Noble 1962 (10.1113/jphysiol.1962.sp006849)"""
    def __init__(self):
        """constructor"""
        self.dt = 0.001 # s
        self.c_m = 12 # # uF/cm^2
        self.g_leak = 75 # uS/cm^2
        self.e_leak = -60 # mV
        self.reset()

    # Sodium
    def i_na(self, v, m, h):
        """Na+ current"""
        return (400000 * m ** 3 * h + 140) * (v - 40)

    def alpha_m(self, v):
        return 100 * (- v - 48) / (np.exp((- v - 48) / 15) - 1)

    def beta_m(self, v):
        return 120 * (v + 8) / (np.exp((v + 8) / 5) - 1)

    def alpha_h(self, v):
        return 170 * np.exp((- v - 90) / 20)

    def beta_h(self, v):
        return 1000 / (1 + np.exp((- v - 42) / 10))

    # Potassium
    def i_k(self, v, n):
        """K+ current"""
        g_k1 = 1200 * np.exp((- v - 90) / 50) + 15 * np.exp((v + 90) / 60)
        g_k2 = 1200 * n ** 4
        return (g_k1 + g_k2) * (v + 100)

    def alpha_n(self, v):
        return 0.1 * (- v - 50) / (np.exp((- v - 50) / 10) - 1)

    def beta_n(self, v):
        return 2 * np.exp((- v - 90) / 80)

    # Leak
    def i_leak(self, v):
        """leak current"""
        return self.g_leak * (v - self.e_leak)

    # Stimulation
    def i_stim(self, t, t_start, t_end):
        """stimulation current"""
        if t >= t_start and t <= t_end:
            return 20
        return 0

    # Others
    def calc_currents(self, v, m, h, n):
        """calculate currents"""
        return (self.i_na(v, m, h),
                self.i_k(v, n),
                self.i_leak(v))

    def reset(self):
        """reset the initiation values of variables"""
        self.v0 = - 75 # mV
        self.m0 = 0.0537
        self.h0 = 0.772
        self.n0 = 0.0971

    def rhs(self, y, t):
        """right-hand side"""
        v, m, h, n = y

        i_na, i_k, i_leak = self.calc_currents(v, m, h, n)
        dvdt = - 1 / self.c_m * (i_na + i_k + i_leak) # - self.i_stim(t, 1, 9))
        dmdt = self.alpha_m(v) * (1 - m) - self.beta_m(v) * m
        dhdt = self.alpha_h(v) * (1 - h) - self.beta_h(v) * h
        dndt = self.alpha_n(v) * (1 - n) - self.beta_n(v) * n

        return np.array([dvdt, dmdt, dhdt, dndt])

    def run(self, t_total, disp=False):
        """run the model"""
        self.reset()
        y0 = [self.v0, self.m0, self.h0, self.n0]
        sol = odeint(self.rhs, y0, np.linspace(0, t_total - self.dt, int(t_total / self.dt)))

        if disp:
            time_axis = np.linspace(0, t_total - self.dt, int(t_total / self.dt))
            plt.figure()
            plt.plot(time_axis, sol[:, 0])
            plt.xlabel("Time (s)")
            plt.ylabel("Potential (mV)")
            plt.show()

        return sol

if __name__ == '__main__':
    neuron = NobleModel()
    sol = neuron.run(t_total=10, disp=True)







