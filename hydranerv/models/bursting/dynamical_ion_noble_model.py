import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from hydranerv.utils import utils
from hydranerv.models.basic.noble_model import NobleModel

class DynamicalIonNobleModel(NobleModel):
    """Add ion concentration dependency to reversal potentials"""
    def __init__(self):
        super().__init__()
        self.beta = 7 # (ratio of the intra and extra volume)
        self.rho = 1.25 # mM/s
        self.glia = 200 / 3 # mM/s
        self.epsilon = 4 / 3 # Hz
        self.k_bath = 8 # mM
        self.gamma = 0.000044494542 # (unit conversion factor: current -> mM/s)
        self.tau = 1 # (balances the time units)
        self.reset()

        # print(self.e_na(self.na_i0, self.na_o(self.na_i0)))
        # print(self.e_k(self.k_i(self.k_o0), self.k_o0))

    # Sodium
    def i_na(self, v, e_na, m, h):
        """Na+ current"""
        return (400000 * m ** 3 * h + 140) * (v - e_na)

    def e_na(self, na_i, na_o):
        """Na+ reversal potential"""
        return 26.64 * np.log(na_o / na_i)
        # return 40

    def na_o(self, na_i):
        """extracellular Na+ concentration"""
        return 144 - self.beta * (na_i - 18)

    # Potassium
    def i_k(self, v, e_k, n):
        """K+ current"""
        g_k1 = 1200 * np.exp((- v - 90) / 50) + 15 * np.exp((v + 90) / 60)
        g_k2 = 1200 * n ** 4
        return (g_k1 + g_k2) * (v - e_k)

    def e_k(self, k_i, k_o):
        """K+ reversal potential"""
        return 26.64 * np.log(k_o / k_i)
        # return -100

    def k_i(self, na_i):
        """intracellular K+ concentration"""
        return 140 + (18 - na_i)

    # Others
    def i_pump(self, na_i, k_o):
        """pump current"""
        return self.rho / (1 + np.exp((25 - na_i) / 3)) / (1 + np.exp(5.5 - k_o))

    def i_glia(self, k_o):
        """glia current"""
        return self.glia / (1 + np.exp((18 - k_o) / 2.5))

    def i_diffusion(self, k_o):
        """diffusion molar current"""
        return self.epsilon * (k_o - self.k_bath)

    def calc_currents(self, v, m, h, n, k_o, na_i):
        """calculate currents"""
        return (self.i_na(v, self.e_na(na_i, self.na_o(na_i)), m, h),
                self.i_k(v, self.e_k(self.k_i(na_i), k_o), n),
                self.i_leak(v),
                self.i_pump(na_i, k_o),
                self.i_glia(k_o),
                self.i_diffusion(k_o))

    def reset(self):
        """reset the initiation values of variables"""
        super().reset()
        self.k_o0 = 7
        self.na_i0 = 18.50771473212327

    def rhs(self, y, t):
        """right-hand side"""
        v, m, h, n, k_o, na_i = y

        i_na, i_k, i_leak, i_pump, i_glia, i_diffusion = self.calc_currents(v, m, h, n, k_o, na_i)
        dvdt = - 1 / self.c_m * (i_na + i_k + i_leak)
        dmdt = (self.alpha_m(v) * (1 - m) - self.beta_m(v) * m)
        dhdt = (self.alpha_h(v) * (1 - h) - self.beta_h(v) * h)
        dndt = (self.alpha_n(v) * (1 - n) - self.beta_n(v) * n)
        dkodt = 1 / self.tau * (self.gamma * self.beta * i_k - 2 * self.beta * i_pump - i_glia - i_diffusion)
        dnaidt = 1 / self.tau * (- self.gamma * i_na - 3 * i_pump)

        return np.array([dvdt, dmdt, dhdt, dndt, dkodt, dnaidt])

    def run(self, t_total, disp=False):
        """run the model"""
        self.reset()
        y0 = [self.v0, self.m0, self.h0, self.n0, self.k_o0, self.na_i0]
        sol = odeint(self.rhs, y0, np.linspace(0, t_total - self.dt, int(t_total / self.dt)))

        if disp:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 5))
            time_axis = np.linspace(0, t_total - self.dt, int(t_total / self.dt))
            ax1.plot(time_axis, sol[:, 0])
            ax1.set_ylabel('Potential (mV)')
            ax2.plot(time_axis, sol[:, 4])
            ax2.set_ylabel('K_o (mM)')
            ax3.plot(time_axis, sol[:, 5])
            ax3.set_ylabel('Na_i (mM)')
            ax3.set_xlabel('Time (s)')
            plt.show()

        return sol

if __name__ == '__main__':
    neuron = DynamicalIonNobleModel()
    sol = neuron.run(t_total=10, disp=True)
