import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from hydranerv.utils import utils

class BarretoNeuron:
    """Reproduce paper Barreto & Cressman 2011 (https://pubmed.ncbi.nlm.nih.gov/22654181/)"""
    def __init__(self, dt=0.001):
        """constructor"""
        self.dt = dt * 1000 # ms
        self.c_m = 1 # uF/cm^2
        self.g_na = 100 # mS/cm^2
        self.g_nal = 0.0175 # mS/cm^2
        self.g_k = 40 # mS/cm^2
        self.g_kl = 0.05 # mS/cm^2
        self.g_cll = 0.05 # mS/cm^2
        self.e_cl = 26.64 * np.log(6 / 130) # mV
        self.beta = 7 # (ratio of the intra and extra volume)
        self.rho = 1.25 # mM/s
        self.glia = 200 / 3 # mM/s
        self.epsilon = 4 / 3 # Hz
        self.k_bath = 8 # mM
        self.gamma = 0.044494542 # (unit conversion factor: current -> mM/s)
        self.tau = 1000 # (balances the time units)
        self.phi = 3.0

    # Sodium
    def i_na(self, v, e_na, m, h):
        """Na+ current"""
        return self.g_na * m ** 3 * h * (v - e_na) + self.g_nal * (v - e_na)

    def e_na(self, na_i, na_o):
        """Na+ reversal potential"""
        return 26.64 * np.log(na_o / na_i)

    def na_o(self, na_i):
        """extracellular Na+ concentration"""
        return 144 - self.beta * (na_i - 18)

    def m_inf(self, v):
        """Na+ activation"""
        alpha = self.alpha_m(v)
        beta = self.beta_m(v)
        return alpha / (alpha + beta)

    def alpha_m(self, v):
        return 0.1 * (v + 30) / (1 - np.exp(- 0.1 * (v + 30)))

    def beta_m(self, v):
        return 4 * np.exp(- (v + 55) / 18)

    def alpha_h(self, v):
        return 0.07 * np.exp(- (v + 44) / 20)

    def beta_h(self, v):
        return 1 / (1 + np.exp(- 0.1 * (v + 14)))

    # Potassium
    def i_k(self, v, e_k, n):
        """K+ current"""
        return self.g_k * n ** 4 * (v - e_k) + self.g_kl * (v - e_k)

    def e_k(self, k_i, k_o):
        """K+ reversal potential"""
        return 26.64 * np.log(k_o / k_i)

    def k_i(self, na_i):
        """intracellular K+ concentration"""
        return 140 + (18 - na_i)

    def alpha_n(self, v):
        return 0.01 * (v + 34) / (1 - np.exp(- 0.1 * (v + 34)))

    def beta_n(self, v):
        return 0.125 * np.exp(- (v + 44) / 80)

    # Chloride
    def i_cl(self, v):
        """Cl- current"""
        return self.g_cll * (v - self.e_cl)

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

    def calc_currents(self, v, h, n, k_o, na_i):
        return (self.i_na(v, self.e_na(na_i, self.na_o(na_i)), self.m_inf(v), h),
                self.i_k(v, self.e_k(self.k_i(na_i), k_o), n),
                self.i_cl(v),
                self.i_pump(na_i, k_o),
                self.i_glia(k_o),
                self.i_diffusion(k_o))

    def reset(self):
        """reset the initiation values of variables"""
        self.v0 = - 67.37362807007007436
        self.h0 = 0.9791958267098079
        self.n0 = 0.06841365960747725
        self.k_o0 = 3.921555549814548
        self.na_i0 = 18.50771473212327

    def rhs(self, y, t):
        """right-hand side"""
        v, h, n, k_o, na_i = y

        i_na, i_k, i_cl, i_pump, i_glia, i_diffusion = self.calc_currents(v, h, n, k_o, na_i)
        dvdt = - 1 / self.c_m * (i_na + i_k + i_cl)
        dhdt = self.phi * (self.alpha_h(v) * (1 - h) - self.beta_h(v) * h)
        dndt = self.phi * (self.alpha_n(v) * (1 - n) - self.beta_n(v) * n)
        dkodt = 1 / self.tau * (self.gamma * self.beta * i_k - 2 * self.beta * i_pump - i_glia - i_diffusion)
        dnaidt = 1 / self.tau * (- self.gamma * i_na - 3 * i_pump)

        return np.array([dvdt, dhdt, dndt, dkodt, dnaidt])

    def run(self, t_total, disp=False):
        """run the model"""
        t_total_ms = t_total * 1000
        self.reset()
        y0 = [self.v0, self.h0, self.n0, self.k_o0, self.na_i0]
        sol = odeint(self.rhs, y0, np.linspace(0, t_total_ms - self.dt, int(t_total_ms / self.dt)))
        # sol = utils.euler_odeint(self.rhs, y0, t_total, self.dt)

        if disp:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 5))
            time_axis = np.linspace(0, t_total - self.dt, int(t_total_ms / self.dt))
            ax1.plot(time_axis, sol[:, 0])
            ax1.set_ylabel('Potential (mV)')
            ax2.plot(time_axis, sol[:, 3])
            ax2.set_ylabel('K_o (mM)')
            ax3.plot(time_axis, sol[:, 4])
            ax3.set_ylabel('Na_i (mM)')
            ax3.set_xlabel('Time (s)')
            plt.show()

        return sol


if __name__ == '__main__':
    t_total = 100 # s
    dt = 0.00001 # s
    neuron = BarretoNeuron(dt)
    sol = neuron.run(t_total, disp=True)



