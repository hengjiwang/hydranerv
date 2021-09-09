import numpy as np
import matplotlib.pyplot as plt

class AdamsNeuron:
    """Reproduce paper Adams1985 (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1193447/)"""
    def __init__(self):
        self.dt = 0.001 # s
        self.c_m = 0.05 # uF
        self.r_m = 20 # MOhm
        self.v_max = 40 # mV
        self.v_min = -50 # mV
        self.v_th = -30 # mV
        self.t_pulse = 0.05 # s
        self.t_last = -np.inf # s
        self.t_d = 2 # s
        self.t_h = 20 # s
        self.reset()

    def reset(self):
        """reset cell"""
        self.t = 0 # s
        self.v_train = [-40] # mV
        self.i_d_train = [0] # nA
        self.i_h_train = [2] # nA

    def i_leak(self, v):
        """leak current"""
        return v / self.r_m

    def step(self):
        """step function"""

        v = self.v_train[-1]
        i_d = self.i_d_train[-1]
        i_h = self.i_h_train[-1]

        if self.t - self.t_last < self.t_pulse:
            v = self.v_max
            i_d -= self.dt * i_d / self.t_d
            i_h -= self.dt * i_h / self.t_h
        elif v == self.v_max:
            v = self.v_min
            i_d -= self.dt * i_d / self.t_d
            i_h -= self.dt * i_h / self.t_h
        elif v >= self.v_th:
            v = self.v_max
            i_d += - 1
            i_h += 0.25
            self.t_last = self.t
        else:
            v += - self.dt / self.c_m * (i_d + i_h + self.i_leak(v))
            i_d -= self.dt * i_d / self.t_d
            i_h -= self.dt * i_h / self.t_h

        self.v_train.append(v)
        self.i_d_train.append(i_d)
        self.i_h_train.append(i_h)
        self.t += self.dt

    def run(self, T):
        """run the model"""
        self.reset()
        time_axis = np.arange(self.dt, T, self.dt)
        for t in time_axis:
            self.step()

    def disp(self):
        """display simulation results"""
        time_axis = np.arange(0, len(self.v_train) * self.dt, self.dt)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15,8))
        ax1.plot(time_axis, self.v_train)
        ax1.set_ylabel('Potential (mV)')
        ax2.plot(time_axis, self.i_d_train)
        ax2.set_ylabel('$I_D$ (nA)')
        ax3.plot(time_axis, self.i_h_train)
        ax3.set_ylabel('$I_H$ (nA)')
        ax3.set_xlabel('Time (s)')
        plt.show()

if __name__ == '__main__':
    neuron = AdamsNeuron()
    neuron.run(100)
    neuron.disp()

