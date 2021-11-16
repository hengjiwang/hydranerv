import numpy as np
import matplotlib.pyplot as plt


class LIFNeuron:
    """leaky integrated-and-fire neuron"""
    def __init__(self, dt=0.01):
        """configurator"""
        self.dt = dt # s
        self.c_m = 0.05 # uF
        self.r_m = 20 # MOhm
        self.v_max = 20 # mV
        self.v_th = - 50 # mV
        self.v_rest = - 60 # mV
        self.t_pulse = .05 # s
        self.t_refrc = .5 # s
        self.reset()

    def reset(self):
        """reset neuron states"""
        self.t = 0 # s
        self.t_last = - np.inf # s
        self.v_train = [self.v_rest] # mV
        self.i_mem_train = [0] # nA
        self.spike_train = [] # s

    def v(self):
        """get potential"""
        return self.v_train[-1]

    def i_leak(self, v):
        """leak current"""
        return (v - self.v_rest) / self.r_m

    def i_mem(self):
        """membrane current"""
        return .6

    def update_v(self, v, i_mem):
        """update potential"""
        if self.t - self.t_last < self.t_pulse:
            v = self.v_max
        elif self.t - self.t_last < self.t_pulse + self.t_refrc:
            v = self.v_rest
        elif v >= self.v_th:
            v = self.v_max
            self.t_last = self.t
            self.spike_train.append(self.t)
        else:
            v += self.dt / self.c_m * (i_mem - self.i_leak(v))
        return v

    def step(self):
        """step function"""

        v = self.v_train[-1]
        i_mem = self.i_mem_train[-1]

        # Update potential
        v = self.update_v(v, i_mem)

        # Update i_mem
        i_mem = self.i_mem()

        self.v_train.append(v)
        self.i_mem_train.append(i_mem)
        self.t += self.dt

    def run(self, t_total):
        """run the model"""
        self.reset()
        time_axis = np.arange(self.dt, t_total, self.dt)
        for t in time_axis:
            self.step()

    def disp(self, i_mem=True):
        """display simulation results"""
        time_axis = np.arange(self.dt, len(self.v_train) * self.dt, self.dt)
        if i_mem:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
            ax1.plot(time_axis, self.v_train[1:])
            ax1.set_ylabel('Potential (mV)')
            ax2.plot(time_axis, self.i_mem_train[1:])
            ax2.set_ylabel('I_mem (nA)')
            ax2.set_xlabel('Time (s)')
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(15, 2))
            ax1.plot(time_axis, self.v_train[1:])
            ax1.set_ylabel('Potential (mV)')
        plt.show()

if __name__ == '__main__':
    neuron = LIFNeuron()
    neuron.run(100)
    neuron.disp()
