import numpy as np
import matplotlib.pyplot as plt
from hydranerv.models.bursting.pressure.lif_neuron import LIFNeuron

class MechSenseNeuron(LIFNeuron):
    """A modified LIF neuron with mechanosensitive ion channel"""
    def __init__(self):
        super().__init__()
        self.p_th = 1 # Pa
        self.k_p = 1 # nA/Pa
        self.r_p_inc = .1 # Pa/s
        # self.r_p_dec = 0.01 # Pa/s
        self.r_d = .1 # Pa/s
        self.r_h = .05 # Pa/s
        self.tau_d = 2 # s
        self.tau_h = 10 # s
        self.reset()

    def reset(self):
        """reset neuron states"""
        super().reset()
        self.acc_train = [0]
        self.p_train = [0.9] # Pa

    def i_ext(self, p):
        """mechanosensitive channel current"""
        if p > self.p_th:
            return self.k_p * (p - self.p_th)
        else:
            return 0

    def update_acc(self):
        """update spikes effect"""
        acc = 0
        for t_spike in self.spike_train:
            acc += self.r_d * np.exp( - (self.t - t_spike) / self.tau_d)
            acc += - self.r_h * np.exp( - (self.t - t_spike) / self.tau_h)
        return acc

    def update_p(self, p, acc):
        """update pressure"""
        p = p + self.dt * (self.r_p_inc + acc)
        return p

    def step(self):
        """step function"""

        v = self.v_train[-1]
        p = self.p_train[-1]
        i_m = self.i_ext_train[-1]

        # Update potential
        v = self.update_v(v, i_m)
        self.v_train.append(v)

        # Update spikes effect
        acc = self.update_acc()
        self.acc_train.append(acc)

        # Update pressure
        p = self.update_p(p, acc)
        self.p_train.append(p)

        # Update mechanosensitive current
        self.i_ext_train.append(self.i_ext(p))

        # Update time
        self.t += self.dt

    def disp(self):
        """display simulation results"""
        time_axis = np.arange(0, len(self.v_train) * self.dt, self.dt)
        # fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 8))
        fig, ax1 = plt.subplots(1, 1, figsize=(20, 2))
        ax1.plot(time_axis, self.v_train)
        ax1.set_ylabel('Potential (mV)')
        # ax2.plot(time_axis, self.acc_train)
        # ax2.set_ylabel('Spikes effects')
        # ax3.plot(time_axis, self.p_train)
        # ax3.set_ylabel('Pressure (Pa)')
        # ax4.plot(time_axis, self.i_ext_train)
        # ax4.set_ylabel('Current (nA)')
        # ax4.set_xlabel('Time (s)')
        ax1.set_title(str(round(self.r_d, 2)) + ', ' +
                      str(round(self.r_h, 2)) + ', ' +
                      str(self.tau_d) + ', ' +
                      str(self.tau_h))
        plt.show()

if __name__ == '__main__':
    neuron = MechSenseNeuron()
    neuron.dt = 0.02
    neuron.run(200)
    neuron.disp()


